import os
import hashlib

import gradio as gr

from ingestion.pdf_reader import extract_text
from chunking.splitter import chunk_text
from embeddings.embedder import embed_chunks
from vectorstore.store import store_chunks, query_chunks
from reasoning.reason import reason
from config.queries import SYSTEM_PROMPTS


# =========================================================
# CURRENT DOCUMENT STATE
# =========================================================

CURRENT_DOCUMENT_ID = None
CURRENT_DOCUMENT_NAME = None


# =========================================================
# DOCUMENT ID
# =========================================================

def create_document_id(file_path):
    """
    Create a unique and safe document ID.

    The filename alone is not enough because two different
    documents can have the same filename.

    We combine:
        - filename
        - file size
        - file modification time

    and generate a short SHA256 hash.
    """

    filename = os.path.basename(file_path)

    file_size = os.path.getsize(file_path)

    modified_time = os.path.getmtime(file_path)

    identifier = (
        f"{filename}|"
        f"{file_size}|"
        f"{modified_time}"
    )

    file_hash = hashlib.sha256(
        identifier.encode("utf-8")
    ).hexdigest()[:12]

    base_name = os.path.splitext(
        filename
    )[0].lower()

    # Make filename safe for ChromaDB
    safe_name = "".join(
        character if character.isalnum() else "_"
        for character in base_name
    )

    safe_name = safe_name.strip("_")

    if not safe_name:
        safe_name = "document"

    return f"{safe_name}_{file_hash}"


# =========================================================
# PROCESS DOCUMENT
# =========================================================

def process_document(file_path):
    """
    Process either a PDF or TXT document.

    Pipeline:

        File
          ↓
        Text extraction
          ↓
        Chunking
          ↓
        Embeddings
          ↓
        ChromaDB
          ↓
        Ready for questions
    """

    global CURRENT_DOCUMENT_ID
    global CURRENT_DOCUMENT_NAME

    if not file_path:
        return (
            "Please upload a PDF or TXT document.",
            "No document loaded."
        )

    try:

        # -------------------------------------------------
        # 1. Validate file
        # -------------------------------------------------

        if not os.path.exists(file_path):
            return (
                "The selected file could not be found.",
                "No document loaded."
            )

        filename = os.path.basename(file_path)

        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension not in [".pdf", ".txt"]:
            return (
                "Unsupported file type. "
                "Please upload a PDF or TXT document.",
                "No document loaded."
            )

        # -------------------------------------------------
        # 2. Extract text
        # -------------------------------------------------

        if extension == ".pdf":

            pages = extract_text(
                file_path
            )

        else:

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:

                text = file.read().strip()

            if not text:
                return (
                    "The TXT document is empty.",
                    "No document loaded."
                )

            # Treat TXT as one page
            pages = [
                {
                    "page_number": 1,
                    "text": text
                }
            ]

        if not pages:
            return (
                "No text could be extracted from the document.",
                "No document loaded."
            )

        # -------------------------------------------------
        # 3. Check that text exists
        # -------------------------------------------------

        total_text = "\n".join(
            page.get("text", "")
            for page in pages
        ).strip()

        if not total_text:
            return (
                "The document contains no readable text.",
                "No document loaded."
            )

        # -------------------------------------------------
        # 4. Create chunks
        # -------------------------------------------------

        chunks = chunk_text(
            pages,
            "rfq"
        )

        if not chunks:
            return (
                "No chunks could be created from the document.",
                "No document loaded."
            )

        # -------------------------------------------------
        # 5. Create embeddings
        # -------------------------------------------------

        embeddings = embed_chunks(
            chunks
        )

        if not embeddings:
            return (
                "No embeddings could be generated.",
                "No document loaded."
            )

        # -------------------------------------------------
        # 6. Create UNIQUE document ID
        # -------------------------------------------------

        document_id = create_document_id(
            file_path
        )

        # -------------------------------------------------
        # 7. Store chunks + embeddings
        # -------------------------------------------------

        store_chunks(
            document_id=document_id,
            chunks=chunks,
            embeddings=embeddings
        )

        # -------------------------------------------------
        # 8. Update current document
        # -------------------------------------------------

        CURRENT_DOCUMENT_ID = document_id
        CURRENT_DOCUMENT_NAME = filename

        # -------------------------------------------------
        # 9. Processing status
        # -------------------------------------------------

        document_type = (
            extension
            .replace(".", "")
            .upper()
        )

        status = (
            "DOCUMENT PROCESSED SUCCESSFULLY\n\n"

            f"File: {filename}\n"
            f"Type: {document_type}\n"
            f"Pages: {len(pages)}\n"
            f"Chunks: {len(chunks)}\n"
            f"Embeddings: {len(embeddings)}\n\n"

            f"Document ID: {document_id}\n\n"

            "Status: READY"
        )

        current_status = (
            f"Current document:\n"
            f"{filename}\n\n"
            "Ready for questions."
        )

        return (
            status,
            current_status
        )

    except Exception as e:

        return (
            f"ERROR PROCESSING DOCUMENT\n\n{str(e)}",
            "Document was not loaded."
        )


# =========================================================
# CLEAR CURRENT DOCUMENT
# =========================================================

def clear_document():
    """
    Clear the active document from application state.

    This does NOT delete the document from ChromaDB.

    It only prevents DocuSense from using the previous
    document for new questions.
    """

    global CURRENT_DOCUMENT_ID
    global CURRENT_DOCUMENT_NAME

    CURRENT_DOCUMENT_ID = None
    CURRENT_DOCUMENT_NAME = None

    return (
        "CURRENT DOCUMENT CLEARED\n\n"
        "The stored document data remains in ChromaDB, "
        "but it is no longer active.",

        "No document loaded.",

        "",   # normal question
        "",   # answer
        "",   # sources
        "",   # conflict question
        ""    # conflict report
    )


# =========================================================
# RETRIEVE RELEVANT INFORMATION
# =========================================================

def get_relevant_information(question):
    """
    Retrieve the most relevant document chunks
    and their metadata.

    Day 7 performance optimization:
    retrieve 5 chunks instead of 10.

    This reduces the amount of context sent to
    the reasoning model and can improve response time.
    """

    if not CURRENT_DOCUMENT_ID:
        return None, None

    results = query_chunks(
        document_id=CURRENT_DOCUMENT_ID,
        question=question,
        k=5
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    return (
        documents,
        metadatas
    )


# =========================================================
# SOURCE PAGE HELPERS
# =========================================================

def get_source_pages(metadatas):
    """
    Extract unique page numbers from retrieved metadata.
    """

    source_pages = []

    for metadata in metadatas:

        page_number = metadata.get(
            "page_number"
        )

        if (
            page_number is not None
            and page_number not in source_pages
        ):
            source_pages.append(
                page_number
            )

    return sorted(
        source_pages
    )


def format_sources(metadatas):
    """
    Format source page information for the UI.
    """

    source_pages = get_source_pages(
        metadatas
    )

    if not source_pages:
        return "Sources: Not available"

    source_text = "Sources:\n"

    for page_number in source_pages:

        source_text += (
            f"- Page {page_number}\n"
        )

    return source_text


# =========================================================
# ASK DOCUSENSE
# =========================================================

def ask_docusense(question):
    """
    Answer a question using only retrieved
    document information.
    """

    if not question.strip():

        return (
            "Please enter a question.",
            ""
        )

    if not CURRENT_DOCUMENT_ID:

        return (
            "Please upload and process a document first.",
            ""
        )

    try:

        # -------------------------------------------------
        # Retrieve relevant chunks
        # -------------------------------------------------

        documents, metadatas = (
            get_relevant_information(
                question
            )
        )

        if not documents:

            return (
                "NOT FOUND IN DOCUMENT",
                ""
            )

        # -------------------------------------------------
        # Generate answer
        # -------------------------------------------------

        answer = reason(
            question=question,
            context_chunks=documents,
            system_prompt=SYSTEM_PROMPTS["rfq"]
        )

        # -------------------------------------------------
        # Format sources
        # -------------------------------------------------

        source_text = format_sources(
            metadatas
        )

        return (
            answer,
            source_text
        )

    except Exception as e:

        return (
            f"ERROR ANSWERING QUESTION\n\n{str(e)}",
            ""
        )


# =========================================================
# CONFLICT DETECTION
# =========================================================

def check_conflicts(question):
    """
    Analyze retrieved document excerpts for
    contradictory information.

    The conflict detector now has its own question
    field in Section 3.
    """

    if not question.strip():

        return "Please enter a conflict question."

    if not CURRENT_DOCUMENT_ID:

        return (
            "Please upload and process a document first."
        )

    try:

        # -------------------------------------------------
        # Retrieve relevant information
        # -------------------------------------------------

        documents, metadatas = (
            get_relevant_information(
                question
            )
        )

        if not documents:

            return "NOT FOUND IN DOCUMENT"

        # -------------------------------------------------
        # Conflict detection prompt
        # -------------------------------------------------

        conflict_prompt = (

            "You are a careful document auditor reviewing "
            "a procurement or tender document.\n\n"

            "Analyze ONLY the provided document excerpts.\n\n"

            "The user's question identifies the topic that "
            "should be checked for contradictions.\n\n"

            "Determine whether the excerpts contain "
            "conflicting information relevant to the "
            "user's question.\n\n"

            "IMPORTANT:\n"
            "Do not assume that two statements are "
            "conflicting simply because they discuss the "
            "same topic.\n\n"

            "Only report a conflict when the document "
            "contains materially different requirements, "
            "dates, values, durations, conditions, or "
            "other information that cannot both be true "
            "in the same context.\n\n"

            "If there is no conflict, respond exactly:\n"
            "NO CONFLICT DETECTED\n\n"

            "If there is a conflict, respond using "
            "this structure:\n\n"

            "⚠️ CONFLICT DETECTED\n\n"

            "Conflicting information:\n\n"

            "• Statement 1:\n"
            "  [accurately quote or summarize the first statement]\n\n"

            "• Statement 2:\n"
            "  [accurately quote or summarize the second statement]\n\n"

            "Recommendation:\n"
            "Verify the applicable clause before making "
            "a decision.\n\n"

            "Do not invent conflicts.\n"
            "Only report conflicts supported by the "
            "document excerpts."
        )

        # -------------------------------------------------
        # Ask reasoning model
        # -------------------------------------------------

        conflict_result = reason(
            question=question,
            context_chunks=documents,
            system_prompt=conflict_prompt
        )

        # -------------------------------------------------
        # Add source pages
        # -------------------------------------------------

        source_text = format_sources(
            metadatas
        )

        conflict_result += (
            "\n\n"
            + source_text
        )

        return conflict_result

    except Exception as e:

        return (
            f"ERROR CHECKING CONFLICTS\n\n{str(e)}"
        )


# =========================================================
# GRADIO INTERFACE
# =========================================================

with gr.Blocks(
    title="DocuSense"
) as app:

    # =====================================================
    # HEADER
    # =====================================================

    gr.Markdown(
        """
        # 📄 DocuSense

        ### AI-Powered RFQ Document Analysis

        Upload an RFQ document, process it, and ask
        questions using the document as the source of truth.
        """
    )

    # =====================================================
    # 1. UPLOAD DOCUMENT
    # =====================================================

    gr.Markdown(
        "## 1. Upload Document"
    )

    document_file = gr.File(
        label="Upload RFQ Document",
        file_types=[
            ".pdf",
            ".txt"
        ],
        type="filepath"
    )

    process_button = gr.Button(
        "Process Document"
    )

    clear_button = gr.Button(
        "Clear Current Document"
    )

    document_status = gr.Textbox(
        label="Document Status",
        lines=10
    )

    current_document = gr.Textbox(
        label="Current Document",
        value="No document loaded.",
        lines=3
    )

    # =====================================================
    # 2. QUESTION ANSWERING
    # =====================================================

    gr.Markdown(
        "## 2. Ask a Question"
    )

    question = gr.Textbox(
        label="Question",
        placeholder="What is the submission deadline?",
        lines=2
    )

    ask_button = gr.Button(
        "Ask DocuSense"
    )

    answer = gr.Textbox(
        label="Answer",
        lines=10
    )

    sources = gr.Textbox(
        label="Sources",
        lines=5
    )

    # =====================================================
    # 3. CONFLICT DETECTION
    # =====================================================

    gr.Markdown(
        "## 3. Conflict Detection"
    )

    conflict_question = gr.Textbox(
        label="Conflict Question",
        placeholder=(
            "Example: Are there conflicting contract durations "
            "in the document?"
        ),
        lines=2
    )

    conflict_button = gr.Button(
        "⚠️ Check for Conflicts"
    )

    conflict_report = gr.Textbox(
        label="Conflict Report",
        lines=14
    )

    # =====================================================
    # EVENT HANDLERS
    # =====================================================

    # Process document
    process_button.click(
        fn=process_document,
        inputs=document_file,
        outputs=[
            document_status,
            current_document
        ]
    )

    # Ask normal question
    ask_button.click(
        fn=ask_docusense,
        inputs=question,
        outputs=[
            answer,
            sources
        ]
    )

    # Check conflicts using dedicated conflict question
    conflict_button.click(
        fn=check_conflicts,
        inputs=conflict_question,
        outputs=conflict_report
    )

    # =====================================================
    # CLEAR DOCUMENT
    # =====================================================

    clear_button.click(
        fn=clear_document,
        inputs=[],
        outputs=[
            document_status,
            current_document,
            question,
            answer,
            sources,
            conflict_question,
            conflict_report
        ]
    )


# =========================================================
# START APPLICATION
# =========================================================

if __name__ == "__main__":

    app.launch()