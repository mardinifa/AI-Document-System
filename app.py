import os
import gradio as gr

from ingestion.pdf_reader import extract_text
from chunking.splitter import chunk_text
from embeddings.embedder import embed_chunks
from vectorstore.store import store_chunks, query_chunks
from reasoning.reason import reason
from config.queries import SYSTEM_PROMPTS


CURRENT_DOCUMENT_ID = None


def process_document(file_path):
    """
    Process either a PDF or TXT document.
    """

    global CURRENT_DOCUMENT_ID

    if not file_path:
        return "Please upload a PDF or TXT document."

    try:
        # -------------------------------------------------
        # 1. Extract text
        # -------------------------------------------------

        filename = os.path.basename(file_path)
        extension = os.path.splitext(filename)[1].lower()

        if extension == ".pdf":

            pages = extract_text(file_path)

        elif extension == ".txt":

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:

                text = file.read().strip()

            if not text:
                return "The TXT document is empty."

            # Treat the TXT document as one page
            pages = [
                {
                    "page_number": 1,
                    "text": text
                }
            ]

        else:

            return "Unsupported file type. Please upload a PDF or TXT document."

        if not pages:
            return "No text could be extracted from the document."

        # -------------------------------------------------
        # 2. Create chunks
        # -------------------------------------------------

        chunks = chunk_text(
            pages,
            "rfq"
        )

        if not chunks:
            return "No chunks could be created from the document."

        # -------------------------------------------------
        # 3. Create embeddings
        # -------------------------------------------------

        embeddings = embed_chunks(chunks)

        # -------------------------------------------------
        # 4. Create document ID
        # -------------------------------------------------

        document_id = os.path.splitext(filename)[0].lower()

        document_id = (
            document_id
            .replace(" ", "_")
            .replace("-", "_")
        )

        # -------------------------------------------------
        # 5. Store chunks and embeddings
        # -------------------------------------------------

        store_chunks(
            document_id=document_id,
            chunks=chunks,
            embeddings=embeddings
        )

        CURRENT_DOCUMENT_ID = document_id

        # -------------------------------------------------
        # 6. Return processing status
        # -------------------------------------------------

        return (
            f"Document processed successfully.\n\n"
            f"File: {filename}\n"
            f"Type: {extension.upper().replace('.', '')}\n"
            f"Pages: {len(pages)}\n"
            f"Chunks: {len(chunks)}\n"
            f"Embeddings: {len(embeddings)}"
        )

    except Exception as e:

        return f"Error processing document: {e}"


def get_relevant_information(question):
    """
    Retrieve the most relevant document chunks
    and their metadata.
    """

    if not CURRENT_DOCUMENT_ID:
        return None, None

    results = query_chunks(
        document_id=CURRENT_DOCUMENT_ID,
        question=question,
        k=10
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    return documents, metadatas


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
            source_pages.append(page_number)

    return sorted(source_pages)


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

        # Retrieve relevant chunks
        documents, metadatas = get_relevant_information(
            question
        )

        if not documents:

            return (
                "NOT FOUND IN DOCUMENT",
                ""
            )

        # Ask reasoning model
        answer = reason(
            question=question,
            context_chunks=documents,
            system_prompt=SYSTEM_PROMPTS["rfq"]
        )

        # Format sources
        source_text = format_sources(
            metadatas
        )

        return (
            answer,
            source_text
        )

    except Exception as e:

        return (
            f"Error answering question: {e}",
            ""
        )


def check_conflicts(question):
    """
    Analyze retrieved document excerpts for
    contradictory information.
    """

    if not question.strip():

        return "Please enter a question."

    if not CURRENT_DOCUMENT_ID:

        return (
            "Please upload and process a document first."
        )

    try:

        # Retrieve relevant information
        documents, metadatas = get_relevant_information(
            question
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

            "Determine whether the excerpts contain "
            "conflicting information relevant to the "
            "user's question.\n\n"

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

        # Ask reasoning model
        conflict_result = reason(
            question=question,
            context_chunks=documents,
            system_prompt=conflict_prompt
        )

        # Add sources
        source_text = format_sources(
            metadatas
        )

        conflict_result += (
            "\n\n" + source_text
        )

        return conflict_result

    except Exception as e:

        return (
            f"Error checking conflicts: {e}"
        )


# =========================================================
# GRADIO INTERFACE
# =========================================================

with gr.Blocks(
    title="DocuSense"
) as app:

    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    gr.Markdown(
        """
        # 📄 DocuSense

        ### AI-Powered RFQ Document Analysis

        Upload an RFQ document, process it, and ask
        questions using the document as the source of truth.
        """
    )

    # -----------------------------------------------------
    # DOCUMENT UPLOAD
    # -----------------------------------------------------

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

    document_status = gr.Textbox(
        label="Document Status",
        lines=6
    )

    process_button.click(
        fn=process_document,
        inputs=document_file,
        outputs=document_status
    )

    # -----------------------------------------------------
    # QUESTION ANSWERING
    # -----------------------------------------------------

    gr.Markdown(
        "## 2. Ask a Question"
    )

    question = gr.Textbox(
        label="Question",
        placeholder="What is the submission deadline?"
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

    ask_button.click(
        fn=ask_docusense,
        inputs=question,
        outputs=[
            answer,
            sources
        ]
    )

    # -----------------------------------------------------
    # CONFLICT DETECTION
    # -----------------------------------------------------

    gr.Markdown(
        "## 3. Conflict Detection"
    )

    conflict_button = gr.Button(
        "⚠️ Check for Conflicts"
    )

    conflict_report = gr.Textbox(
        label="Conflict Report",
        lines=14
    )

    conflict_button.click(
        fn=check_conflicts,
        inputs=question,
        outputs=conflict_report
    )


# =========================================================
# START APPLICATION
# =========================================================

if __name__ == "__main__":

    app.launch()