from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(pages: list[dict], doc_type: str) -> list[dict]:
    size, overlap = 800, 150

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " "]
    )

    chunks = []

    for page in pages:
        for i, piece in enumerate(splitter.split_text(page["text"])):
            chunks.append({
                "chunk_id": f"p{page['page_number']}_c{i}",
                "page_number": page["page_number"],
                "text": piece
            })

    return chunks