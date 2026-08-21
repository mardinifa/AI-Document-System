import chromadb

from embeddings.embedder import _model


client = chromadb.PersistentClient(path="data/chroma")


def store_chunks(
    document_id: str,
    chunks: list[dict],
    embeddings: list[list[float]]
):
    collection = client.get_or_create_collection(
        name=document_id
    )

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[
            {"page_number": c["page_number"]}
            for c in chunks
        ]
    )

    return collection


def query_chunks(
    document_id: str,
    question: str,
    k: int = 5
):
    collection = client.get_collection(
        name=document_id
    )

    q_embedding = _model.encode(
        [question]
    ).tolist()[0]

    return collection.query(
        query_embeddings=[q_embedding],
        n_results=k
    )