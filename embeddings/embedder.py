from sentence_transformers import SentenceTransformer


_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_chunks(chunks: list[dict]) -> list[list[float]]:
    texts = [c["text"] for c in chunks]
    return _model.encode(texts).tolist()