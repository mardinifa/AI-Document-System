from chunking.splitter import chunk_text
from embeddings.embedder import embed_chunks
from vectorstore.store import store_chunks
from pathlib import Path


file_path = Path("data/benchmarks/conflict_sample.txt")

text = file_path.read_text(encoding="utf-8")

pages = [
    {
        "page_number": 1,
        "text": text
    }
]

chunks = chunk_text(pages, "contract")

embeddings = embed_chunks(chunks)

collection = store_chunks(
    "conflict_test",
    chunks,
    embeddings
)

print("Conflict document stored successfully.")
print(f"Number of chunks stored: {len(chunks)}")