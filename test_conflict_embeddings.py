from chunking.splitter import chunk_text
from embeddings.embedder import embed_chunks
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

print(f"Created {len(embeddings)} embeddings")
print(f"Embedding size: {len(embeddings[0])}")
print(f"First few numbers: {embeddings[0][:5]}")