from ingestion.pdf_reader import extract_text
from chunking.splitter import chunk_text
from embeddings.embedder import embed_chunks
from vectorstore.store import store_chunks, query_chunks


pages = extract_text("data/benchmarks/rfq_sample.pdf")

chunks = chunk_text(pages, "rfq")

embeddings = embed_chunks(chunks)

store_chunks(
    "rfq_test",
    chunks,
    embeddings
)

results = query_chunks(
    "rfq_test",
    "What is the submission deadline?"
)

print("Search result:")
print(results["documents"][0])