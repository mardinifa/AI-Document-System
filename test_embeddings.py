from ingestion.pdf_reader import extract_text
from chunking.splitter import chunk_text
from embeddings.embedder import embed_chunks


pages = extract_text("data/benchmarks/rfq_sample.pdf")

chunks = chunk_text(pages, "rfq")

embeddings = embed_chunks(chunks)

print(f"Created {len(embeddings)} embeddings")
print(f"Numbers in first embedding: {len(embeddings[0])}")
print(f"First few numbers: {embeddings[0][:5]}")
