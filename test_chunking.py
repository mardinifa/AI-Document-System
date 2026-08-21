from ingestion.pdf_reader import extract_text
from chunking.splitter import chunk_text


pages = extract_text("data/benchmarks/rfq_sample.pdf")

chunks = chunk_text(pages, "rfq")

print(f"Created {len(chunks)} chunks")
print("\nFirst chunk:")
print(chunks[0]["text"])