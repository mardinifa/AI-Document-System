from chunking.splitter import chunk_text
from pathlib import Path


file_path = Path("data/benchmarks/conflict_sample.txt")

text = file_path.read_text(encoding="utf-8")

# Give the text a simple page number because this is a test TXT file.
pages = [
    {
        "page_number": 1,
        "text": text
    }
]

chunks = chunk_text(pages, "contract")

print(f"Created {len(chunks)} chunks")

for chunk in chunks:
    print("\n" + "=" * 60)
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Page: {chunk['page_number']}")
    print(chunk["text"])