from pathlib import Path


file_path = Path("data/benchmarks/conflict_sample.txt")

text = file_path.read_text(encoding="utf-8")

print("CONFLICT TEST DOCUMENT")
print("=" * 60)
print(text)