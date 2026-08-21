from vectorstore.store import query_chunks


question = "What is the contract duration?"

results = query_chunks(
    "conflict_test",
    question,
    k=5
)

print("QUESTION:")
print(question)

print("\nRELEVANT INFORMATION:")

for i, document in enumerate(results["documents"][0]):
    metadata = results["metadatas"][0][i]

    print("\n" + "=" * 60)
    print(f"Result {i + 1}")
    print(f"Page: {metadata['page_number']}")
    print(document)