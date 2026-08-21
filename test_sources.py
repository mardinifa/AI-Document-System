from vectorstore.store import query_chunks


question = "What is the closing date and time for this tender?"

results = query_chunks(
    "rfq_test",
    question
)

print("QUESTION:")
print(question)

print("\nRELEVANT SOURCES:")

for i, document in enumerate(results["documents"][0]):
    metadata = results["metadatas"][0][i]

    print(f"\nSource {i + 1}")
    print(f"Page: {metadata['page_number']}")
    print(f"Text: {document[:300]}...")