from vectorstore.store import query_chunks
from reasoning.reason import reason
from config.queries import SYSTEM_PROMPTS


question = "What is the submission deadline?"

results = query_chunks(
    "rfq_test",
    question
)

answer = reason(
    question,
    results["documents"][0],
    SYSTEM_PROMPTS["rfq"]
)

print("\nQuestion:")
print(question)

print("\nAnswer:")
print(answer)