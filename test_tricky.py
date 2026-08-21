from vectorstore.store import query_chunks
from reasoning.reason import reason
from config.queries import SYSTEM_PROMPTS


question = "What is the contract value, and who is the successful bidder?"

results = query_chunks(
    "rfq_test",
    question
)

answer = reason(
    question,
    results["documents"][0],
    SYSTEM_PROMPTS["rfq"]
)

print("QUESTION:")
print(question)

print("\nAI ANSWER:")
print(answer)