from vectorstore.store import query_chunks
from reasoning.reason import reason
from config.queries import TYPE_QUERIES, SYSTEM_PROMPTS


for question in TYPE_QUERIES["rfq"]:

    results = query_chunks(
        "rfq_test",
        question
    )

    answer = reason(
        question,
        results["documents"][0],
        SYSTEM_PROMPTS["rfq"]
    )

    print("=" * 70)
    print(f"QUESTION: {question}")
    print(f"ANSWER: {answer}")
    print()