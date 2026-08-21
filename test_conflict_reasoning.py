import ollama

from vectorstore.store import query_chunks


question = "What is the contract duration?"

results = query_chunks(
    "conflict_test",
    question,
    k=5
)

context_chunks = results["documents"][0]

context = "\n\n---\n\n".join(context_chunks)

system_prompt = """
You are a careful document analyst.

Answer questions ONLY using the provided document excerpts.

If different parts of the document provide conflicting information,
DO NOT choose one answer without explanation.

Instead:
1. Identify the conflicting statements.
2. State where each statement appears if section information is available.
3. Clearly say that a conflict exists.
4. Do not invent information.

If there is no conflict, provide the best supported answer.
"""

prompt = f"""
Document excerpts:

{context}

Question:
{question}

Analyze the excerpts carefully and answer the question.
"""

response = ollama.chat(
    model="llama3",
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("QUESTION:")
print(question)

print("\nAI ANALYSIS:")
print(response["message"]["content"])