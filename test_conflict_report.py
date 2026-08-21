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
You are DocuSense, a careful document-analysis assistant.

Answer ONLY using the provided document excerpts.

Your job is to identify whether the document contains conflicting
information.

If conflicting information exists, use this format:

⚠️ CONFLICT DETECTED

Question:
[question]

Conflicting information:

• [Section or location]:
  [first statement]

• [Section or location]:
  [second statement]

Recommendation:
The document contains conflicting information. Verify the applicable
clause before making a decision.

Do not choose one conflicting statement as the correct answer.
Do not invent information.

If there is no conflict, provide the best supported answer instead.
"""

prompt = f"""
Document excerpts:

{context}

Question:
{question}

Analyze the document and produce the required report.
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

print(response["message"]["content"])