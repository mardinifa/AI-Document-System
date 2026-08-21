import ollama


def reason(
    question: str,
    context_chunks: list[str],
    system_prompt: str
) -> str:

    context = "\n\n---\n\n".join(context_chunks)

    prompt = (
        f"Document excerpts:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )

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

    return response["message"]["content"]