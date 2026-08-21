TYPE_QUERIES = {
    "rfq": [
        "What is the submission deadline?",
        "What are the eligibility criteria?",
        "What is the evaluation methodology?",
        "What is the estimated contract value?",
        "What are the submission format requirements?"
    ]
}


SYSTEM_PROMPTS = {
    "rfq": (
        "You are an experienced procurement advisor reviewing a tender document "
        "on behalf of a bidding company. Answer only using the provided document "
        "excerpts. If the answer is not present, respond exactly with "
        "'NOT FOUND IN DOCUMENT'. Be precise about dates, numbers, and requirements."
    )
}