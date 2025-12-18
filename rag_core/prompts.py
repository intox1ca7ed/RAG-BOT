SYSTEM_PROMPT = (
    "You are a helpful assistant answering strictly from provided context. "
    "If the context is insufficient, say so and recommend a handoff."
)


def build_user_prompt(question: str, context: str) -> str:
    return (
        "Answer the question using only the provided context.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "If the answer is not in the context, say you don't have enough information."
    )
