import os
from typing import List, Dict, Any

try:
    import openai
except ImportError:
    openai = None


def generate_answer_from_retrieval(
    question: str,
    retrieved_metadata: List[Dict[str, Any]],
    model: str = "gpt-4o-mini",
    openai_api_key: str = None,
) -> str:
    """
    Generate an answer to a user question based on retrieved image metadata (and optionally vectors).
    Uses OpenAI GPT-4o or a placeholder if not available.

    Args:
        question: The user's question.
        retrieved_metadata: List of metadata dicts for the retrieved images.
        model: LLM model name.
        openai_api_key: API key for OpenAI (optional, will use env if not provided).

    Returns:
        Generated answer as a string.
    """
    # Compose context from metadata
    context = "\n".join(
        [f"Image {i+1}: {meta}" for i, meta in enumerate(retrieved_metadata)]
    )
    prompt = (
        f"You are an expert assistant. The user has asked: '{question}'.\n"
        f"Here is information about the most relevant images retrieved from a database:\n{context}\n"
        "Based on this information, provide a helpful, concise answer to the user's question."
    )

    if openai and (openai_api_key or os.getenv("OPENAI_API_KEY")):
        client = openai.OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=256,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    else:
        # Placeholder: just echo the context and question
        return f"[LLM not available] Question: {question}\nContext:\n{context}"
