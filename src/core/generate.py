import os
from typing import List, Dict, Any, Optional

try:
    import google.generativeai as genai
except ImportError:
    genai = None


def generate_answer_from_retrieval(
    question: str,
    retrieved_metadata: List[Dict[str, Any]],
    model: str = "gemini-pro",
    google_api_key: Optional[str] = None,
    max_output_tokens: int = 256,
    temperature: float = 0.2,
) -> str:
    """
    Generate an answer to a user question based on retrieved image metadata.
    Uses Google Gemini Pro or a placeholder if not available.

    Args:
        question: The user's question.
        retrieved_metadata: List of metadata dicts for the retrieved images.
        model: LLM model name.
        google_api_key: API key for Google AI (optional, will use env if not provided).

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

    api_key = google_api_key or os.getenv("GOOGLE_API_KEY")

    if genai and api_key:
        genai.configure(api_key=api_key)

        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

        gemini_model = genai.GenerativeModel(model, generation_config=generation_config)
        response = gemini_model.generate_content(prompt)

        return response.text.strip()
    else:
        # Placeholder: just echo the context and question
        return f"[LLM not available] Question: {question}\nContext:\n{context}"
