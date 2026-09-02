from typing import List, Tuple
import google.generativeai as genai

from src.pdf_processor import Chunk

MODEL_NAME = "gemini-3.5-flash-lite"
MAX_WORDS_SINGLE_PASS = 120_000


class GeminiAPIError(Exception):
    """Raised when a Gemini API call fails, with a user-friendly message."""


def _get_model(api_key: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME)


def _generate(model, prompt: str) -> str:
    """Calls the model and turns any failure into a friendly GeminiAPIError
    instead of letting a raw exception crash the caller."""
    try:
        response = model.generate_content(prompt)
    except Exception as exc:
        raise GeminiAPIError(f"The Gemini API request failed: {exc}") from exc
    return response.text.strip()


def answer_question(
    question: str,
    retrieved_chunks: List[Tuple[Chunk, float]],
    api_key: str,
) -> dict:
    """
    Generates a grounded, natural-language answer to the question
    using only the retrieved chunks as context.

    Args:
        question: The user's question.
        retrieved_chunks: Output of VectorStore.search().
        api_key: The caller's Gemini API key.

    Returns:
        A dict with 'answer' (str) and 'pages' (sorted list of ints).
    """
    if not retrieved_chunks:
        return {"answer": "I couldn't find anything relevant in the document.", "pages": []}

    context_parts = []
    pages = []
    for chunk, _ in retrieved_chunks:
        context_parts.append(f"[Page {chunk.page_number}]\n{chunk.text}")
        pages.append(chunk.page_number)
    context = "\n\n".join(context_parts)

    prompt = (
        "You are answering a question using ONLY the context below, which "
        "was extracted from a PDF document. If the answer isn't contained "
        "in the context, say so honestly instead of guessing.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Give a clear, direct answer in 1-3 sentences."
    )

    model = _get_model(api_key)
    answer = _generate(model, prompt)

    return {
        "answer": answer,
        "pages": sorted(set(pages)),
    }


def summarize_chunks(chunks: List[Chunk], api_key: str) -> str:
    """
    Summarizes the whole document with length scaled to the
    document's own length and density, rather than a fixed
    max_length. Short documents get a couple of sentences; long or
    dense ones get multiple paragraphs.

    Args:
        chunks: All chunks extracted from the document.
        api_key: The caller's Gemini API key.

    Returns:
        A summary string.
    """
    if not chunks:
        return "No text could be extracted from this document."

    full_text = " ".join(c.text for c in chunks)
    word_count = len(full_text.split())

    if word_count > MAX_WORDS_SINGLE_PASS:
        # Extremely long document: fall back to map-reduce so we
        # don't exceed the model's context window.
        return _map_reduce_summarize(chunks, api_key)

    model = _get_model(api_key)
    prompt = (
        "Summarize the following document in your own words, in clear "
        "paragraph form (not bullet points). Scale the length of your "
        "summary to the length and complexity of the document itself: "
        "a short or simple document deserves a short summary (a couple "
        "of sentences), while a long or dense document deserves a "
        "longer one (multiple paragraphs). Don't pad it artificially, "
        "and don't just restate the first page.\n\n"
        f"Document ({word_count} words):\n{full_text}"
    )
    return _generate(model, prompt)


def _map_reduce_summarize(chunks: List[Chunk], api_key: str, section_words: int = 20_000) -> str:
    """
    For documents too large for a single call: summarizes the
    document in sections, then summarizes those summaries together
    into one coherent final summary.
    """
    model = _get_model(api_key)

    sections: List[str] = []
    current: List[str] = []
    current_len = 0
    for chunk in chunks:
        current.append(chunk.text)
        current_len += len(chunk.text.split())
        if current_len >= section_words:
            sections.append(" ".join(current))
            current, current_len = [], 0
    if current:
        sections.append(" ".join(current))

    section_summaries = []
    for section in sections:
        prompt = (
            "Summarize this section of a larger document in 2-4 sentences, "
            "focusing on the key points:\n\n" + section
        )
        section_summaries.append(_generate(model, prompt))

    combined = "\n\n".join(section_summaries)
    final_prompt = (
        "Below are section-by-section summaries of a long document. "
        "Combine them into a single coherent summary in paragraph form, "
        "scaled to the overall length and complexity of the document. "
        "Remove redundancy between sections.\n\n" + combined
    )
    return _generate(model, final_prompt)
