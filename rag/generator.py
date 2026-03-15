"""
Answer generation using the OpenAI Chat Completions API directly.

The prompt template grounds the LLM strictly in the retrieved email context
and instructs it to cite sources. Temperature is set to 0 for deterministic,
factual responses.
"""

import os
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about a collection of emails.
You will be given relevant email excerpts as context. Follow these rules strictly:

1. Answer ONLY based on the provided email context. Do not use outside knowledge.
2. Cite which email(s) your answer comes from by mentioning the sender name and subject line.
3. If the context does not contain enough information to answer the question, say "I don't have enough information in the provided emails to answer that question."
4. Be concise and direct. Summarize when appropriate.
5. If multiple emails are relevant, synthesize the information and cite all sources."""


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(api_key=api_key)


def _format_context(results: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    parts = []
    for i, result in enumerate(results, 1):
        meta = result["metadata"]
        parts.append(
            f"--- Email {i} ---\n"
            f"Source: {meta['source']}\n"
            f"Subject: {meta['subject']}\n"
            f"From: {meta['sender_name']} <{meta['sender_email']}>\n"
            f"To: {meta['receiver_name']} <{meta['receiver_email']}>\n"
            f"Relevance Score: {result.get('score', 'N/A'):.4f}\n\n"
            f"{result['text']}\n"
        )
    return "\n".join(parts)


def generate_answer(question: str, results: list[dict]) -> str:
    """
    Generate an answer to a question using retrieved email chunks as context.
    Calls the OpenAI Chat Completions API directly.
    """
    if not results:
        return "No relevant emails were found for your question."

    context = _format_context(results)

    user_message = (
        f"Context (retrieved emails):\n\n{context}\n\n"
        f"---\n\n"
        f"Question: {question}\n\n"
        f"Answer based only on the emails above:"
    )

    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    return response.choices[0].message.content


def get_rag_response(
    question: str,
    index,
    metadata: list[dict],
    k: int = 5,
) -> dict:
    """
    End-to-end RAG: embed query -> retrieve top-k -> generate answer.
    Returns a dict with 'answer', 'sources', and 'question' keys.
    """
    from .embedder import embed_query
    from .retriever import search

    query_vec = embed_query(question)
    results = search(query_vec, index, metadata, k=k)
    answer = generate_answer(question, results)

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "source": r["metadata"]["source"],
                "subject": r["metadata"]["subject"],
                "sender": r["metadata"]["sender_name"],
                "score": r.get("score", 0),
            }
            for r in results
        ],
    }
