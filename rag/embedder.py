"""
Embedding module using the OpenAI API directly (no framework wrappers).

Uses text-embedding-3-small (1536 dimensions) for a good balance of quality,
speed, and cost. Embeddings are L2-normalized so inner product == cosine similarity.
Batches requests in groups of 100 to stay within API rate limits.
"""

import os
import numpy as np
from openai import OpenAI

MODEL = "text-embedding-3-small"
DIMENSIONS = 1536
BATCH_SIZE = 100


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(api_key=api_key)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize vectors so inner product equals cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return vectors / norms


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts using OpenAI's embedding API.
    Returns an (N, DIMENSIONS) numpy array of L2-normalized vectors.
    """
    client = _get_client()
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(model=MODEL, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    vectors = np.array(all_embeddings, dtype=np.float32)
    return _normalize(vectors)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns a (1, DIMENSIONS) normalized vector."""
    client = _get_client()
    response = client.embeddings.create(model=MODEL, input=[query])
    vector = np.array([response.data[0].embedding], dtype=np.float32)
    return _normalize(vector)
