"""
FAISS-based vector retrieval module.

Uses IndexFlatIP (inner product) on L2-normalized vectors, which is equivalent
to cosine similarity. The index and associated chunk metadata are persisted to
disk so embedding only needs to happen once.
"""

import os
import json
import numpy as np
import faiss

from .chunker import Chunk

INDEX_FILE = "index/faiss.index"
META_FILE = "index/metadata.json"


def build_index(chunks: list[Chunk], embeddings: np.ndarray, index_dir: str = "index") -> faiss.Index:
    """
    Build a FAISS inner-product index from chunk embeddings and persist to disk.
    """
    os.makedirs(index_dir, exist_ok=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    index_path = os.path.join(index_dir, "faiss.index")
    meta_path = os.path.join(index_dir, "metadata.json")

    faiss.write_index(index, index_path)

    metadata = [{"text": c.text, "metadata": c.metadata} for c in chunks]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return index


def load_index(index_dir: str = "index") -> tuple[faiss.Index, list[dict]]:
    """Load a persisted FAISS index and its chunk metadata from disk."""
    index_path = os.path.join(index_dir, "faiss.index")
    meta_path = os.path.join(index_dir, "metadata.json")

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Index not found at {index_dir}/. Run build_index.py first."
        )

    index = faiss.read_index(index_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


def search(
    query_embedding: np.ndarray,
    index: faiss.Index,
    metadata: list[dict],
    k: int = 5,
) -> list[dict]:
    """
    Search the FAISS index for the top-k most similar chunks.
    Returns a list of dicts with 'text', 'metadata', and 'score' keys.
    """
    scores, indices = index.search(query_embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        entry = metadata[idx].copy()
        entry["score"] = float(score)
        results.append(entry)

    return results
