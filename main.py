#!/usr/bin/env python3
"""
CLI entry point for the Mini RAG system.

Usage:
    python main.py "Who sent emails about budget approvals?"
    python main.py                          # interactive mode
"""

import sys
import time
from dotenv import load_dotenv

load_dotenv()

from rag.chunker import load_emails, chunk_documents
from rag.embedder import embed_texts
from rag.retriever import build_index, load_index
from rag.generator import get_rag_response


def _build_index():
    """Build the FAISS index from email documents."""
    print("=" * 50)
    print("Building FAISS index from email documents")
    print("=" * 50)

    print("\n[1/3] Loading and chunking emails...")
    t0 = time.time()
    emails = load_emails("emails")
    chunks = chunk_documents(emails, chunk_size=500, chunk_overlap=50)
    print(f"  Loaded {len(emails)} emails -> {len(chunks)} chunks ({time.time() - t0:.2f}s)")

    print("\n[2/3] Generating embeddings (OpenAI text-embedding-3-small)...")
    t0 = time.time()
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    print(f"  Embedded {len(texts)} chunks -> shape {embeddings.shape} ({time.time() - t0:.2f}s)")

    print("\n[3/3] Building and persisting FAISS index...")
    t0 = time.time()
    index = build_index(chunks, embeddings)
    print(f"  Index built with {index.ntotal} vectors ({time.time() - t0:.2f}s)")
    print(f"\nIndex saved to index/\n")


def print_response(response: dict):
    print(f"\nQuestion: {response['question']}")
    print("-" * 50)
    print(f"\n{response['answer']}\n")
    print("-" * 50)
    print("Sources:")
    for i, src in enumerate(response["sources"], 1):
        print(f"  {i}. [{src['subject']}] from {src['sender']} ({src['source']}) "
              f"- score: {src['score']:.4f}")
    print()


def interactive_mode(index, metadata):
    print("Mini RAG System - Interactive Mode")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("Ask a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question or question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        response = get_rag_response(question, index, metadata, k=5)
        print_response(response)


def main():
    try:
        index, metadata = load_index()
    except FileNotFoundError:
        print("Index not found. Building it now...\n")
        _build_index()
        index, metadata = load_index()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        response = get_rag_response(question, index, metadata, k=5)
        print_response(response)
    else:
        interactive_mode(index, metadata)


if __name__ == "__main__":
    main()
