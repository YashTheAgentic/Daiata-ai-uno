#!/usr/bin/env python3
"""
Evaluation script for the Mini RAG system.

Runs 14 handcrafted queries covering all 9 email topics from the dataset.
For each query, computes retrieval precision@k and hit@k metrics, then
generates an answer so reviewers can visually verify faithfulness.

Usage:
    python eval.py                  # retrieval metrics only (fast, no LLM calls)
    python eval.py --with-answers   # also generate and display LLM answers
"""

import argparse
import json
from dotenv import load_dotenv

load_dotenv()

from rag.retriever import load_index
from rag.embedder import embed_query
from rag.retriever import search
from rag.generator import generate_answer

# 14 queries across all 9 email topics from the dataset spec.
# "expected_subjects" lists subjects that count as relevant hits.
TEST_QUERIES = [
    # --- Budget Approval ---
    {
        "query": "Who sent emails about budget approvals?",
        "expected_subjects": ["Budget Approval"],
    },
    {
        "query": "What budget proposals need approval and what do they cover?",
        "expected_subjects": ["Budget Approval"],
    },
    # --- Training Opportunity ---
    {
        "query": "What training opportunities or workshops are available for the team?",
        "expected_subjects": ["Training Opportunity"],
    },
    # --- Technical Issue ---
    {
        "query": "Are there any technical issues or system problems reported?",
        "expected_subjects": ["Technical Issue"],
    },
    {
        "query": "What production incidents have occurred and what workarounds are in place?",
        "expected_subjects": ["Technical Issue"],
    },
    # --- Meeting Request ---
    {
        "query": "Who wants to schedule meetings and what are they about?",
        "expected_subjects": ["Meeting Request"],
    },
    # --- Client Feedback ---
    {
        "query": "What feedback has been received from clients?",
        "expected_subjects": ["Client Feedback"],
    },
    {
        "query": "Which clients praised our work and what improvements did they suggest?",
        "expected_subjects": ["Client Feedback"],
    },
    # --- Project Update ---
    {
        "query": "What are the latest project status updates?",
        "expected_subjects": ["Project Update"],
    },
    {
        "query": "Which projects are on track and which hit roadblocks?",
        "expected_subjects": ["Project Update"],
    },
    # --- Team Announcement ---
    {
        "query": "Are there any team restructuring or organizational announcements?",
        "expected_subjects": ["Team Announcement"],
    },
    # --- Deadline Extension ---
    {
        "query": "Who requested deadline extensions and why?",
        "expected_subjects": ["Deadline Extension"],
    },
    {
        "query": "Which projects need more time to complete?",
        "expected_subjects": ["Deadline Extension"],
    },
    # --- Vendor Proposal ---
    {
        "query": "What vendor proposals have been received and what do they offer?",
        "expected_subjects": ["Vendor Proposal"],
    },
]


def precision_at_k(results: list[dict], expected_subjects: list[str], k: int) -> float:
    """Fraction of top-k results whose subject matches an expected subject."""
    top_k = results[:k]
    if not top_k:
        return 0.0
    hits = sum(
        1 for r in top_k
        if r["metadata"]["subject"] in expected_subjects
    )
    return hits / k


def hit_at_k(results: list[dict], expected_subjects: list[str], k: int) -> bool:
    """Whether at least one expected subject appears in top-k results."""
    return any(
        r["metadata"]["subject"] in expected_subjects
        for r in results[:k]
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Mini RAG system")
    parser.add_argument(
        "--with-answers", action="store_true",
        help="Also generate LLM answers for each query (slower, costs API calls)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Mini RAG System - Evaluation")
    print(f"Queries: {len(TEST_QUERIES)} | Topics covered: 9")
    print(f"Generation: {'enabled' if args.with_answers else 'disabled (use --with-answers)'}")
    print("=" * 70)

    index, metadata = load_index()
    print(f"Index loaded: {index.ntotal} vectors\n")

    all_results = []

    for i, test in enumerate(TEST_QUERIES, 1):
        query = test["query"]
        expected = test["expected_subjects"]

        query_vec = embed_query(query)
        results = search(query_vec, index, metadata, k=5)

        p_at_3 = precision_at_k(results, expected, k=3)
        p_at_5 = precision_at_k(results, expected, k=5)
        h_at_3 = hit_at_k(results, expected, k=3)
        h_at_5 = hit_at_k(results, expected, k=5)

        retrieved_subjects = [r["metadata"]["subject"] for r in results]
        retrieved_sources = [
            f"{r['metadata']['sender_name']} ({r['metadata']['source']}, score={r.get('score', 0):.4f})"
            for r in results
        ]

        result = {
            "query": query,
            "expected_subjects": expected,
            "retrieved_subjects": retrieved_subjects,
            "retrieved_sources": retrieved_sources,
            "precision_at_3": round(p_at_3, 4),
            "precision_at_5": round(p_at_5, 4),
            "hit_at_3": h_at_3,
            "hit_at_5": h_at_5,
        }

        print(f"[{i}/{len(TEST_QUERIES)}] {query}")
        print(f"  Expected:     {expected}")
        print(f"  Retrieved:    {retrieved_subjects}")
        print(f"  Sources:      {retrieved_sources[:3]}...")
        print(f"  Precision@3:  {p_at_3:.0%}  |  Precision@5:  {p_at_5:.0%}  "
              f"|  Hit@3: {'Y' if h_at_3 else 'N'}  |  Hit@5: {'Y' if h_at_5 else 'N'}")

        if args.with_answers:
            answer = generate_answer(query, results)
            result["generated_answer"] = answer
            print(f"\n  --- Generated Answer ---")
            for line in answer.split("\n"):
                print(f"  {line}")
            print(f"  --- End Answer ---")

        print()
        all_results.append(result)

    # Aggregate metrics
    n = len(all_results)
    avg_p3 = sum(r["precision_at_3"] for r in all_results) / n
    avg_p5 = sum(r["precision_at_5"] for r in all_results) / n
    hit3_rate = sum(1 for r in all_results if r["hit_at_3"]) / n
    hit5_rate = sum(1 for r in all_results if r["hit_at_5"]) / n

    summary = {
        "num_queries": n,
        "topics_covered": 9,
        "avg_precision_at_3": round(avg_p3, 4),
        "avg_precision_at_5": round(avg_p5, 4),
        "hit_rate_at_3": round(hit3_rate, 4),
        "hit_rate_at_5": round(hit5_rate, 4),
        "generation_included": args.with_answers,
    }

    print("=" * 70)
    print("AGGREGATE RESULTS")
    print(f"  Queries:          {n} across 9 topics")
    print(f"  Avg Precision@3:  {avg_p3:.1%}")
    print(f"  Avg Precision@5:  {avg_p5:.1%}")
    print(f"  Hit Rate@3:       {hit3_rate:.1%}")
    print(f"  Hit Rate@5:       {hit5_rate:.1%}")
    print("=" * 70)

    output = {"summary": summary, "details": all_results}
    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to results.json")


if __name__ == "__main__":
    main()
