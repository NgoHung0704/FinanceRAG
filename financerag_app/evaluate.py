"""Retrieval metrics for FinanceRAG (NDCG@k, Recall@k, MRR).

Pure standard-library implementation (uses ``math.log2``) matching the
convention used across the notebooks: ``DCG = sum (2^rel - 1) / log2(rank + 1)``.
Kept dependency-free so it can validate the retriever without the ML stack.
"""

from __future__ import annotations

from math import log2
from typing import Dict, List


def ndcg_at_k(retrieved: List[str], relevance: Dict[str, int], k: int = 10) -> float:
    """NDCG@k for a single query.

    ``retrieved`` is the ranked list of doc ids; ``relevance`` maps doc id ->
    graded relevance (0 = irrelevant).
    """
    if not relevance:
        return 0.0
    gains = [relevance.get(doc_id, 0) for doc_id in retrieved[:k]]
    dcg = sum((2 ** g - 1) / log2(i + 2) for i, g in enumerate(gains) if g > 0)

    ideal = sorted((g for g in relevance.values() if g > 0), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / log2(i + 2) for i, g in enumerate(ideal))
    if idcg == 0:
        return 0.0
    return max(0.0, min(1.0, dcg / idcg))


def recall_at_k(retrieved: List[str], relevance: Dict[str, int], k: int = 10) -> float:
    relevant = {d for d, s in relevance.items() if s > 0}
    if not relevant:
        return 0.0
    hits = sum(1 for d in retrieved[:k] if d in relevant)
    return hits / len(relevant)


def mrr(retrieved: List[str], relevance: Dict[str, int]) -> float:
    relevant = {d for d, s in relevance.items() if s > 0}
    for i, d in enumerate(retrieved, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def evaluate(
    results: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10,
) -> Dict[str, float]:
    """Average metrics over all queries that have relevance judgements."""
    ndcgs, recalls, mrrs = [], [], []
    for qid, relevance in qrels.items():
        retrieved = results.get(qid, [])
        ndcgs.append(ndcg_at_k(retrieved, relevance, k))
        recalls.append(recall_at_k(retrieved, relevance, k))
        mrrs.append(mrr(retrieved, relevance))
    n = len(ndcgs) or 1
    return {
        f"NDCG@{k}": sum(ndcgs) / n,
        f"Recall@{k}": sum(recalls) / n,
        "MRR": sum(mrrs) / n,
        "num_queries": len(ndcgs),
    }
