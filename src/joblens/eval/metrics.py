"""Retrieval metrics. Three of them, because they disagree and that is useful.

recall@k  did we find the relevant postings at all, anywhere in the top k
MRR       how high was the first relevant one
nDCG@k    how good is the whole ordering, discounted by position

A change can lift recall and drop MRR: it found one more relevant posting and
buried it at rank 9. Reporting one number hides that. All three are reported
and the README shows all three.

Graded relevance is supported because the golden set uses it: a posting can be
exactly what the query asked for (2) or merely adjacent (1). recall and MRR
treat anything above 0 as relevant; nDCG uses the grade.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[int], relevant: dict[int, int], k: int) -> float:
    """Share of the relevant postings that appear in the top k."""
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    found = sum(1 for pid, grade in relevant.items() if grade > 0 and pid in top)
    total = sum(1 for grade in relevant.values() if grade > 0)
    return found / total if total else 0.0


def precision_at_k(retrieved: Sequence[int], relevant: dict[int, int], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for pid in top if relevant.get(pid, 0) > 0) / len(top)


def reciprocal_rank(retrieved: Sequence[int], relevant: dict[int, int]) -> float:
    """1/rank of the first relevant result, 0 if there is none."""
    for rank, posting_id in enumerate(retrieved, start=1):
        if relevant.get(posting_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def dcg(grades: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(retrieved: Sequence[int], relevant: dict[int, int], k: int) -> float:
    """Discounted gain over the ideal ordering of the same judgements."""
    gains = [float(relevant.get(pid, 0)) for pid in retrieved[:k]]
    ideal = sorted((float(g) for g in relevant.values()), reverse=True)[:k]
    best = dcg(ideal)
    return dcg(gains) / best if best else 0.0


def score_run(
    retrieved: Sequence[int], relevant: dict[int, int], ks: Sequence[int] = (5, 10)
) -> dict[str, float]:
    """Every metric for one query."""
    result: dict[str, float] = {"mrr": reciprocal_rank(retrieved, relevant)}
    for k in ks:
        result[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        result[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        result[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
    return result


def mean_scores(per_query: Sequence[dict[str, float]]) -> dict[str, float]:
    """Macro average: every query counts the same regardless of how many
    relevant postings it has, so one broad query cannot dominate the report."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: sum(row[k] for row in per_query) / len(per_query) for k in keys}
