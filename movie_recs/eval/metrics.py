from typing import List, Set
import math


def precision_at_k(predicted: List[int], actual: Set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    pred_k = predicted[:k]
    if not pred_k:
        return 0.0
    hits = sum(1 for p in pred_k if p in actual)
    return hits / float(len(pred_k))


def recall_at_k(predicted: List[int], actual: Set[int], k: int) -> float:
    if not actual:
        return 0.0
    pred_k = predicted[:k]
    hits = sum(1 for p in pred_k if p in actual)
    return hits / float(len(actual))


def ndcg_at_k(predicted: List[int], actual: Set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain for binary relevance."""
    pred_k = predicted[:k]
    dcg = 0.0
    for i, p in enumerate(pred_k, start=1):
        rel = 1.0 if p in actual else 0.0
        denom = math.log2(i + 1)
        dcg += rel / denom
    # ideal DCG
    ideal_hits = min(len(actual), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg


__all__ = ["precision_at_k", "recall_at_k", "ndcg_at_k"]
