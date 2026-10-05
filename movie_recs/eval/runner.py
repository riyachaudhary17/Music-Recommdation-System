from typing import Callable, Dict, List
import pandas as pd
from .metrics import precision_at_k, recall_at_k, ndcg_at_k


def evaluate_topk(
    recommender: Callable[[int, pd.DataFrame, pd.DataFrame, int], pd.DataFrame],
    train: pd.DataFrame,
    test: pd.DataFrame,
    movies: pd.DataFrame,
    users: List[int] = None,
    k: int = 10,
) -> Dict[str, float]:
    """Evaluate a recommender on test users using top-k ranking metrics.

    recommender(user_id, train_ratings, movies, top_n) -> DataFrame with movieId column

    Returns dict with average precision, recall, ndcg at k.
    """
    if users is None:
        users = test.userId.unique().tolist()

    precisions = []
    recalls = []
    ndcgs = []

    for u in users:
        actual = set(test[test.userId == u].movieId.tolist())
        if not actual:
            continue
        preds_df = recommender(u, train, movies, top_n=k)
        if preds_df is None or preds_df.empty:
            precisions.append(0.0)
            recalls.append(0.0)
            ndcgs.append(0.0)
            continue
        preds = preds_df.movieId.tolist()
        precisions.append(precision_at_k(preds, actual, k))
        recalls.append(recall_at_k(preds, actual, k))
        ndcgs.append(ndcg_at_k(preds, actual, k))

    n = len(precisions)
    if n == 0:
        return {"precision": 0.0, "recall": 0.0, "ndcg": 0.0}

    return {"precision": sum(precisions) / n, "recall": sum(recalls) / n, "ndcg": sum(ndcgs) / n}


__all__ = ["evaluate_topk"]
