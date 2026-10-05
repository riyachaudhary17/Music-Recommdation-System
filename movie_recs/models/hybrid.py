from typing import List, Dict, Optional
import numpy as np
import pandas as pd

from .svd import train_svd, recommend_svd
from .baselines import item_based_recommendations
from .content import build_item_tfidf, recommend_content


def _normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=float)
    minv = vals.min()
    maxv = vals.max()
    if maxv == minv:
        # all equal -> give uniform score
        return {k: 1.0 for k in scores}
    norm = {k: (v - minv) / (maxv - minv) for k, v in scores.items()}
    return norm


def hybrid_recommend(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    top_n: int = 10,
    alpha: float = 0.5,
    cf_method: str = "svd",
    n_factors: int = 20,
    candidate_pool: int = 100,
    vectorizer=None,
    item_vectors=None,
) -> pd.DataFrame:
    """Combine CF and content scores into a hybrid ranking.

    - cf_method: 'svd' or 'item' (item-based)
    - alpha: weight for CF score (0..1); content weight = 1-alpha
    - candidate_pool: number of top items to retrieve from each method before merging
    - vectorizer/item_vectors: optional prebuilt content artefacts
    """
    # Get candidate lists
    cf_scores = {}
    content_scores = {}

    # CF recommendations
    if cf_method == "svd":
        try:
            u, s, vt, user_ids, item_ids, gmean = train_svd(ratings, n_factors=n_factors)
            cf_recs = recommend_svd(user_id, u, s, vt, user_ids, item_ids, ratings, movies, top_n=candidate_pool)
            cf_scores = {int(r.movieId): float(r.score) for _, r in cf_recs.iterrows()}
        except Exception:
            # fallback to item-based
            cf_recs = item_based_recommendations(user_id, ratings, movies, top_n=candidate_pool)
            cf_scores = {int(r.movieId): float(r.score) for _, r in cf_recs.iterrows()}
    else:
        cf_recs = item_based_recommendations(user_id, ratings, movies, top_n=candidate_pool)
        cf_scores = {int(r.movieId): float(r.score) for _, r in cf_recs.iterrows()}

    # Content recommendations
    if vectorizer is None or item_vectors is None:
        try:
            vec, item_vecs, item_ids = build_item_tfidf(movies)
        except Exception:
            vec = None
            item_vecs = None
    else:
        vec = vectorizer
        item_vecs = item_vectors

    try:
        content_recs = recommend_content(user_id, ratings, movies, top_n=candidate_pool, vectorizer=vec, item_vectors=item_vecs)
        content_scores = {int(r.movieId): float(r.score) for _, r in content_recs.iterrows()}
    except Exception:
        content_scores = {}

    # Normalize scores
    cf_norm = _normalize_scores(cf_scores)
    content_norm = _normalize_scores(content_scores)

    # Merge candidate ids
    all_ids = set(cf_norm.keys()) | set(content_norm.keys())
    combined = {}
    for mid in all_ids:
        s_cf = cf_norm.get(mid, 0.0)
        s_ct = content_norm.get(mid, 0.0)
        combined[mid] = alpha * s_cf + (1.0 - alpha) * s_ct

    if not combined:
        return pd.DataFrame(columns=["movieId", "title", "score"])

    # Sort and return top_n
    sorted_items = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_n]
    rec_item_ids = [mid for mid, _ in sorted_items]
    rec_scores = [float(score) for _, score in sorted_items]
    recs = pd.DataFrame({"movieId": rec_item_ids, "score": rec_scores})
    recs = recs.merge(movies, on="movieId")[ ["movieId", "title", "score"] ]
    return recs


__all__ = ["hybrid_recommend"]
