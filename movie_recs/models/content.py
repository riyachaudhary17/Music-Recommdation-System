from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _build_corpus(movies: pd.DataFrame, fields: List[str] = None) -> List[str]:
    if fields is None:
        fields = ["title", "genres"]
    def _safe_join(row):
        parts = []
        for f in fields:
            val = row.get(f, "")
            if pd.isna(val):
                val = ""
            parts.append(str(val))
        return " ".join(parts)
    corpus = movies.apply(_safe_join, axis=1).tolist()
    return corpus


def build_item_tfidf(movies: pd.DataFrame, fields: List[str] = None, max_features: int = 5000) -> Tuple[TfidfVectorizer, np.ndarray, List[int]]:
    """Build TF-IDF vectors for movie items.

    Returns (vectorizer, item_vectors (n_items x dim), item_ids)
    """
    item_ids = movies["movieId"].tolist()
    corpus = _build_corpus(movies, fields)
    vec = TfidfVectorizer(stop_words="english", max_features=max_features)
    X = vec.fit_transform(corpus)  # shape (n_items, dim)
    return vec, X, item_ids


def recommend_content(user_id: int, ratings: pd.DataFrame, movies: pd.DataFrame, top_n: int = 10, vectorizer=None, item_vectors=None, fields: List[str] = None) -> pd.DataFrame:
    """Recommend top-N items for a user using content-based filtering.

    Approach: build TF-IDF item vectors (title+genres), compute user profile as weighted
    average of item vectors for items the user rated, then score all items by cosine
    similarity to the user profile and return top-N unseen items.
    """
    # Build or accept prebuilt item_vectors
    if item_vectors is None or vectorizer is None:
        vectorizer, item_vectors, item_ids = build_item_tfidf(movies, fields)
    else:
        item_ids = movies["movieId"].tolist()

    # Map item id to row index
    id_to_idx = {mid: i for i, mid in enumerate(item_ids)}

    user_ratings = ratings[ratings.userId == user_id]
    if user_ratings.empty:
        # fall back to popularity
        agg = ratings.groupby("movieId").rating.agg(["count", "mean"]).rename(columns={"mean": "avg_rating"})
        agg = agg.sort_values(["count", "avg_rating"], ascending=[False, False])
        top = agg.reset_index().head(top_n)
        return top.merge(movies, on="movieId")[ ["movieId", "title", "count", "avg_rating"] ].rename(columns={"count": "score"})

    # Collect vectors and weights
    vecs = []
    weights = []
    for _, r in user_ratings.iterrows():
        mid = r.movieId
        if mid not in id_to_idx:
            continue
        idx = id_to_idx[mid]
        v = item_vectors[idx]
        # item_vectors may be sparse matrix row; convert to array
        if hasattr(v, "toarray"):
            v = v.toarray().ravel()
        elif hasattr(v, "A1"):
            v = np.array(v).ravel()
        else:
            v = np.asarray(v).ravel()
        vecs.append(v)
        weights.append(float(r.rating))

    if not vecs:
        # no overlap; fallback
        agg = ratings.groupby("movieId").rating.agg(["count", "mean"]).rename(columns={"mean": "avg_rating"})
        agg = agg.sort_values(["count", "avg_rating"], ascending=[False, False])
        top = agg.reset_index().head(top_n)
        return top.merge(movies, on="movieId")[ ["movieId", "title", "count", "avg_rating"] ].rename(columns={"count": "score"})

    vecs = np.vstack(vecs)
    weights = np.array(weights)
    # Normalize weights to sum to 1
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()

    # Weighted average user profile
    user_profile = weights @ vecs  # shape (dim,)
    # compute cosine similarity against all item vectors
    # item_vectors may be sparse matrix
    if hasattr(item_vectors, "toarray"):
        item_mat = item_vectors.toarray()
    else:
        item_mat = np.asarray(item_vectors)

    # Normalize vectors
    def _norm(x):
        norm = np.linalg.norm(x)
        return x / norm if norm > 0 else x

    user_profile = _norm(user_profile)
    item_mat_norm = np.vstack([_norm(row) for row in item_mat])

    sims = item_mat_norm @ user_profile

    # Mask seen items
    seen = set(user_ratings.movieId.tolist())
    scores = sims.copy()
    for mid in seen:
        if mid in id_to_idx:
            scores[id_to_idx[mid]] = -np.inf

    top_idx = np.argsort(scores)[-top_n:][::-1]
    rec_item_ids = [item_ids[i] for i in top_idx if scores[i] != -np.inf]
    rec_scores = [float(scores[i]) for i in top_idx if scores[i] != -np.inf]

    recs = pd.DataFrame({"movieId": rec_item_ids, "score": rec_scores})
    recs = recs.merge(movies, on="movieId")[ ["movieId", "title", "score"] ]
    return recs


__all__ = ["build_item_tfidf", "recommend_content"]
