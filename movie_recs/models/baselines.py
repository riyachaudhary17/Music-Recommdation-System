from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def top_n_popular(ratings: pd.DataFrame, movies: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return top-n popular movies by number of ratings then average rating.

    Args:
        ratings: DataFrame with columns [userId, movieId, rating]
        movies: DataFrame with columns [movieId, title, ...]
        n: number of movies to return

    Returns:
        DataFrame of top-n movies with columns [movieId, title, count, avg_rating]
    """
    agg = ratings.groupby("movieId").rating.agg(["count", "mean"]).rename(columns={"mean": "avg_rating"})
    agg = agg.sort_values(["count", "avg_rating"], ascending=[False, False])
    top = agg.reset_index().head(n)
    return top.merge(movies, on="movieId")[["movieId", "title", "count", "avg_rating"]]


def _build_item_user_matrix(ratings: pd.DataFrame) -> Tuple[csr_matrix, List[int], List[int]]:
    """Build a sparse item-user matrix (items x users).

    Returns matrix, item_ids, user_ids where item_ids[i] is the movieId for row i
    and user_ids[j] is the userId for column j.
    """
    # Ensure consistent ordering
    user_ids = np.sort(ratings.userId.unique())
    item_ids = np.sort(ratings.movieId.unique())

    user_idx = {u: i for i, u in enumerate(user_ids)}
    item_idx = {m: i for i, m in enumerate(item_ids)}

    rows = ratings["movieId"].map(item_idx).to_numpy()
    cols = ratings["userId"].map(user_idx).to_numpy()
    data = ratings["rating"].to_numpy(dtype=float)

    mat = csr_matrix((data, (rows, cols)), shape=(len(item_ids), len(user_ids)))
    return mat, list(item_ids), list(user_ids)


def item_based_recommendations(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    top_n: int = 10,
    k: int = 20,
) -> pd.DataFrame:
    """Recommend top-N items for a user using item-item collaborative filtering.

    Args:
        user_id: target user id
        ratings: DataFrame with [userId, movieId, rating]
        movies: DataFrame with [movieId, title]
        top_n: number of recommendations to return
        k: number of similar items to consider per item rated by user

    Returns:
        DataFrame with recommended movieId, title, score
    """
    # Build matrix
    mat, item_ids, user_ids = _build_item_user_matrix(ratings)

    # Compute item-item similarity (cosine on rows of item-user matrix)
    # Use dense cosine_similarity on small-ish item dimension
    sim = cosine_similarity(mat)

    # Map ids to indices
    item_to_idx = {m: i for i, m in enumerate(item_ids)}
    user_to_idx = {u: i for i, u in enumerate(user_ids)}

    if user_id not in user_to_idx:
        # New user: return top popular
        popular = top_n_popular(ratings, movies, n=top_n)
        return popular.rename(columns={"count": "score", "avg_rating": "avg_rating"})[["movieId", "title", "score"]]

    uidx = user_to_idx[user_id]

    # User's ratings vector across items
    user_ratings_by_item = mat[:, uidx].toarray().ravel()  # length = n_items
    rated_indices = np.where(user_ratings_by_item > 0)[0]

    if len(rated_indices) == 0:
        return top_n_popular(ratings, movies, n=top_n).rename(columns={"count": "score", "avg_rating": "avg_rating"})[["movieId", "title", "score"]]

    # Score unseen items: weighted sum of similarities * user rating
    n_items = sim.shape[0]
    scores = np.zeros(n_items, dtype=float)
    sim_sums = np.zeros(n_items, dtype=float)

    for idx in rated_indices:
        # take top-k similar items to idx (including itself)
        sims = sim[idx]
        # zero out self to avoid recommending same item with inflated score
        sims[idx] = 0.0
        top_k_idx = np.argsort(sims)[-k:]
        sim_vals = sims[top_k_idx]
        scores[top_k_idx] += sim_vals * user_ratings_by_item[idx]
        sim_sums[top_k_idx] += np.abs(sim_vals)

    # Normalize
    with np.errstate(divide="ignore", invalid="ignore"):
        final_scores = np.divide(scores, sim_sums)
        final_scores[np.isnan(final_scores)] = 0.0

    # Mask items the user has already rated
    final_scores[rated_indices] = -np.inf

    # Get top_n indices
    top_idx = np.argsort(final_scores)[-top_n:][::-1]

    rec_item_ids = [item_ids[i] for i in top_idx if final_scores[i] != -np.inf]
    rec_scores = [float(final_scores[i]) for i in top_idx if final_scores[i] != -np.inf]

    recs = pd.DataFrame({"movieId": rec_item_ids, "score": rec_scores})
    recs = recs.merge(movies, on="movieId")[ ["movieId", "title", "score"] ]
    return recs


__all__ = ["top_n_popular", "item_based_recommendations"]
