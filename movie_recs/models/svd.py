from typing import Tuple, List
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


def _build_user_item_matrix(ratings: pd.DataFrame) -> Tuple[csr_matrix, List[int], List[int]]:
    """Build a sparse user-item matrix (users x items).

    Returns matrix, user_ids, item_ids where user_ids[i] is the userId for row i
    and item_ids[j] is the movieId for column j.
    """
    user_ids = np.sort(ratings.userId.unique())
    item_ids = np.sort(ratings.movieId.unique())

    user_idx = {u: i for i, u in enumerate(user_ids)}
    item_idx = {m: i for i, m in enumerate(item_ids)}

    rows = ratings["userId"].map(user_idx).to_numpy()
    cols = ratings["movieId"].map(item_idx).to_numpy()
    data = ratings["rating"].to_numpy(dtype=float)

    mat = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(item_ids)))
    return mat, list(user_ids), list(item_ids)


def train_svd(ratings: pd.DataFrame, n_factors: int = 20, center: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[int], np.ndarray]:
    """Train an SVD (truncated) on the user-item ratings matrix.

    Returns (u, s, vt, user_ids, item_ids, global_mean)
    - u: user factors (n_users x k)
    - s: singular values (k,)
    - vt: item factors (k x n_items)
    - user_ids, item_ids: lists mapping indices to ids
    - global_mean: scalar mean used for centering
    """
    mat, user_ids, item_ids = _build_user_item_matrix(ratings)
    # Convert to dense for svds? svds accepts sparse input.
    # Centering by subtracting global mean or user mean can improve quality.
    if center:
        # compute global mean of non-zero entries
        nonzero = mat.data
        if nonzero.size == 0:
            global_mean = 0.0
        else:
            global_mean = float(nonzero.mean())
        mat_centered = mat.copy().astype(float)
        mat_centered.data = mat_centered.data - global_mean
        u, s, vt = svds(mat_centered, k=min(n_factors, min(mat_centered.shape)-1))
    else:
        global_mean = 0.0
        u, s, vt = svds(mat.astype(float), k=min(n_factors, min(mat.shape)-1))

    # svds returns singular values in ascending order; reverse for convenience
    u = u[:, ::-1]
    s = s[::-1]
    vt = vt[::-1, :]

    return u, s, vt, user_ids, item_ids, np.array([global_mean])


def recommend_svd(user_id: int, u: np.ndarray, s: np.ndarray, vt: np.ndarray, user_ids: List[int], item_ids: List[int], ratings: pd.DataFrame, movies: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Produce top-N recommendations for user_id using trained SVD factors.

    If user_id is unknown, returns popularity baseline (most rated).
    """
    user_to_idx = {u_id: i for i, u_id in enumerate(user_ids)}
    if user_id not in user_to_idx:
        # fallback to popularity
        agg = ratings.groupby("movieId").rating.agg(["count", "mean"]).rename(columns={"mean": "avg_rating"})
        agg = agg.sort_values(["count", "avg_rating"], ascending=[False, False])
        top = agg.reset_index().head(top_n)
        return top.merge(movies, on="movieId")[["movieId", "title", "count", "avg_rating"]].rename(columns={"count": "score"})

    uid = user_to_idx[user_id]
    # reconstruct predicted ratings via u @ diag(s) @ vt
    user_vec = u[uid, :]
    scores = user_vec @ np.diag(s) @ vt
    # scores is length n_items

    # Mask already seen items (user has rated)
    user_rated = ratings[ratings.userId == user_id].movieId.unique()
    mask = np.isin(item_ids, user_rated)
    scores_masked = np.array(scores).ravel()
    scores_masked[mask] = -np.inf

    top_idx = np.argsort(scores_masked)[-top_n:][::-1]
    rec_item_ids = [item_ids[i] for i in top_idx if scores_masked[i] != -np.inf]
    rec_scores = [float(scores_masked[i]) for i in top_idx if scores_masked[i] != -np.inf]

    recs = pd.DataFrame({"movieId": rec_item_ids, "score": rec_scores})
    recs = recs.merge(movies, on="movieId")[ ["movieId", "title", "score"] ]
    return recs


__all__ = ["train_svd", "recommend_svd"]
