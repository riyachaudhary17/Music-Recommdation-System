from pathlib import Path
from typing import Optional, Dict
import numpy as np
import pandas as pd

from .persistence import load_svd, load_tfidf
from .svd import recommend_svd
from .content import recommend_content


def _normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=float)
    minv = vals.min()
    maxv = vals.max()
    if maxv == minv:
        return {k: 1.0 for k in scores}
    norm = {k: (v - minv) / (maxv - minv) for k, v in scores.items()}
    return norm


def load_models_and_recommend(models_dir: str, method: str, user_id: int, ratings: pd.DataFrame, movies: pd.DataFrame, top_n: int = 10, alpha: float = 0.5):
    """Load persisted models from models_dir and produce recommendations.

    method: 'svd', 'content', or 'hybrid'
    """
    md = Path(models_dir)
    svd_path = md / "svd_factors.npz"
    tf_prefix = md / "tfidf"

    svd_loaded = None
    tf_loaded = None

    if svd_path.exists():
        try:
            u, s, vt, user_ids, item_ids, gmean = load_svd(str(svd_path))
            svd_loaded = dict(u=u, s=s, vt=vt, user_ids=user_ids, item_ids=item_ids, gmean=gmean)
        except Exception:
            svd_loaded = None

    try:
        if (tf_prefix.with_name(tf_prefix.name + "_vectorizer.joblib")).exists():
            vec, item_vectors = load_tfidf(str(tf_prefix))
            tf_loaded = dict(vectorizer=vec, item_vectors=item_vectors)
    except Exception:
        tf_loaded = None

    if method == "svd":
        if svd_loaded is None:
            raise FileNotFoundError("SVD model not found in models_dir")
        recs = recommend_svd(user_id, svd_loaded["u"], svd_loaded["s"], svd_loaded["vt"], svd_loaded["user_ids"], svd_loaded["item_ids"], ratings, movies, top_n=top_n)
        return recs

    if method == "content":
        if tf_loaded is None:
            raise FileNotFoundError("TF-IDF artifacts not found in models_dir")
        recs = recommend_content(user_id, ratings, movies, top_n=top_n, vectorizer=tf_loaded["vectorizer"], item_vectors=tf_loaded["item_vectors"])
        return recs

    if method == "hybrid":
        # Need both or fallback
        cf_scores = {}
        content_scores = {}
        if svd_loaded is not None:
            cf_recs = recommend_svd(user_id, svd_loaded["u"], svd_loaded["s"], svd_loaded["vt"], svd_loaded["user_ids"], svd_loaded["item_ids"], ratings, movies, top_n=top_n*5)
            cf_scores = {int(r.movieId): float(r.score) for _, r in cf_recs.iterrows()}
        if tf_loaded is not None:
            content_recs = recommend_content(user_id, ratings, movies, top_n=top_n*5, vectorizer=tf_loaded["vectorizer"], item_vectors=tf_loaded["item_vectors"])
            content_scores = {int(r.movieId): float(r.score) for _, r in content_recs.iterrows()}

        if not cf_scores and not content_scores:
            raise FileNotFoundError("No persisted models available for hybrid recommendation")

        cf_norm = _normalize_scores(cf_scores)
        content_norm = _normalize_scores(content_scores)

        all_ids = set(cf_norm.keys()) | set(content_norm.keys())
        combined = {}
        for mid in all_ids:
            s_cf = cf_norm.get(mid, 0.0)
            s_ct = content_norm.get(mid, 0.0)
            combined[mid] = alpha * s_cf + (1.0 - alpha) * s_ct

        sorted_items = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_n]
        rec_item_ids = [mid for mid, _ in sorted_items]
        rec_scores = [float(score) for _, score in sorted_items]
        recs = pd.DataFrame({"movieId": rec_item_ids, "score": rec_scores})
        recs = recs.merge(movies, on="movieId")[ ["movieId", "title", "score"] ]
        return recs

    raise ValueError(f"Unknown method: {method}")


__all__ = ["load_models_and_recommend"]
