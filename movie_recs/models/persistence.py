from pathlib import Path
import numpy as np
import joblib
from typing import Tuple, List, Any
from scipy import sparse


def save_svd(path: str, u: np.ndarray, s: np.ndarray, vt: np.ndarray, user_ids: List[int], item_ids: List[int], gmean: Any):
    """Save SVD factors to a .npz file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Convert lists to numpy arrays for storage
    np.savez_compressed(p, u=u, s=s, vt=vt, user_ids=np.array(user_ids, dtype=np.int64), item_ids=np.array(item_ids, dtype=np.int64), gmean=np.asarray(gmean))


def load_svd(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[int], np.ndarray]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"SVD file not found: {p}")
    data = np.load(p, allow_pickle=True)
    u = data["u"]
    s = data["s"]
    vt = data["vt"]
    user_ids = data["user_ids"].tolist()
    item_ids = data["item_ids"].tolist()
    gmean = data["gmean"]
    return u, s, vt, user_ids, item_ids, gmean


def save_tfidf(path_prefix: str, vectorizer: Any, item_vectors: Any):
    """Save TF-IDF vectorizer (joblib) and item_vectors (sparse .npz) using given prefix.

    Writes: {prefix}_vectorizer.joblib and {prefix}_item_vectors.npz
    """
    vp = Path(path_prefix)
    vp.parent.mkdir(parents=True, exist_ok=True)
    vec_path = vp.with_name(vp.name + "_vectorizer.joblib")
    items_path = vp.with_name(vp.name + "_item_vectors.npz")
    joblib.dump(vectorizer, vec_path)
    # item_vectors may be sparse matrix
    if sparse.issparse(item_vectors):
        sparse.save_npz(items_path, item_vectors)
    else:
        # save dense numpy array
        np.savez_compressed(items_path, item_vectors=item_vectors)


def load_tfidf(path_prefix: str):
    vp = Path(path_prefix)
    vec_path = vp.with_name(vp.name + "_vectorizer.joblib")
    items_path = vp.with_name(vp.name + "_item_vectors.npz")
    if not vec_path.exists():
        raise FileNotFoundError(f"Vectorizer file not found: {vec_path}")
    vectorizer = joblib.load(vec_path)
    # Try load sparse first
    if items_path.exists():
        try:
            item_vectors = sparse.load_npz(items_path)
        except Exception:
            data = np.load(items_path, allow_pickle=True)
            item_vectors = data["item_vectors"]
    else:
        raise FileNotFoundError(f"Item vectors file not found: {items_path}")
    return vectorizer, item_vectors


__all__ = ["save_svd", "load_svd", "save_tfidf", "load_tfidf"]
