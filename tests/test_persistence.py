import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from movie_recs.models.persistence import save_svd, load_svd, save_tfidf, load_tfidf
from movie_recs.models.content import build_item_tfidf


def test_save_and_load_svd():
    u = np.random.RandomState(0).rand(3, 2)
    s = np.array([2.0, 1.0])
    vt = np.random.RandomState(1).rand(2, 4)
    user_ids = [1, 2, 3]
    item_ids = [10, 20, 30, 40]
    gmean = np.array([3.5])

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "svd_test.npz"
        save_svd(str(p), u, s, vt, user_ids, item_ids, gmean)
        lu, ls, lvt, luser_ids, litem_ids, lg = load_svd(str(p))
        assert lu.shape == u.shape
        assert ls.shape == s.shape
        assert lvt.shape == vt.shape
        assert luser_ids == user_ids
        assert litem_ids == item_ids


def test_save_and_load_tfidf():
    movies = pd.DataFrame([
        {"movieId": 1, "title": "Alpha", "genres": "Action"},
        {"movieId": 2, "title": "Beta", "genres": "Drama"},
    ])
    vec, item_vectors, item_ids = build_item_tfidf(movies)
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "tf_test"
        save_tfidf(str(prefix), vec, item_vectors)
        lvec, litems = load_tfidf(str(prefix))
        # vectorizer should transform same corpus to same shape
        X1 = vec.transform(["Alpha Action", "Beta Drama"])  # shape (2, d)
        X2 = lvec.transform(["Alpha Action", "Beta Drama"])  # shape (2, d)
        assert X1.shape == X2.shape
        # item vectors shape must match
        if hasattr(item_vectors, "shape"):
            assert item_vectors.shape[0] == litems.shape[0]
