import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from movie_recs.models.persistence import save_svd, save_tfidf
from movie_recs.models.loader import load_models_and_recommend
from movie_recs.models.content import build_item_tfidf


def _small_dataset():
    movies = pd.DataFrame([
        {"movieId": 1, "title": "Action A", "genres": "Action"},
        {"movieId": 2, "title": "Action B", "genres": "Action"},
        {"movieId": 3, "title": "RomCom", "genres": "Romance|Comedy"},
    ])
    ratings = pd.DataFrame([
        {"userId": 1, "movieId": 1, "rating": 5.0},
        {"userId": 1, "movieId": 3, "rating": 2.0},
        {"userId": 2, "movieId": 2, "rating": 4.0},
    ])
    return ratings, movies


def test_load_models_recommend():
    ratings, movies = _small_dataset()
    # create simple svd factors (fake) and tfidf artifacts and persist them
    u = np.random.RandomState(0).rand(2, 2)
    s = np.array([1.0, 0.5])
    vt = np.random.RandomState(1).rand(2, 3)
    user_ids = [1, 2]
    item_ids = [1, 2, 3]
    gmean = np.array([3.0])

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        svd_path = p / "svd_factors.npz"
        save_svd(str(svd_path), u, s, vt, user_ids, item_ids, gmean)
        # build tfidf and persist
        vec, item_vectors, ids = build_item_tfidf(movies)
        tf_prefix = p / "tfidf"
        save_tfidf(str(tf_prefix), vec, item_vectors)

        # test svd
        recs_svd = load_models_and_recommend(str(p), "svd", 1, ratings, movies, top_n=2)
        assert isinstance(recs_svd, pd.DataFrame)

        # test content
        recs_ct = load_models_and_recommend(str(p), "content", 1, ratings, movies, top_n=2)
        assert isinstance(recs_ct, pd.DataFrame)

        # test hybrid
        recs_h = load_models_and_recommend(str(p), "hybrid", 1, ratings, movies, top_n=2, alpha=0.6)
        assert isinstance(recs_h, pd.DataFrame)
