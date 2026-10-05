import pandas as pd
from movie_recs.eval.split import train_test_split_by_user
from movie_recs.eval.metrics import precision_at_k, recall_at_k, ndcg_at_k
from movie_recs.eval.runner import evaluate_topk


def _small_dataset():
    ratings = pd.DataFrame(
        [
            {"userId": 1, "movieId": 10, "rating": 5.0},
            {"userId": 1, "movieId": 20, "rating": 3.0},
            {"userId": 1, "movieId": 30, "rating": 4.0},
            {"userId": 2, "movieId": 10, "rating": 4.0},
            {"userId": 2, "movieId": 40, "rating": 4.0},
            {"userId": 3, "movieId": 20, "rating": 5.0},
        ]
    )
    movies = pd.DataFrame(
        [
            {"movieId": 10, "title": "Movie A"},
            {"movieId": 20, "title": "Movie B"},
            {"movieId": 30, "title": "Movie C"},
            {"movieId": 40, "title": "Movie D"},
        ]
    )
    return ratings, movies


def test_train_test_split():
    ratings, _ = _small_dataset()
    train, test = train_test_split_by_user(ratings, test_size=0.5, seed=1)
    # every user with >1 rating should have at least 1 test
    assert test.userId.nunique() >= 1
    assert train.shape[0] + test.shape[0] == ratings.shape[0]


def test_metrics_and_runner():
    ratings, movies = _small_dataset()
    # simple recommender that returns movie 10 for every user
    def dummy_rec(u, train, movies_df, top_n):
        return pd.DataFrame({"movieId": [10]})

    # Prepare a split where some users have movie 10 in test
    train, test = train_test_split_by_user(ratings, test_size=0.5, seed=2)
    res = evaluate_topk(dummy_rec, train, test, movies, k=1)
    assert set(res.keys()) == {"precision", "recall", "ndcg"}
    # metrics are between 0 and 1
    assert 0.0 <= res["precision"] <= 1.0
    assert 0.0 <= res["recall"] <= 1.0
    assert 0.0 <= res["ndcg"] <= 1.0

    # Test metrics functions directly
    pred = [10, 20, 30]
    actual = {20, 30}
    assert precision_at_k(pred, actual, 2) == 0.5
    assert recall_at_k(pred, actual, 3) == 1.0
    ndcg_val = ndcg_at_k(pred, actual, 3)
    assert 0.0 <= ndcg_val <= 1.0
