import pandas as pd
from movie_recs.models.svd import train_svd, recommend_svd


def _small_dataset():
    ratings = pd.DataFrame(
        [
            {"userId": 1, "movieId": 10, "rating": 5.0},
            {"userId": 1, "movieId": 20, "rating": 3.0},
            {"userId": 2, "movieId": 10, "rating": 4.0},
            {"userId": 2, "movieId": 30, "rating": 4.0},
            {"userId": 3, "movieId": 20, "rating": 5.0},
            {"userId": 3, "movieId": 30, "rating": 3.0},
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


def test_train_and_recommend():
    ratings, movies = _small_dataset()
    u, s, vt, user_ids, item_ids, gmean = train_svd(ratings, n_factors=2)
    assert u.shape[0] == len(user_ids)
    assert vt.shape[1] == len(item_ids)
    recs = recommend_svd(1, u, s, vt, user_ids, item_ids, ratings, movies, top_n=2)
    assert "movieId" in recs.columns
    # user 1 has rated 10 and 20, so recommended items should not include them
    assert not any([mid in [10, 20] for mid in recs.movieId.tolist()])
