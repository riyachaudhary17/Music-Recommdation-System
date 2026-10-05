import pandas as pd
from movie_recs.models.baselines import top_n_popular, item_based_recommendations


def _small_dataset():
    # small synthetic dataset
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


def test_top_n_popular():
    ratings, movies = _small_dataset()
    top = top_n_popular(ratings, movies, n=2)
    # The most rated movie is movieId 10 (2 ratings), then 20 and 30 both have 2 ratings but 20 avg higher
    assert top.iloc[0]["movieId"] == 10
    assert len(top) == 2


def test_item_based_recommendations():
    ratings, movies = _small_dataset()
    # user 1 has rated 10 and 20, should recommend 30 or 40 (30 has related ratings)
    recs = item_based_recommendations(1, ratings, movies, top_n=2, k=2)
    assert "movieId" in recs.columns
    # recommend list length <= requested top_n
    assert len(recs) <= 2
    # movie 40 has no ratings so should not be strongly recommended; movie 30 is connected via other users
    ids = recs.movieId.tolist()
    assert 30 in ids or 40 in ids
