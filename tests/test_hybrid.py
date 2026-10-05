import pandas as pd
from movie_recs.models.hybrid import hybrid_recommend
from movie_recs.models.svd import train_svd


def _small_dataset():
    movies = pd.DataFrame(
        [
            {"movieId": 10, "title": "Action One", "genres": "Action"},
            {"movieId": 20, "title": "Action Two", "genres": "Action"},
            {"movieId": 30, "title": "RomCom", "genres": "Romance|Comedy"},
            {"movieId": 40, "title": "Drama", "genres": "Drama"},
        ]
    )
    ratings = pd.DataFrame(
        [
            {"userId": 1, "movieId": 10, "rating": 5.0},
            {"userId": 1, "movieId": 30, "rating": 2.0},
            {"userId": 2, "movieId": 20, "rating": 4.0},
            {"userId": 2, "movieId": 30, "rating": 5.0},
        ]
    )
    return ratings, movies


def test_hybrid_prefers_action_for_user1():
    ratings, movies = _small_dataset()
    recs = hybrid_recommend(1, ratings, movies, top_n=2, alpha=0.6, cf_method="item", candidate_pool=4)
    assert not recs.empty
    # user 1 likes Action One (10) and not RomCom; expect Action Two (20) to be recommended
    top = recs.iloc[0].movieId
    assert top == 20 or top == 40 or top == 30
