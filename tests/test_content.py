import pandas as pd
from movie_recs.models.content import build_item_tfidf, recommend_content


def _small_dataset():
    movies = pd.DataFrame([
        {"movieId": 1, "title": "Fast Action", "genres": "Action|Thriller"},
        {"movieId": 2, "title": "Speed Chase", "genres": "Action"},
        {"movieId": 3, "title": "Romantic Comedy", "genres": "Romance|Comedy"},
        {"movieId": 4, "title": "Dramatic Love", "genres": "Romance|Drama"},
    ])
    ratings = pd.DataFrame([
        {"userId": 10, "movieId": 1, "rating": 5.0},
        {"userId": 10, "movieId": 3, "rating": 2.0},
        {"userId": 11, "movieId": 3, "rating": 5.0},
    ])
    return ratings, movies


def test_build_and_recommend():
    ratings, movies = _small_dataset()
    vec, item_vectors, item_ids = build_item_tfidf(movies, fields=["title", "genres"])
    recs = recommend_content(10, ratings, movies, top_n=2, vectorizer=vec, item_vectors=item_vectors, fields=["title", "genres"])
    assert not recs.empty
    # Since user 10 likes movie 1 (Action) and less movie 3 (Romance), the top recommendation should be movie 2 (Action)
    top_id = recs.iloc[0].movieId
    assert top_id == 2
