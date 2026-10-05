from fastapi.testclient import TestClient
from movie_recs.api.server import app

client = TestClient(app)


def test_sentiment_endpoint():
    resp = client.post("/sentiment", json={"text": "I absolutely love this movie, it was amazing"})
    assert resp.status_code == 200
    data = resp.json()
    assert "scores" in data and "label" in data
    assert data["label"] in {"positive", "neutral", "negative"}

    # empty text
    resp2 = client.post("/sentiment", json={"text": ""})
    assert resp2.status_code == 200
    d2 = resp2.json()
    assert "scores" in d2 and "label" in d2
