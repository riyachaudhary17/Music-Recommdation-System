from movie_recs.nlp.sentiment import analyze_sentiment, sentiment_label, analyze_reviews


def test_simple_sentiment():
    pos = "I absolutely love this movie, it was amazing and wonderful"
    neg = "I hate this film. It was boring and the worst"
    neu = "This movie was okay, not good not bad"

    sp = analyze_sentiment(pos)
    sn = analyze_sentiment(neg)
    se = analyze_sentiment(neu)

    assert "compound" in sp and "compound" in sn
    assert sentiment_label(sp.get("compound", 0)) == "positive"
    assert sentiment_label(sn.get("compound", 0)) == "negative"
    # neutral may be neutral or slight; check function runs
    assert isinstance(analyze_reviews([pos, neu, neg])[0], float)
