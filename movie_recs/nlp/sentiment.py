from typing import Dict, List, Tuple

# Prefer using vaderSentiment when available, fall back to a simple rule-based scorer
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _HAS_VADER = True
except Exception:
    _HAS_VADER = False

# simple fallback lexicons
_POSITIVE = set(["good", "great", "excellent", "amazing", "love", "loved", "like", "liked", "best", "wonderful", "enjoyed"]) 
_NEGATIVE = set(["bad", "terrible", "awful", "hate", "hated", "dislike", "boring", "worst", "poor", "disappointing"]) 


def _fallback_score(text: str) -> Dict[str, float]:
    words = [w.strip(".,!?;:\"'()[]") .lower() for w in (text or "").split()]
    if not words:
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    pos = sum(1 for w in words if w in _POSITIVE)
    neg = sum(1 for w in words if w in _NEGATIVE)
    total = len(words)
    pos_score = pos / total
    neg_score = neg / total
    neu_score = max(0.0, 1.0 - pos_score - neg_score)
    # compound scaled between -1 and 1
    compound = (pos_score - neg_score)
    return {"neg": float(neg_score), "neu": float(neu_score), "pos": float(pos_score), "compound": float(compound)}


def analyze_sentiment(text: str) -> Dict[str, float]:
    """Return sentiment scores: dict with keys neg, neu, pos, compound (float -1..1).

    Uses VADER if available; otherwise a simple fallback.
    """
    if _HAS_VADER:
        analyzer = SentimentIntensityAnalyzer()
        return analyzer.polarity_scores(text or "")
    else:
        return _fallback_score(text)


def sentiment_label(compound: float) -> str:
    """Return label 'positive'|'negative'|'neutral' from compound score using VADER thresholds."""
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def analyze_reviews(reviews: List[str]) -> Tuple[float, str]:
    """Aggregate sentiment over multiple review texts; returns (average_compound, label)."""
    if not reviews:
        return 0.0, "neutral"
    compounds = [analyze_sentiment(r).get("compound", 0.0) for r in reviews]
    avg = float(sum(compounds) / len(compounds))
    return avg, sentiment_label(avg)


__all__ = ["analyze_sentiment", "sentiment_label", "analyze_reviews"]
