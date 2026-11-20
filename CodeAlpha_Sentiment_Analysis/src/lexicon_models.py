"""Lexicon / rule-based sentiment: VADER and TextBlob.

Both return a continuous score between -1 (very negative) and +1 (very positive).
"""
import numpy as np

_VADER = None


def vader_scores(texts):
    """VADER compound score. Works best on informal text (tweets, reviews)."""
    global _VADER
    if _VADER is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _VADER = SentimentIntensityAnalyzer()
    return np.array([_VADER.polarity_scores(t)["compound"] for t in texts])


def textblob_scores(texts):
    """TextBlob polarity. Returns None if TextBlob is not installed."""
    try:
        from textblob import TextBlob
    except ImportError:
        return None
    return np.array([TextBlob(t).sentiment.polarity for t in texts])


def score_to_label(scores, pos_thr=0.05, neg_thr=-0.05):
    scores = np.asarray(scores, dtype=float)
    return np.where(scores >= pos_thr, "positive",
                    np.where(scores <= neg_thr, "negative", "neutral"))
