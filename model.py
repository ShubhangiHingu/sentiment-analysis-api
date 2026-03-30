"""Sentiment model helpers with graceful fallback.

If transformers/torch cannot load (e.g. missing VC++ runtime or unsupported
Python build), we fall back to a tiny keyword-based scorer so the API keeps
running for development and testing.
"""

from __future__ import annotations

from typing import Optional

sentiment_pipeline: Optional[object] = None
MODEL_BACKEND = "keyword-fallback"

try:
    from transformers import pipeline

    # This might take a minute on first run as model files are downloaded.
    sentiment_pipeline = pipeline("sentiment-analysis")
    MODEL_BACKEND = "transformers"
except Exception:
    sentiment_pipeline = None


POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "amazing",
    "love",
    "happy",
    "awesome",
    "best",
    "fantastic",
    "nice",
}

NEGATIVE_WORDS = {
    "bad",
    "terrible",
    "awful",
    "hate",
    "sad",
    "worst",
    "horrible",
    "poor",
    "angry",
    "disappointed",
}


def _keyword_fallback(text: str) -> dict:
    words = [w.strip(".,!?;:()[]{}\"'").lower() for w in text.split()]
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)

    if pos == neg:
        return {"label": "NEUTRAL", "score": 0.50}
    if pos > neg:
        score = min(0.55 + 0.10 * (pos - neg), 0.95)
        return {"label": "POSITIVE", "score": round(score, 4)}
    score = min(0.55 + 0.10 * (neg - pos), 0.95)
    return {"label": "NEGATIVE", "score": round(score, 4)}


def analyze_sentiment(text: str) -> dict:
    if sentiment_pipeline is not None:
        result = sentiment_pipeline(text)[0]
        return {"label": result["label"], "score": float(result["score"])}
    return _keyword_fallback(text)