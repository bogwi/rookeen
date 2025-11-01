from __future__ import annotations

import time
from typing import Any

from rookeen.analyzers.base import BaseAnalyzer, register_analyzer

try:  # Import for type checking only to avoid hard dependency at import time
    from spacy.tokens import Doc
except Exception:  # pragma: no cover
    Doc = object
from rookeen.models import AnalysisType, LinguisticAnalysisResult

# Try multiple sentiment libraries in order of preference
_sentiment_analyzer: tuple[str, Any] | None = None


def _get_sentiment_analyzer() -> tuple[str, Any] | None:
    """Get the best available sentiment analyzer."""
    global _sentiment_analyzer

    if _sentiment_analyzer is not None:
        return _sentiment_analyzer

    # Try VADER first (fastest, no dependencies)
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _sentiment_analyzer = ("vader", SentimentIntensityAnalyzer())
        return _sentiment_analyzer
    except ImportError:
        pass

    # Try TextBlob (good balance of speed and accuracy)
    try:
        from textblob import TextBlob

        _sentiment_analyzer = ("textblob", TextBlob)
        return _sentiment_analyzer
    except ImportError:
        pass

    # spaCy sentiment if available
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
        if hasattr(nlp, "sentiment") or any(hasattr(comp, "sentiment") for comp in nlp.pipeline):
            _sentiment_analyzer = ("spacy", nlp)
            return _sentiment_analyzer
    except (ImportError, OSError):
        pass

    return None


@register_analyzer
class SentimentAnalyzer(BaseAnalyzer):
    """Production-ready sentiment analyzer using best available library."""

    name = "sentiment"
    analysis_type = AnalysisType.SENTIMENT

    async def analyze(self, doc: Doc, lang: str) -> LinguisticAnalysisResult:
        """Analyze sentiment using the best available method."""
        start = time.perf_counter()
        text = doc.text

        analyzer = _get_sentiment_analyzer()
        if not analyzer:
            return LinguisticAnalysisResult(
                analysis_type=self.analysis_type,
                name=self.name,
                results={"supported": False, "note": "No sentiment analysis library available"},
                processing_time=time.perf_counter() - start,
                confidence=0.0,
            )

        analyzer_type, analyzer_instance = analyzer

        try:
            if analyzer_type == "vader":
                scores = analyzer_instance.polarity_scores(text)
                # Convert VADER compound score to sentiment label
                compound = scores["compound"]
                if compound >= 0.05:
                    label = "positive"
                elif compound <= -0.05:
                    label = "negative"
                else:
                    label = "neutral"

                result = {
                    "supported": True,
                    "label": label,
                    "score": abs(compound),
                    "method": "vader",
                    "scores": scores,
                    "processing_time": time.perf_counter() - start,
                }

            elif analyzer_type == "textblob":
                blob = analyzer_instance(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity

                if polarity > 0.1:
                    label = "positive"
                elif polarity < -0.1:
                    label = "negative"
                else:
                    label = "neutral"

                result = {
                    "supported": True,
                    "label": label,
                    "score": abs(polarity),
                    "polarity": polarity,
                    "subjectivity": subjectivity,
                    "method": "textblob",
                    "processing_time": time.perf_counter() - start,
                }

            else:
                result = {"supported": False, "note": f"Unsupported analyzer: {analyzer_type}"}

        except Exception as e:
            result = {
                "supported": False,
                "note": f"Analysis failed: {e!s}",
                "method": analyzer_type,
            }

        confidence = result.get("score", 0.0) if result.get("supported") else 0.0

        return LinguisticAnalysisResult(
            analysis_type=self.analysis_type,
            name=self.name,
            results=result,
            processing_time=time.perf_counter() - start,
            confidence=confidence,
        )
