from __future__ import annotations

import time

from rookeen.models import AnalysisType, LinguisticAnalysisResult

from .base import BaseAnalyzer, register_analyzer

try:
    from spacy.tokens import Doc
except Exception:  # pragma: no cover
    Doc = object


@register_analyzer
class ReadabilityAnalyzer(BaseAnalyzer):
    name = "readability"
    analysis_type = AnalysisType.READABILITY

    async def analyze(self, doc: Doc, lang: str) -> LinguisticAnalysisResult:
        start_ts = time.perf_counter()
        from textstat import textstat

        text = doc.text or ""
        # Compute language-agnostic metrics but caveat they are tuned for English
        results: dict[str, float | int | str | bool] = {
            "supported": True,
            "note": "Readability metrics are calibrated for English; interpret non-English results with caution.",
            "flesch_reading_ease": float(textstat.flesch_reading_ease(text)),
            "flesch_kincaid_grade": float(textstat.flesch_kincaid_grade(text)),
            "smog_index": float(textstat.smog_index(text)),
            "automated_readability_index": float(textstat.automated_readability_index(text)),
            "coleman_liau_index": float(textstat.coleman_liau_index(text)),
            "linsear_write_formula": float(textstat.linsear_write_formula(text)),
            "dale_chall_readability_score": float(textstat.dale_chall_readability_score(text)),
            "difficult_words": int(textstat.difficult_words(text)),
            "text_standard": str(textstat.text_standard(text, float_output=False)),
        }

        processing_time = time.perf_counter() - start_ts
        return LinguisticAnalysisResult(
            analysis_type=self.analysis_type,
            name=self.name,
            results=results,
            processing_time=processing_time,
            confidence=1.0,
        )
