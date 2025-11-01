from __future__ import annotations

import time
from collections import Counter

from rookeen.models import AnalysisType, LinguisticAnalysisResult

from .base import BaseAnalyzer, register_analyzer

try:
    import yake
except Exception:  # pragma: no cover
    yake = None

try:
    from spacy.tokens import Doc
except Exception:  # pragma: no cover
    Doc = object


@register_analyzer
class KeywordAnalyzer(BaseAnalyzer):
    name = "keywords"
    analysis_type = AnalysisType.KEYWORDS

    def __init__(self, use_yake: bool | None = None) -> None:
        # If None, auto-enable if yake is importable
        self.use_yake = (yake is not None) if use_yake is None else use_yake

    async def analyze(self, doc: Doc, lang: str) -> LinguisticAnalysisResult:
        start_ts = time.perf_counter()

        results: dict[str, object]
        # method_note = ""
        if self.use_yake and yake is not None:
            # YAKE-based keyword extraction if available
            text = doc.text or ""
            kw_extractor = yake.KeywordExtractor(lan=lang if len(lang) == 2 else "en", n=1, top=20)
            keywords: list[tuple[str, float]] = kw_extractor.extract_keywords(text)
            # YAKE returns lower score for more important keywords; invert to score=1/r
            normalized: list[tuple[str, float]] = []
            for phrase, score in keywords:
                inv = 0.0
                try:
                    inv = 1.0 / float(score) if float(score) > 0 else 0.0
                except Exception:  # pragma: no cover
                    inv = 0.0
                normalized.append((phrase, inv))
            results = {
                "method": "yake",
                "keywords": normalized,
            }
        else:
            # Frequency-based TF over lemma-lower of alpha, non-stop tokens
            alpha_nonstop = [t for t in doc if t.is_alpha and not t.is_stop]
            total_alpha = len(alpha_nonstop)
            lemma_counts = Counter((t.lemma_ or t.text).lower() for t in alpha_nonstop)
            scored: list[tuple[str, float]] = []
            for lemma, count in lemma_counts.items():
                score = (count / total_alpha) if total_alpha else 0.0
                scored.append((lemma, score))
            scored.sort(key=lambda kv: (-kv[1], kv[0]))
            results = {
                "method": "frequency",
                "keywords": scored[:20],
            }

        processing_time = time.perf_counter() - start_ts
        return LinguisticAnalysisResult(
            analysis_type=self.analysis_type,
            name=self.name,
            results=results,
            processing_time=processing_time,
            confidence=1.0,
        )
