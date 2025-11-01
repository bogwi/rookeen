from __future__ import annotations

import time
from collections import Counter, defaultdict

from rookeen.models import AnalysisType, LinguisticAnalysisResult

from .base import BaseAnalyzer, register_analyzer

try:
    from spacy.tokens import Doc
except Exception:  # pragma: no cover
    Doc = object


@register_analyzer
class POSAnalyzer(BaseAnalyzer):
    name = "pos"
    analysis_type = AnalysisType.POS

    async def analyze(self, doc: Doc, lang: str) -> LinguisticAnalysisResult:
        start_ts = time.perf_counter()

        upos_counts: dict[str, int] = Counter(t.pos_ for t in doc)

        # Ratios relative to all tokens (avoid divide-by-zero)
        total_tokens = sum(upos_counts.values())
        upos_ratios: dict[str, float] = {}
        if total_tokens:
            for tag, count in upos_counts.items():
                upos_ratios[tag] = count / total_tokens
        else:
            upos_ratios = {tag: 0.0 for tag in upos_counts}

        # Top lemmas by UPOS
        top_lemmas_by_upos: dict[str, list[tuple[str, int]]] = {}
        buckets: dict[str, list[str]] = defaultdict(list)
        for t in doc:
            if t.is_alpha and not t.is_stop:
                key = t.pos_ or "X"
                lemma = (t.lemma_ or t.text).lower()
                buckets[key].append(lemma)

        for tag, lemmas in buckets.items():
            counts = Counter(lemmas)
            top_lemmas_by_upos[tag] = counts.most_common(5)

        results = {
            "upos_counts": dict(upos_counts),
            "upos_ratios": upos_ratios,
            "top_lemmas_by_upos": top_lemmas_by_upos,
        }

        processing_time = time.perf_counter() - start_ts
        return LinguisticAnalysisResult(
            analysis_type=self.analysis_type,
            name=self.name,
            results=results,
            processing_time=processing_time,
            confidence=1.0,
        )
