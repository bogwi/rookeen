from __future__ import annotations

import time
from collections import Counter
from statistics import mean

from rookeen.models import AnalysisType, LinguisticAnalysisResult

from .base import BaseAnalyzer, register_analyzer

try:
    from spacy.tokens import Doc
except Exception:  # pragma: no cover
    Doc = object


@register_analyzer
class LexicalStatsAnalyzer(BaseAnalyzer):
    name = "lexical_stats"
    analysis_type = AnalysisType.LEXICAL_STATS

    async def analyze(self, doc: Doc, lang: str) -> LinguisticAnalysisResult:
        start_ts = time.perf_counter()

        alpha_tokens = [t for t in doc if t.is_alpha and not t.is_stop]
        total_tokens = len(alpha_tokens)

        lemmas: list[str] = []
        token_lengths: list[int] = []
        for t in alpha_tokens:
            lemma = (t.lemma_ or t.text).lower()
            lemmas.append(lemma)
            token_lengths.append(len(t.text))

        unique_lemmas = len(set(lemmas))
        sentences = sum(1 for _ in doc.sents) if hasattr(doc, "sents") else 0
        avg_token_length = float(mean(token_lengths)) if token_lengths else 0.0

        sent_lengths: list[int] = []
        if hasattr(doc, "sents"):
            for s in doc.sents:
                sent_lengths.append(sum(1 for t in s if t.is_alpha))
        avg_sentence_length_tokens = float(mean(sent_lengths)) if sent_lengths else 0.0

        type_token_ratio = (unique_lemmas / total_tokens) if total_tokens else 0.0

        lemma_counts: dict[str, int] = Counter(lemmas)
        top_lemmas = sorted(lemma_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20]

        results = {
            "total_tokens": total_tokens,
            "unique_lemmas": unique_lemmas,
            "sentences": sentences,
            "avg_token_length": avg_token_length,
            "avg_sentence_length_tokens": avg_sentence_length_tokens,
            "type_token_ratio": type_token_ratio,
            "top_lemmas": top_lemmas,
        }

        processing_time = time.perf_counter() - start_ts
        return LinguisticAnalysisResult(
            analysis_type=self.analysis_type,
            name=self.name,
            results=results,
            processing_time=processing_time,
            confidence=1.0,
        )
