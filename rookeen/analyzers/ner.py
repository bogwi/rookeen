from __future__ import annotations

import time
from collections import Counter, defaultdict

from rookeen.models import AnalysisType, LinguisticAnalysisResult

from .base import BaseAnalyzer, register_analyzer

try:
    from spacy.language import Language
    from spacy.tokens import Doc
except Exception:  # pragma: no cover
    Doc = object
    Language = object


@register_analyzer
class NERAnalyzer(BaseAnalyzer):
    name = "ner"
    analysis_type = AnalysisType.NER

    def __init__(self, nlp: Language | None = None) -> None:
        # Optional injection of nlp to introspect pipes if caller provides it
        self._nlp = nlp

    async def analyze(self, doc: Doc, lang: str) -> LinguisticAnalysisResult:
        start_ts = time.perf_counter()

        # nlp = getattr(doc, "vocab", None)
        has_ner_pipe = False
        # Try best-effort to determine if NER pipe exists
        # nlp_obj = getattr(doc, "_", None)
        if hasattr(doc, "_") and hasattr(doc._, "get"):  # pragma: no cover - defensive
            pass
        # Use injected nlp if provided for pipe inspection
        if self._nlp is not None and hasattr(self._nlp, "has_pipe"):
            try:
                has_ner_pipe = bool(self._nlp.has_pipe("ner"))
            except Exception:  # pragma: no cover
                has_ner_pipe = False

        ents = list(getattr(doc, "ents", []) or [])
        if (not ents) and not has_ner_pipe:
            processing_time = time.perf_counter() - start_ts
            return LinguisticAnalysisResult(
                analysis_type=self.analysis_type,
                name=self.name,
                results={
                    "supported": False,
                    "counts_by_label": {},
                    "examples_by_label": {},
                    "total_entities": 0,
                },
                processing_time=processing_time,
                confidence=1.0,
            )

        counts_by_label: dict[str, int] = Counter(ent.label_ for ent in ents)
        examples_by_label: dict[str, list[str]] = defaultdict(list)
        for ent in ents:
            if len(examples_by_label[ent.label_]) < 10:
                examples_by_label[ent.label_].append(ent.text)
        total_entities = sum(counts_by_label.values())

        results = {
            "supported": True,
            "counts_by_label": dict(counts_by_label),
            "examples_by_label": dict(examples_by_label),
            "total_entities": total_entities,
        }

        processing_time = time.perf_counter() - start_ts
        return LinguisticAnalysisResult(
            analysis_type=self.analysis_type,
            name=self.name,
            results=results,
            processing_time=processing_time,
            confidence=1.0,
        )
