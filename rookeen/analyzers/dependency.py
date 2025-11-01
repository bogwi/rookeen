from __future__ import annotations

import time
from collections import Counter

from rookeen.analyzers.base import BaseAnalyzer, register_analyzer
from rookeen.models import AnalysisType, LinguisticAnalysisResult

try:
    from spacy.tokens import Doc
except Exception:  # pragma: no cover
    Doc = object


@register_analyzer
class DependencyAnalyzer(BaseAnalyzer):
    name = "dependency"
    analysis_type = AnalysisType.POS

    async def analyze(self, doc: Doc, lang: str) -> LinguisticAnalysisResult:
        start = time.perf_counter()
        if not hasattr(doc, "has_annotation") or not doc.has_annotation("DEP"):
            return LinguisticAnalysisResult(
                analysis_type=self.analysis_type,
                name=self.name,
                results={"supported": False, "note": "Parser not available"},
                processing_time=time.perf_counter() - start,
                confidence=0.8,
                metadata={},
            )
        dep_counts: Counter[str] = Counter(t.dep_ for t in doc)
        head_pos_counts: Counter[str] = Counter(f"{t.head.pos_}->{t.dep_}" for t in doc)
        return LinguisticAnalysisResult(
            analysis_type=self.analysis_type,
            name=self.name,
            results={
                "supported": True,
                "dep_counts": dict(dep_counts),
                "head_pos_dep": dict(head_pos_counts.most_common(20)),
            },
            processing_time=time.perf_counter() - start,
            confidence=0.9,
            metadata={},
        )
