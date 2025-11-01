from __future__ import annotations

import os
import time

from spacy.tokens.doc import Doc

from rookeen.analyzers.base import BaseAnalyzer, register_analyzer
from rookeen.analyzers.embeddings_backends import get_backend
from rookeen.models import AnalysisType, LinguisticAnalysisResult

DEFAULT_BACKEND = os.getenv("ROOKEEN_EMBEDDINGS_BACKEND", "miniLM")
DEFAULT_MODEL_MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def _resolve_backend_and_model() -> tuple[str, str]:
    # Prefer CLI/env in later steps; for analyzer-only context, use env/defaults
    backend = os.getenv("ROOKEEN_EMBEDDINGS_BACKEND", DEFAULT_BACKEND)
    model = os.getenv("ROOKEEN_EMBEDDINGS_MODEL", "")
    return backend, model


@register_analyzer
class EmbeddingsAnalyzer(BaseAnalyzer):
    """Analyzer for generating sentence embeddings using pluggable backends."""

    name = "embeddings"
    analysis_type = AnalysisType.EMBEDDINGS

    async def analyze(self, doc: Doc, lang: str) -> LinguisticAnalysisResult:
        """Generate embeddings for the document text via selected backend."""
        start = time.perf_counter()
        backend_key, model_name = _resolve_backend_and_model()

        # Construct backend
        try:
            kwargs: dict[str, object] = {}
            if backend_key == "miniLM":
                kwargs["model_name"] = model_name or DEFAULT_MODEL_MINILM
            elif backend_key == "bge-m3":
                kwargs["model_name"] = model_name or "BAAI/bge-m3"
            elif backend_key == "openai-te3":
                kwargs["model_name"] = model_name or os.getenv("ROOKEEN_OPENAI_MODEL", "text-embedding-3-small")
                kwargs["api_key"] = os.getenv("ROOKEEN_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
            m = get_backend(backend_key, **kwargs)
        except Exception as e:
            # Avoid leaking secrets in error messages
            msg = str(e)
            for k in (os.getenv("ROOKEEN_OPENAI_API_KEY"), os.getenv("OPENAI_API_KEY")):
                if k:
                    msg = msg.replace(k, "***REDACTED***")
            return LinguisticAnalysisResult(
                analysis_type=self.analysis_type,
                name=self.name,
                results={"supported": False, "note": f"embeddings backend unavailable: {msg}"},
                processing_time=time.perf_counter() - start,
                confidence=0.0,
            )

        try:
            vec = m.embed(doc.text)
            prov = m.provenance()
            return LinguisticAnalysisResult(
                analysis_type=self.analysis_type,
                name=self.name,
                results={
                    "supported": True,
                    **prov,
                    "vector": vec,
                },
                processing_time=time.perf_counter() - start,
                confidence=1.0,
            )
        except Exception as e:
            # Avoid leaking secrets in error messages
            msg = str(e)
            for k in (os.getenv("ROOKEEN_OPENAI_API_KEY"), os.getenv("OPENAI_API_KEY")):
                if k:
                    msg = msg.replace(k, "***REDACTED***")
            return LinguisticAnalysisResult(
                analysis_type=self.analysis_type,
                name=self.name,
                results={"supported": False, "note": f"embedding failed: {msg}"},
                processing_time=time.perf_counter() - start,
                confidence=0.0,
            )
