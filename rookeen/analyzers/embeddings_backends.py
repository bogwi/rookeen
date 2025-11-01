from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import OpenAI
    from sentence_transformers import SentenceTransformer

__all__ = [
    "EmbeddingBackend",
    "register_backend",
    "get_backend",
]


class EmbeddingBackend(ABC):
    """Abstract interface for pluggable embeddings backends.

    Concrete implementations must provide:
    - load(): materialize any heavyweight resources (models, clients). Should be idempotent.
    - embed(text): return a single L2-normalized embedding vector as a list[float].
    - provenance(): return backend/model metadata, including keys like
      {"backend": str, "model": str, "dim": int, "normalized": bool}.
    """

    @abstractmethod
    def load(self) -> None:
        """Load or initialize underlying resources. Idempotent."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Compute an embedding vector for the provided text.

        Implementations should return a Python list of floats suitable for JSON serialization.
        The vector should be L2-normalized for cosine-similarity compatibility.
        """

    @abstractmethod
    def provenance(self) -> dict[str, Any]:
        """Return metadata describing the backend/model configuration."""


_REGISTRY: dict[str, type[EmbeddingBackend]] = {}


def register_backend(name: str) -> Callable[[type[EmbeddingBackend]], type[EmbeddingBackend]]:
    """Class decorator to register an embeddings backend under a unique key.

    Usage:
        @register_backend("miniLM")
        class MiniLMBackend(EmbeddingBackend):
            ...
    """

    def _wrap(cls: type[EmbeddingBackend]) -> type[EmbeddingBackend]:
        if not issubclass(cls, EmbeddingBackend):
            raise TypeError(f"{cls!r} must subclass EmbeddingBackend")
        if name in _REGISTRY:
            raise ValueError(f"Embedding backend '{name}' already registered")
        _REGISTRY[name] = cls
        return cls

    return _wrap


def get_backend(name: str, **kwargs: Any) -> EmbeddingBackend:
    """Instantiate a registered embeddings backend by key.

    Raises KeyError if the backend name is unknown.
    Any keyword arguments are forwarded to the backend constructor.
    """

    try:
        cls = _REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown embeddings backend '{name}'. Known: [{known}]") from exc
    return cls(**kwargs)


@register_backend("bge-m3")
class BgeM3Backend(EmbeddingBackend):
    """Local BGE-M3 embeddings backend using sentence-transformers.

    Defaults to CPU. The optional 'device' argument can be set by callers if needed.
    Returns L2-normalized vectors for cosine-similarity workflows.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None) -> None:
        self.model_name = model_name
        # Default to CPU; auto-detect Apple Silicon (MPS) or CUDA when no explicit device provided
        if device is None:
            try:
                import torch

                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "mps"
                elif torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"
            except Exception:
                self.device = "cpu"
        else:
            self.device = device
        self._model: SentenceTransformer | None = None

    def load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - import-time failure path
            raise RuntimeError(
                "sentence-transformers not installed; install rookeen[embeddings]"
            ) from exc
        self._model = SentenceTransformer(self.model_name, device=self.device)

    def embed(self, text: str) -> list[float]:
        if self._model is None:
            self.load()
        assert self._model is not None
        vec = self._model.encode([text], normalize_embeddings=False)[0]
        # L2 normalize
        norm = math.sqrt(float((vec ** 2).sum())) or 1.0
        result: list[float] = (vec / norm).tolist()
        return result

    def provenance(self) -> dict[str, Any]:
        return {
            "backend": "bge-m3",
            "model": self.model_name,
            "dim": 1024,
            "normalized": True,
        }


@register_backend("openai-te3")
class OpenAITe3Backend(EmbeddingBackend):
    """OpenAI text-embedding-3 backend (API-based).

    Supports models: text-embedding-3-small (1536 dims), text-embedding-3-large (3072 dims).
    API key is taken from constructor or environment (ROOKEEN_OPENAI_API_KEY / OPENAI_API_KEY).
    Returns L2-normalized vectors for cosine-similarity workflows.
    """

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str | None = None) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("ROOKEEN_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self._client: OpenAI | None = None

    def load(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "OpenAI API key not provided. Set OPENAI_API_KEY or ROOKEEN_OPENAI_API_KEY."
            )
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - import-time failure path
            raise RuntimeError(
                "openai package not installed; install rookeen[embeddings]"
            ) from exc
        # Optional timeout via env (seconds)
        timeout_env = os.getenv("ROOKEEN_OPENAI_TIMEOUT") or os.getenv("OPENAI_TIMEOUT")
        timeout = None
        if timeout_env:
            try:
                timeout = float(timeout_env)
            except Exception:
                timeout = None
        self._client = OpenAI(api_key=self.api_key, timeout=timeout) if timeout else OpenAI(api_key=self.api_key)

    def embed(self, text: str) -> list[float]:
        if self._client is None:
            self.load()
        assert self._client is not None
        resp = self._client.embeddings.create(model=self.model_name, input=text)
        vec = resp.data[0].embedding
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def provenance(self) -> dict[str, Any]:
        dim = 1536 if self.model_name.endswith("small") else 3072
        return {
            "backend": "openai-te3",
            "model": self.model_name,
            "dim": dim,
            "normalized": True,
        }


@register_backend("miniLM")
class MiniLMBackend(EmbeddingBackend):
    """MiniLM backend using sentence-transformers baseline model.

    Default model: sentence-transformers/all-MiniLM-L6-v2 (384 dims).
    Returns L2-normalized vectors for cosine similarity.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None) -> None:
        self.model_name = model_name
        if device is None:
            try:
                import torch

                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "mps"
                elif torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"
            except Exception:
                self.device = "cpu"
        else:
            self.device = device
        self._model: SentenceTransformer | None = None

    def load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - import-time failure path
            raise RuntimeError(
                "sentence-transformers not installed; install rookeen[embeddings]"
            ) from exc
        self._model = SentenceTransformer(self.model_name, device=self.device)

    def embed(self, text: str) -> list[float]:
        if self._model is None:
            self.load()
        assert self._model is not None
        vec = self._model.encode([text], normalize_embeddings=False)[0]
        # L2 normalize
        norm = math.sqrt(float((vec ** 2).sum())) or 1.0
        result: list[float] = (vec / norm).tolist()
        return result

    def provenance(self) -> dict[str, Any]:
        return {
            "backend": "miniLM",
            "model": self.model_name,
            "dim": 384,
            "normalized": True,
        }

