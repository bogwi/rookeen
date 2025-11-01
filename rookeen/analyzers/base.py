from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, TypeVar

from rookeen.models import AnalysisType, LinguisticAnalysisResult

try:  # Import for type checking only to avoid hard dependency at import time
    import spacy  # noqa: F401
    from spacy.tokens import Doc
except Exception:  # pragma: no cover
    Doc = object


class BaseAnalyzer(ABC):
    """Abstract base class for all analyzers operating on spaCy Doc.

    Subclasses must set a unique `name` and an `analysis_type` from AnalysisType,
    and implement the asynchronous `analyze` method.
    """

    name: ClassVar[str]
    analysis_type: ClassVar[AnalysisType]

    @abstractmethod
    async def analyze(self, doc: Doc, lang: str) -> LinguisticAnalysisResult:
        """Run the analyzer over a spaCy Doc and return a structured result.

        The `lang` argument is the normalized ISO 639-1 language code for context.
        """
        raise NotImplementedError


AnalyzerT = TypeVar("AnalyzerT", bound="BaseAnalyzer")


_ANALYZER_REGISTRY: dict[str, type[BaseAnalyzer]] = {}


def register_analyzer(cls: type[AnalyzerT]) -> type[AnalyzerT]:
    """Class decorator to register an analyzer by its `name`.

    Ensures names are unique and easily discoverable for flag-based selection.
    """
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError(f"Analyzer class {cls.__name__} must define a non-empty class var `name`.")
    if name in _ANALYZER_REGISTRY:
        raise ValueError(
            f"Analyzer name '{name}' is already registered to {_ANALYZER_REGISTRY[name].__name__}."
        )
    _ANALYZER_REGISTRY[name] = cls
    return cls


def available_analyzers() -> list[str]:
    """Return a sorted list of available analyzer names."""
    return sorted(_ANALYZER_REGISTRY.keys())


def get_analyzer(name: str) -> type[BaseAnalyzer]:
    """Fetch an analyzer class by its registered name."""
    try:
        return _ANALYZER_REGISTRY[name]
    except KeyError as exc:  # pragma: no cover - simple guard
        raise KeyError(f"Unknown analyzer '{name}'. Known: {sorted(_ANALYZER_REGISTRY)}") from exc
