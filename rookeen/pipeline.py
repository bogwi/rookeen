from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Iterable, Sequence
from typing import Any

from rookeen.analyzers.base import BaseAnalyzer
from rookeen.analyzers.dependency import DependencyAnalyzer
from rookeen.language import detect_language, get_spacy_model, model_name_for, normalize_lang
from rookeen.models import LinguisticAnalysisResult, WebPageContent
from rookeen.scraping import AsyncWebScraper
from rookeen.utils.logging import get_logger

try:  # pragma: no cover - import guard for type hints
    from spacy.language import Language
    from spacy.tokens import Doc
except Exception:  # pragma: no cover
    Doc = object
    Language = object


class AsyncLinguisticPipeline:
    """Asynchronous linguistic analysis pipeline built around spaCy.

    - Accepts a list of analyzers (instances of BaseAnalyzer)
    - Produces a spaCy Doc once per text and runs analyzers concurrently
    - Injects language/model metadata into each analyzer result
    """

    def __init__(
        self, analyzers: Sequence[BaseAnalyzer], preload_languages: Iterable[str] | None = None
    ) -> None:
        self.analyzers: list[BaseAnalyzer] = list(analyzers)
        self.preload_languages: list[str] = list(preload_languages or [])

    async def _run_analyzer(
        self, analyzer: BaseAnalyzer, doc: Doc, lang: str
    ) -> LinguisticAnalysisResult:
        # Provide nlp to analyzers that optionally accept it (e.g., NER uses has_pipe)
        nlp_obj: Language | None = getattr(doc, "_.nlp", None)
        # Some analyzers may define a private _nlp slot; set defensively
        if hasattr(analyzer, "_nlp") and nlp_obj is not None:
            with contextlib.suppress(Exception):  # pragma: no cover
                analyzer._nlp = nlp_obj
        return await analyzer.analyze(doc, lang)

    async def analyze_text(
        self,
        text: str,
        lang_override: str | None,
        auto_download: bool,
        default_language: str | None = None,
    ) -> tuple[Doc, list[LinguisticAnalysisResult], dict[str, Any], dict[str, Any]]:
        # Start overall timing
        started_at = time.time()
        start_perf = time.perf_counter()

        # 0) Preload spaCy models if requested
        if self.preload_languages:
            for code in self.preload_languages:
                get_spacy_model(code, auto_download=auto_download)

        # 1) Determine language with precedence: CLI --lang > config > auto-detect
        logger = get_logger("rookeen.pipeline")

        if lang_override and lang_override.strip():
            lang_code = normalize_lang(lang_override)
            # Confidence is nominal when overridden
            lang_conf = 1.0
            logger.debug(
                "Using CLI language override",
                extra={"language": lang_code, "confidence": lang_conf},
            )
        elif default_language and default_language.strip():
            lang_code = normalize_lang(default_language)
            # Confidence is nominal when from config
            lang_conf = 1.0
            logger.debug(
                "Using config default language",
                extra={"language": lang_code, "confidence": lang_conf},
            )
        else:
            lang_code, lang_conf = detect_language(text)
            if lang_conf < 0.6:
                logger.warning(
                    "Low language detection confidence",
                    extra={"language": lang_code, "confidence": lang_conf, "threshold": 0.6},
                )
            logger.debug(
                "Using auto-detected language",
                extra={"language": lang_code, "confidence": lang_conf},
            )

        # 2) Load spaCy model
        nlp: Language = get_spacy_model(lang_code, auto_download=auto_download)

        # 3) Create Doc once
        doc: Doc = nlp(text)

        # 3.5) Dynamically add DependencyAnalyzer if parser is present and not already included
        analyzer_names = {getattr(a, "name", None) for a in self.analyzers}
        if (
            hasattr(nlp, "has_pipe")
            and nlp.has_pipe("parser")
            and "dependency" not in analyzer_names
        ):
            self.analyzers.append(DependencyAnalyzer())

        # 4) Run analyzers concurrently
        tasks = [self._run_analyzer(analyzer, doc, lang_code) for analyzer in self.analyzers]
        results: list[LinguisticAnalysisResult] = []
        if tasks:
            results = list(await asyncio.gather(*tasks))

        # 5) Inject metadata per result
        model_pkg = model_name_for(lang_code)
        for res in results:
            res.metadata = {
                **(res.metadata or {}),
                "language": {"code": lang_code, "confidence": lang_conf},
                "model": model_pkg,
            }
            # Ensure `name` is set (analyzers already set it by convention)
            if not getattr(res, "name", "") and hasattr(res, "analysis_type"):
                res.name = res.analysis_type.value

        # Calculate timing
        ended_at = time.time()
        total_seconds = time.perf_counter() - start_perf

        # 6) Return doc, results, context, timing
        context: dict[str, Any] = {
            "language": lang_code,
            "confidence": lang_conf,
            "model": model_pkg,
        }
        timing: dict[str, Any] = {
            "started_at": started_at,
            "ended_at": ended_at,
            "total_seconds": total_seconds,
        }
        return doc, results, context, timing

    async def analyze_web_page(
        self,
        url: str,
        lang_override: str | None = None,
        auto_download: bool = False,
        default_language: str | None = None,
        rate_limit: float = 0.5,
        robots_policy: str = "respect",
    ) -> tuple[
        WebPageContent, Doc, list[LinguisticAnalysisResult], dict[str, Any], dict[str, Any]
    ]:
        """Fetch a web page and analyze its content.

        Returns a tuple of `(content, doc, results, context, timing)` where `content` is a
        `WebPageContent` instance enriched with language fields by the pipeline, and `doc`
        is the spaCy Doc produced for the fetched text.
        """
        async with AsyncWebScraper(rate_limit=rate_limit, robots_policy=robots_policy) as scraper:
            content = await scraper.fetch_page(url)
        # Analyze the fetched text
        doc, results, context, timing = await self.analyze_text(
            content.text, lang_override, auto_download, default_language
        )
        # Enrich content with detected language fields
        content.language = context["language"]
        content.language_confidence = float(context["confidence"])
        return content, doc, results, context, timing
