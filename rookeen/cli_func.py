from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys
import time
from collections.abc import Iterable
from importlib import metadata
from typing import Any

import click

from rookeen.config import RookeenSettings
from rookeen.errors import (
    FETCH,
    GENERIC,
    MODEL,
    USAGE,
    RookeenError,
    emit_and_exit,
)
from rookeen.export import doc_to_conllu, doc_to_spacy_json, dump_docbin
from rookeen.export.parquet import analyzers_to_parquet
from rookeen.language import normalize_lang
from rookeen.pipeline import AsyncLinguisticPipeline
from rookeen.utils.logging import get_logger, new_trace_id

# Exit codes per spec
EXIT_OK = 0
EXIT_GENERIC = 1
EXIT_USAGE = 2
EXIT_FETCH = 3
EXIT_MODEL = 4


def _get_version() -> str:
    try:
        return metadata.version("rookeen")
    except Exception:
        return "0.1.0"


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def _slugify_filename(text: str) -> str:
    safe = [c if c.isalnum() or c in ("-", "_") else "-" for c in text.strip()]
    s = "".join(safe).strip("-")
    return s or "output"


def _derive_output_base_from_url(url: str, output_dir: str = "results") -> str:
    try:
        from urllib.parse import urlparse

        p = urlparse(url)
        host = (p.netloc or "url").replace("www.", "")
        host_slug = _slugify_filename(host)
    except Exception:
        host_slug = "url"
    ts = int(time.time())
    return os.path.join(output_dir, f"{host_slug}_{ts}")


def _normalize_output_base(output: str | None, default_base: str) -> str:
    if not output:
        return default_base
    base = output
    if base.lower().endswith(".json"):
        base = base[:-5]
    return base


def _json_path_from_base(base: str) -> str:
    return base if base.lower().endswith(".json") else base + ".json"


def _spacy_json_path_from_base(base: str) -> str:
    return base + ".spacy.json"


def _docbin_path_from_base(base: str) -> str:
    return base + ".docbin"


def _conllu_path_from_base(base: str) -> str:
    return base + ".conllu"


def _write_json(path: str, payload: dict[str, Any]) -> None:
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _build_pipeline(
    preload_languages: Iterable[str],
    enable_embeddings: bool = False,
    enable_sentiment: bool = False,
    enabled_analyzers: list[str] | None = None,
    disabled_analyzers: list[str] | None = None,
) -> AsyncLinguisticPipeline:
    """Build pipeline with selective analyzer control."""
    from rookeen.analyzers.base import available_analyzers, get_analyzer

    # Try to register optional analyzers if specific flags are used
    if enable_embeddings:
        with contextlib.suppress(ImportError):
            from rookeen.analyzers.embeddings import EmbeddingsAnalyzer  # noqa: F401
    if enable_sentiment:
        with contextlib.suppress(ImportError):
            from rookeen.analyzers.sentiment import SentimentAnalyzer  # noqa: F401

    # Start with all available analyzers
    if enabled_analyzers is None or len(enabled_analyzers) == 0:
        enabled_analyzers = available_analyzers()

    # Apply disable filters
    if disabled_analyzers:
        enabled_analyzers = [name for name in enabled_analyzers if name not in disabled_analyzers]

    # Apply specific flags - ADD optional analyzers if explicitly requested
    if enable_embeddings and "embeddings" not in enabled_analyzers:
        enabled_analyzers.append("embeddings")
    elif not enable_embeddings and "embeddings" in enabled_analyzers:
        enabled_analyzers.remove("embeddings")

    if enable_sentiment and "sentiment" not in enabled_analyzers:
        enabled_analyzers.append("sentiment")
    elif not enable_sentiment and "sentiment" in enabled_analyzers:
        enabled_analyzers.remove("sentiment")

    # Instantiate analyzers
    analyzers = []
    for name in enabled_analyzers:
        try:
            analyzer_cls = get_analyzer(name)
            analyzers.append(analyzer_cls())
        except Exception:
            # Skip analyzers that can't be instantiated
            continue

    return AsyncLinguisticPipeline(analyzers, preload_languages=preload_languages)


def _results_to_json(
    *,
    source_type: str,
    source_value: str,
    language_code: str,
    language_conf: float,
    model_name: str,
    content_title: str,
    content_word_count: int,
    content_char_count: int,
    analyzers: list[Any],
    timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        from urllib.parse import urlparse

        domain = ""
        if source_type == "url":
            domain = urlparse(source_value).netloc
    except Exception:
        domain = ""

    payload: dict[str, Any] = {
        "tool": "rookeen",
        "version": _get_version(),
        "source": {
            "type": source_type,
            "value": source_value,
            "fetched_at": time.time(),
            "domain": domain,
        },
        "language": {
            "code": language_code,
            "confidence": float(language_conf),
            "model": model_name,
        },
        "content": {
            "title": content_title,
            "char_count": int(content_char_count),
            "word_count": int(content_word_count),
        },
        "analyzers": [
            {
                "name": getattr(r, "name", getattr(r, "analysis_type", "unknown")),
                "processing_time": float(getattr(r, "processing_time", 0.0)),
                "confidence": float(getattr(r, "confidence", 1.0)),
                "results": getattr(r, "results", {}),
                "metadata": getattr(r, "metadata", {}),
            }
            for r in analyzers
        ],
        "timing": timing or {},
    }
    return payload


def _make_logger(verbose: bool) -> logging.Logger:
    level = "DEBUG" if verbose else "INFO"
    return get_logger("rookeen.cli", level=level)


def _maybe_preload_embeddings(
    embeddings_backend: str | None,
    embeddings_model: str | None,
    openai_api_key: str | None,
) -> None:
    if not embeddings_backend:
        return
    try:
        from rookeen.analyzers.embeddings_backends import get_backend
    except Exception:
        return
    kwargs: dict[str, object] = {}
    if embeddings_backend == "miniLM":
        kwargs["model_name"] = (
            embeddings_model or "sentence-transformers/all-MiniLM-L6-v2"
        )
    elif embeddings_backend == "bge-m3":
        kwargs["model_name"] = embeddings_model or "BAAI/bge-m3"
    elif embeddings_backend == "openai-te3":
        kwargs["model_name"] = embeddings_model or os.getenv(
            "ROOKEEN_OPENAI_MODEL", "text-embedding-3-small"
        )
        kwargs["api_key"] = (
            openai_api_key
            or os.getenv("ROOKEEN_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
    try:
        be = get_backend(embeddings_backend, **kwargs)
        be.load()
    except Exception:
        # Best-effort preload; analyzer will still attempt lazily
        pass


def _parse_languages_csv(codes: str | None) -> list[str]:
    if not codes:
        return []
    langs = [normalize_lang(c.strip()) for c in codes.split(",") if c.strip()]
    seen: set[str] = set()
    out: list[str] = []
    for c in langs:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def analyze_url(
    url: str,
    output_base: str | None,
    format_: str | None,
    lang_override: str | None,
    preload_languages_csv: str | None,
    models_auto_download: bool | None,
    export_spacy_json: bool,
    export_docbin: bool,
    export_conllu: bool,
    conllu_engine: str,
    ud_auto_download: bool,
    allow_non_ud_conllu: bool,
    stdout: bool,
    trace_id: str | None,
    verbose: bool,
    export_parquet: bool,
    rate_limit: float,
    robots_policy: str,
    enable_embeddings: bool,
    enable_sentiment: bool,
    embeddings_backend: str | None,
    embeddings_model: str | None,
    openai_api_key: str | None,
    embeddings_preload: bool,
    enabled_analyzers: list[str] | None,
    disabled_analyzers: list[str] | None,
    settings: RookeenSettings,
) -> None:
    """Analyze a single URL."""
    logger = _make_logger(verbose)
    trace = trace_id or new_trace_id()

    # Persist CLI -> env for analyzer to resolve backend/model/api key
    if embeddings_backend:
        os.environ.setdefault("ROOKEEN_EMBEDDINGS_BACKEND", embeddings_backend)
    if embeddings_model:
        os.environ.setdefault("ROOKEEN_EMBEDDINGS_MODEL", embeddings_model)
    if openai_api_key:
        os.environ.setdefault("ROOKEEN_OPENAI_API_KEY", openai_api_key)

    # Optional preload to avoid first-call latency
    if embeddings_preload and (embeddings_backend or os.getenv("ROOKEEN_EMBEDDINGS_BACKEND")):
        _maybe_preload_embeddings(
            embeddings_backend or os.getenv("ROOKEEN_EMBEDDINGS_BACKEND"),
            embeddings_model or os.getenv("ROOKEEN_EMBEDDINGS_MODEL"),
            openai_api_key or os.getenv("ROOKEEN_OPENAI_API_KEY"),
        )

    effective_format = (format_ or settings.format).lower()
    preload = (
        _parse_languages_csv(preload_languages_csv)
        if preload_languages_csv is not None
        else list(settings.languages_preload)
    )
    pipeline = _build_pipeline(
        preload,
        enable_embeddings=enable_embeddings,
        enable_sentiment=enable_sentiment,
        enabled_analyzers=enabled_analyzers,
        disabled_analyzers=disabled_analyzers,
    )

    default_base = _derive_output_base_from_url(url, settings.output_dir)
    base = _normalize_output_base(output_base, default_base)
    json_path = _json_path_from_base(base)

    try:
        logger.debug(
            "starting_analysis",
            extra={"trace_id": trace, "run_id": trace, "url": url},
        )

        content, doc, results, ctx, timing = asyncio.run(
            pipeline.analyze_web_page(
                url,
                lang_override=lang_override,
                auto_download=(
                    models_auto_download
                    if models_auto_download is not None
                    else settings.models_auto_download
                ),
                default_language=settings.default_language or None,
                rate_limit=rate_limit,
                robots_policy=robots_policy,
            )
        )

        payload = _results_to_json(
            source_type="url",
            source_value=url,
            language_code=ctx["language"],
            language_conf=float(ctx["confidence"]),
            model_name=ctx["model"],
            content_title=content.title,
            content_word_count=content.word_count,
            content_char_count=content.char_count,
            analyzers=results,
            timing=timing,
        )

        if stdout:
            # Stream JSON to stdout for pipeline composition
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            _write_json(json_path, payload)
            if export_parquet:
                try:
                    parquet_path = base + ".parquet"
                    analyzers_to_parquet(payload["analyzers"], parquet_path)
                except Exception as exc:
                    logger.error(
                        "parquet_export_failed", extra={"trace_id": trace, "error": str(exc)}
                    )

        # Optional token-level exports (skip when using stdout)
        if not stdout:
            if export_spacy_json:
                spacy_json_path = _spacy_json_path_from_base(base)
                _ensure_dir(spacy_json_path)
                with open(spacy_json_path, "w", encoding="utf-8") as f:
                    json.dump(doc_to_spacy_json(doc), f, ensure_ascii=False, indent=2)
            if export_docbin:
                docbin_path = _docbin_path_from_base(base)
                _ensure_dir(docbin_path)
                dump_docbin(doc, docbin_path)
            if export_conllu:
                conllu_path = _conllu_path_from_base(base)
                _ensure_dir(conllu_path)
                engine = (conllu_engine or "auto").lower()
                if engine in ("auto", "stanza"):
                    try:
                        from rookeen.export.ud_conllu import text_to_conllu

                        raw_text = content.text
                        conllu_text = text_to_conllu(
                            raw_text, ctx["language"], auto_download=ud_auto_download
                        )
                        with open(conllu_path, "w", encoding="utf-8") as f:
                            f.write(conllu_text)
                    except Exception as e:
                        if not allow_non_ud_conllu:
                            raise RuntimeError(
                                "Stanza engine unavailable; install 'rookeen[ud]' or pass --allow-non-ud-conllu --conllu-engine basic"
                            ) from e
                        logger.warning(
                            "conllu_basic_fallback",
                            extra={
                                "trace_id": trace,
                                "run_id": trace,
                                "reason": str(e),
                            },
                        )
                        with open(conllu_path, "w", encoding="utf-8") as f:
                            f.write(doc_to_conllu(doc))
                elif engine == "basic":
                    logger.warning(
                        "conllu_basic_non_ud",
                        extra={
                            "trace_id": trace,
                            "run_id": trace,
                            "recommendation": "Use --conllu-engine stanza for UD-compliant output",
                        },
                    )
                    with open(conllu_path, "w", encoding="utf-8") as f:
                        f.write(doc_to_conllu(doc))

        if effective_format in ("md", "html", "all"):
            click.echo("Note: MD/HTML rendering not implemented in this step; JSON written.")

        logger.debug(
            "finished_analysis",
            extra={
                "trace_id": trace,
                "run_id": trace,
                "url": url,
                "output": json_path if not stdout else "stdout",
                "language": ctx["language"],
            },
        )
        if not stdout:
            click.echo(json_path)
        sys.exit(EXIT_OK)
    except ValueError as ve:
        logger.error("usage_error", extra={"trace_id": trace, "run_id": trace})
        emit_and_exit(RookeenError(USAGE.code, USAGE.name, f"{ve}"))
    except RuntimeError as re:
        msg = str(re)
        if "spaCy model" in msg or "Installed '" in msg:
            logger.error("model_error", extra={"trace_id": trace, "run_id": trace})
            emit_and_exit(RookeenError(MODEL.code, MODEL.name, msg))
        logger.error("generic_error", extra={"trace_id": trace, "run_id": trace})
        emit_and_exit(RookeenError(GENERIC.code, GENERIC.name, msg))
    except Exception as exc:
        try:
            import aiohttp

            if isinstance(exc, aiohttp.ClientError):
                logger.error("fetch_error", extra={"trace_id": trace, "run_id": trace})
                emit_and_exit(RookeenError(FETCH.code, FETCH.name, f"{exc}"))
        except Exception:
            pass
        logger.error("generic_error", extra={"trace_id": trace, "run_id": trace})
        emit_and_exit(RookeenError(GENERIC.code, GENERIC.name, f"{exc}"))


def analyze_stdin(
    output_base: str | None,
    format_: str | None,
    lang_override: str | None,
    preload_languages_csv: str | None,
    models_auto_download: bool | None,
    export_spacy_json: bool,
    export_docbin: bool,
    export_conllu: bool,
    conllu_engine: str,
    ud_auto_download: bool,
    allow_non_ud_conllu: bool,
    stdout: bool,
    trace_id: str | None,
    verbose: bool,
    export_parquet: bool,
    enable_embeddings: bool,
    enable_sentiment: bool,
    embeddings_backend: str | None,
    embeddings_model: str | None,
    openai_api_key: str | None,
    embeddings_preload: bool,
    enabled_analyzers: list[str] | None,
    disabled_analyzers: list[str] | None,
    settings: RookeenSettings,
) -> None:
    """Analyze text from stdin."""
    logger = _make_logger(verbose)
    trace = trace_id or new_trace_id()

    # Persist CLI -> env for analyzer to resolve backend/model/api key
    if embeddings_backend:
        os.environ.setdefault("ROOKEEN_EMBEDDINGS_BACKEND", embeddings_backend)
    if embeddings_model:
        os.environ.setdefault("ROOKEEN_EMBEDDINGS_MODEL", embeddings_model)
    if openai_api_key:
        os.environ.setdefault("ROOKEEN_OPENAI_API_KEY", openai_api_key)

    # Optional preload to avoid first-call latency
    if embeddings_preload and (embeddings_backend or os.getenv("ROOKEEN_EMBEDDINGS_BACKEND")):
        _maybe_preload_embeddings(
            embeddings_backend or os.getenv("ROOKEEN_EMBEDDINGS_BACKEND"),
            embeddings_model or os.getenv("ROOKEEN_EMBEDDINGS_MODEL"),
            openai_api_key or os.getenv("ROOKEEN_OPENAI_API_KEY"),
        )

    effective_format = (format_ or settings.format).lower()
    preload = (
        _parse_languages_csv(preload_languages_csv)
        if preload_languages_csv is not None
        else list(settings.languages_preload)
    )
    pipeline = _build_pipeline(
        preload,
        enable_embeddings=enable_embeddings,
        enable_sentiment=enable_sentiment,
        enabled_analyzers=enabled_analyzers,
        disabled_analyzers=disabled_analyzers,
    )

    try:
        # Read text from stdin
        text = sys.stdin.read()
    except Exception as exc:
        logger.error("usage_error", extra={"trace_id": trace, "run_id": trace})
        emit_and_exit(RookeenError(USAGE.code, USAGE.name, f"Failed to read from stdin: {exc}"))

    default_base = os.path.join(settings.output_dir, f"stdin_{int(time.time())}")
    base = _normalize_output_base(output_base, default_base)
    json_path = _json_path_from_base(base)

    try:
        logger.debug(
            "starting_stdin_analysis",
            extra={"trace_id": trace, "run_id": trace, "text_length": len(text)},
        )
        doc, results, ctx, timing = asyncio.run(
            pipeline.analyze_text(
                text,
                lang_override=lang_override,
                auto_download=(
                    models_auto_download
                    if models_auto_download is not None
                    else settings.models_auto_download
                ),
                default_language=settings.default_language or None,
            )
        )

        payload = _results_to_json(
            source_type="stdin",
            source_value="<stdin>",
            language_code=ctx["language"],
            language_conf=float(ctx["confidence"]),
            model_name=ctx["model"],
            content_title="<stdin>",
            content_word_count=len(text.split()),
            content_char_count=len(text),
            analyzers=results,
            timing=timing,
        )

        if stdout:
            # Stream JSON to stdout for pipeline composition
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            _write_json(json_path, payload)
            if export_parquet:
                try:
                    parquet_path = base + ".parquet"
                    analyzers_to_parquet(payload["analyzers"], parquet_path)
                except Exception as exc:
                    logger.error(
                        "parquet_export_failed", extra={"trace_id": trace, "error": str(exc)}
                    )

        # Optional token-level exports (skip when using stdout)
        if not stdout:
            if export_spacy_json:
                spacy_json_path = _spacy_json_path_from_base(base)
                _ensure_dir(spacy_json_path)
                with open(spacy_json_path, "w", encoding="utf-8") as f:
                    json.dump(doc_to_spacy_json(doc), f, ensure_ascii=False, indent=2)
            if export_docbin:
                docbin_path = _docbin_path_from_base(base)
                _ensure_dir(docbin_path)
                dump_docbin(doc, docbin_path)
            if export_conllu:
                conllu_path = _conllu_path_from_base(base)
                _ensure_dir(conllu_path)
                engine = (conllu_engine or "auto").lower()
                if engine in ("auto", "stanza"):
                    try:
                        from rookeen.export.ud_conllu import text_to_conllu

                        conllu_text = text_to_conllu(
                            text, ctx["language"], auto_download=ud_auto_download
                        )
                        with open(conllu_path, "w", encoding="utf-8") as f:
                            f.write(conllu_text)
                    except Exception as e:
                        if not allow_non_ud_conllu:
                            raise RuntimeError(
                                "Stanza engine unavailable; install 'rookeen[ud]' or pass --allow-non-ud-conllu --conllu-engine basic"
                            ) from e
                        logger.warning(
                            "conllu_basic_fallback",
                            extra={
                                "trace_id": trace,
                                "run_id": trace,
                                "reason": str(e),
                            },
                        )
                        with open(conllu_path, "w", encoding="utf-8") as f:
                            f.write(doc_to_conllu(doc))
                elif engine == "basic":
                    logger.warning(
                        "conllu_basic_non_ud",
                        extra={
                            "trace_id": trace,
                            "run_id": trace,
                            "recommendation": "Use --conllu-engine stanza for UD-compliant output",
                        },
                    )
                    with open(conllu_path, "w", encoding="utf-8") as f:
                        f.write(doc_to_conllu(doc))

        if effective_format in ("md", "html", "all"):
            click.echo("Note: MD/HTML rendering not implemented in this step; JSON written.")

        logger.debug(
            "finished_stdin_analysis",
            extra={
                "trace_id": trace,
                "run_id": trace,
                "output": json_path if not stdout else "stdout",
                "language": ctx["language"],
            },
        )
        if not stdout:
            click.echo(json_path)
        sys.exit(EXIT_OK)
    except ValueError as ve:
        logger.error("usage_error", extra={"trace_id": trace, "run_id": trace})
        emit_and_exit(RookeenError(USAGE.code, USAGE.name, f"{ve}"))
    except RuntimeError as re:
        msg = str(re)
        if "spaCy model" in msg or "Installed '" in msg:
            logger.error("model_error", extra={"trace_id": trace, "run_id": trace})
            emit_and_exit(RookeenError(MODEL.code, MODEL.name, msg))
        logger.error("generic_error", extra={"trace_id": trace, "run_id": trace})
        emit_and_exit(RookeenError(GENERIC.code, GENERIC.name, msg))
    except Exception as exc:
        logger.error("generic_error", extra={"trace_id": trace, "run_id": trace})
        emit_and_exit(RookeenError(GENERIC.code, GENERIC.name, f"{exc}"))


def analyze_file(
    path: str,
    output_base: str | None,
    format_: str | None,
    lang_override: str | None,
    preload_languages_csv: str | None,
    models_auto_download: bool | None,
    export_spacy_json: bool,
    export_docbin: bool,
    export_conllu: bool,
    conllu_engine: str,
    ud_auto_download: bool,
    allow_non_ud_conllu: bool,
    stdout: bool,
    trace_id: str | None,
    verbose: bool,
    export_parquet: bool,
    enable_embeddings: bool,
    enable_sentiment: bool,
    embeddings_backend: str | None,
    embeddings_model: str | None,
    openai_api_key: str | None,
    embeddings_preload: bool,
    enabled_analyzers: list[str] | None,
    disabled_analyzers: list[str] | None,
    settings: RookeenSettings,
) -> None:
    """Analyze a local text file."""
    logger = _make_logger(verbose)
    trace = trace_id or new_trace_id()

    # Persist CLI -> env for analyzer to resolve backend/model/api key
    if embeddings_backend:
        os.environ.setdefault("ROOKEEN_EMBEDDINGS_BACKEND", embeddings_backend)
    if embeddings_model:
        os.environ.setdefault("ROOKEEN_EMBEDDINGS_MODEL", embeddings_model)
    if openai_api_key:
        os.environ.setdefault("ROOKEEN_OPENAI_API_KEY", openai_api_key)

    # Optional preload to avoid first-call latency
    if embeddings_preload and (embeddings_backend or os.getenv("ROOKEEN_EMBEDDINGS_BACKEND")):
        _maybe_preload_embeddings(
            embeddings_backend or os.getenv("ROOKEEN_EMBEDDINGS_BACKEND"),
            embeddings_model or os.getenv("ROOKEEN_EMBEDDINGS_MODEL"),
            openai_api_key or os.getenv("ROOKEEN_OPENAI_API_KEY"),
        )

    effective_format = (format_ or settings.format).lower()
    preload = (
        _parse_languages_csv(preload_languages_csv)
        if preload_languages_csv is not None
        else list(settings.languages_preload)
    )
    pipeline = _build_pipeline(
        preload,
        enable_embeddings=enable_embeddings,
        enable_sentiment=enable_sentiment,
        enabled_analyzers=enabled_analyzers,
        disabled_analyzers=disabled_analyzers,
    )

    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except Exception as exc:
        logger.error("usage_error", extra={"trace_id": trace, "run_id": trace})
        emit_and_exit(RookeenError(USAGE.code, USAGE.name, f"Failed to read file: {exc}"))

    default_base = os.path.join(settings.output_dir, _slugify_filename(os.path.basename(path)))
    base = _normalize_output_base(output_base, default_base)
    json_path = _json_path_from_base(base)

    try:
        logger.debug(
            "starting_file_analysis",
            extra={"trace_id": trace, "run_id": trace, "path": os.path.abspath(path)},
        )
        doc, results, ctx, timing = asyncio.run(
            pipeline.analyze_text(
                text,
                lang_override=lang_override,
                auto_download=(
                    models_auto_download
                    if models_auto_download is not None
                    else settings.models_auto_download
                ),
                default_language=settings.default_language or None,
            )
        )

        payload = _results_to_json(
            source_type="file",
            source_value=os.path.abspath(path),
            language_code=ctx["language"],
            language_conf=float(ctx["confidence"]),
            model_name=ctx["model"],
            content_title=os.path.basename(path),
            content_word_count=len(text.split()),
            content_char_count=len(text),
            analyzers=results,
            timing=timing,
        )

        if stdout:
            # Stream JSON to stdout for pipeline composition
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            _write_json(json_path, payload)
            if export_parquet:
                try:
                    parquet_path = base + ".parquet"
                    analyzers_to_parquet(payload["analyzers"], parquet_path)
                except Exception as exc:
                    logger.error(
                        "parquet_export_failed", extra={"trace_id": trace, "error": str(exc)}
                    )

        # Optional token-level exports (skip when using stdout)
        if not stdout:
            if export_spacy_json:
                spacy_json_path = _spacy_json_path_from_base(base)
                _ensure_dir(spacy_json_path)
                with open(spacy_json_path, "w", encoding="utf-8") as f:
                    json.dump(doc_to_spacy_json(doc), f, ensure_ascii=False, indent=2)
            if export_docbin:
                docbin_path = _docbin_path_from_base(base)
                _ensure_dir(docbin_path)
                dump_docbin(doc, docbin_path)
            if export_conllu:
                conllu_path = _conllu_path_from_base(base)
                _ensure_dir(conllu_path)
                engine = (conllu_engine or "auto").lower()
                if engine in ("auto", "stanza"):
                    try:
                        from rookeen.export.ud_conllu import text_to_conllu

                        conllu_text = text_to_conllu(
                            text, ctx["language"], auto_download=ud_auto_download
                        )
                        with open(conllu_path, "w", encoding="utf-8") as f:
                            f.write(conllu_text)
                    except Exception as e:
                        if not allow_non_ud_conllu:
                            raise RuntimeError(
                                "Stanza engine unavailable; install 'rookeen[ud]' or pass --allow-non-ud-conllu --conllu-engine basic"
                            ) from e
                        logger.warning(
                            "conllu_basic_fallback",
                            extra={
                                "trace_id": trace,
                                "run_id": trace,
                                "reason": str(e),
                            },
                        )
                        with open(conllu_path, "w", encoding="utf-8") as f:
                            f.write(doc_to_conllu(doc))
                elif engine == "basic":
                    logger.warning(
                        "conllu_basic_non_ud",
                        extra={
                            "trace_id": trace,
                            "run_id": trace,
                            "recommendation": "Use --conllu-engine stanza for UD-compliant output",
                        },
                    )
                    with open(conllu_path, "w", encoding="utf-8") as f:
                        f.write(doc_to_conllu(doc))

        if effective_format in ("md", "html", "all"):
            click.echo("Note: MD/HTML rendering not implemented in this step; JSON written.")

        logger.debug(
            "finished_file_analysis",
            extra={
                "trace_id": trace,
                "run_id": trace,
                "path": os.path.abspath(path),
                "output": json_path if not stdout else "stdout",
                "language": ctx["language"],
            },
        )
        if not stdout:
            click.echo(json_path)
        sys.exit(EXIT_OK)
    except ValueError as ve:
        logger.error("usage_error", extra={"trace_id": trace, "run_id": trace})
        emit_and_exit(RookeenError(USAGE.code, USAGE.name, f"{ve}"))
    except RuntimeError as re:
        msg = str(re)
        if "spaCy model" in msg or "Installed '" in msg:
            logger.error("model_error", extra={"trace_id": trace, "run_id": trace})
            emit_and_exit(RookeenError(MODEL.code, MODEL.name, msg))
        logger.error("generic_error", extra={"trace_id": trace, "run_id": trace})
        emit_and_exit(RookeenError(GENERIC.code, GENERIC.name, msg))
    except Exception as exc:
        logger.error("generic_error", extra={"trace_id": trace, "run_id": trace})
        emit_and_exit(RookeenError(GENERIC.code, GENERIC.name, f"{exc}"))


def batch_analyze(
    url_list_file: str,
    output_dir: str | None,
    format_: str | None,
    lang_override: str | None,
    preload_languages_csv: str | None,
    models_auto_download: bool | None,
    export_spacy_json: bool,
    export_docbin: bool,
    export_conllu: bool,
    conllu_engine: str,
    ud_auto_download: bool,
    allow_non_ud_conllu: bool,
    trace_id: str | None,
    verbose: bool,
    export_parquet: bool,
    rate_limit: float,
    robots_policy: str,
    enable_embeddings: bool,
    enable_sentiment: bool,
    embeddings_backend: str | None,
    embeddings_model: str | None,
    openai_api_key: str | None,
    embeddings_preload: bool,
    enabled_analyzers: list[str] | None,
    disabled_analyzers: list[str] | None,
    settings: RookeenSettings,
) -> None:
    """Analyze a list of URLs from a file (one per line; '#' comments allowed)."""
    logger = _make_logger(verbose)
    trace = trace_id or new_trace_id()
    effective_output_dir = output_dir or settings.output_dir
    # effective_format = (format_ or settings.format).lower()

    # Persist CLI -> env for analyzer to resolve backend/model/api key
    if embeddings_backend:
        os.environ.setdefault("ROOKEEN_EMBEDDINGS_BACKEND", embeddings_backend)
    if embeddings_model:
        os.environ.setdefault("ROOKEEN_EMBEDDINGS_MODEL", embeddings_model)
    if openai_api_key:
        os.environ.setdefault("ROOKEEN_OPENAI_API_KEY", openai_api_key)

    # Optional preload to avoid first-call latency
    if embeddings_preload and (embeddings_backend or os.getenv("ROOKEEN_EMBEDDINGS_BACKEND")):
        _maybe_preload_embeddings(
            embeddings_backend or os.getenv("ROOKEEN_EMBEDDINGS_BACKEND"),
            embeddings_model or os.getenv("ROOKEEN_EMBEDDINGS_MODEL"),
            openai_api_key or os.getenv("ROOKEEN_OPENAI_API_KEY"),
        )

    preload = (
        _parse_languages_csv(preload_languages_csv)
        if preload_languages_csv is not None
        else list(settings.languages_preload)
    )
    pipeline = _build_pipeline(
        preload,
        enable_embeddings=enable_embeddings,
        enable_sentiment=enable_sentiment,
        enabled_analyzers=enabled_analyzers,
        disabled_analyzers=disabled_analyzers,
    )

    try:
        with open(url_list_file, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines()]
    except Exception as exc:
        logger.error("usage_error", extra={"trace_id": trace, "run_id": trace})
        emit_and_exit(RookeenError(USAGE.code, USAGE.name, f"Failed to read URL list: {exc}"))

    urls = [ln for ln in lines if ln and not ln.startswith("#")]
    if not urls:
        logger.error("usage_error", extra={"trace_id": trace, "run_id": trace})
        emit_and_exit(RookeenError(USAGE.code, USAGE.name, "No URLs to process."))

    os.makedirs(effective_output_dir, exist_ok=True)

    failures = 0
    for url in urls:
        base = os.path.join(
            effective_output_dir,
            _derive_output_base_from_url(url, effective_output_dir).split("/", 1)[-1],
        )
        json_path = _json_path_from_base(base)
        try:
            logger.debug(
                "starting_batch_item",
                extra={"trace_id": trace, "run_id": trace, "url": url},
            )
            content, doc, results, ctx, timing = asyncio.run(
                pipeline.analyze_web_page(
                    url,
                    lang_override=lang_override,
                    auto_download=(
                        models_auto_download
                        if models_auto_download is not None
                        else settings.models_auto_download
                    ),
                    default_language=settings.default_language or None,
                    rate_limit=rate_limit,
                    robots_policy=robots_policy,
                )
            )
            payload = _results_to_json(
                source_type="url",
                source_value=url,
                language_code=ctx["language"],
                language_conf=float(ctx["confidence"]),
                model_name=ctx["model"],
                content_title=content.title,
                content_word_count=content.word_count,
                content_char_count=content.char_count,
                analyzers=results,
                timing=timing,
            )
            _write_json(json_path, payload)
            if export_parquet:
                try:
                    parquet_path = base + ".parquet"
                    analyzers_to_parquet(payload["analyzers"], parquet_path)
                except Exception as exc:
                    logger.error(
                        "parquet_export_failed",
                        extra={"trace_id": trace, "url": url, "error": str(exc)},
                    )
            if export_spacy_json:
                spacy_json_path = _spacy_json_path_from_base(base)
                _ensure_dir(spacy_json_path)
                with open(spacy_json_path, "w", encoding="utf-8") as f:
                    json.dump(doc_to_spacy_json(doc), f, ensure_ascii=False, indent=2)
            if export_docbin:
                docbin_path = _docbin_path_from_base(base)
                _ensure_dir(docbin_path)
                dump_docbin(doc, docbin_path)
            if export_conllu:
                conllu_path = _conllu_path_from_base(base)
                _ensure_dir(conllu_path)
                engine = (conllu_engine or "auto").lower()
                if engine in ("auto", "stanza"):
                    try:
                        from rookeen.export.ud_conllu import text_to_conllu

                        raw_text = content.text
                        conllu_text = text_to_conllu(
                            raw_text, ctx["language"], auto_download=ud_auto_download
                        )
                        with open(conllu_path, "w", encoding="utf-8") as f:
                            f.write(conllu_text)
                    except Exception as e:
                        if not allow_non_ud_conllu:
                            raise RuntimeError(
                                "Stanza engine unavailable; install 'rookeen[ud]' or pass --allow-non-ud-conllu --conllu-engine basic"
                            ) from e
                        logger.warning(
                            "conllu_basic_fallback",
                            extra={
                                "trace_id": trace,
                                "run_id": trace,
                                "reason": str(e),
                                "url": url,
                            },
                        )
                        with open(conllu_path, "w", encoding="utf-8") as f:
                            f.write(doc_to_conllu(doc))
                elif engine == "basic":
                    logger.warning(
                        "conllu_basic_non_ud",
                        extra={
                            "trace_id": trace,
                            "run_id": trace,
                            "recommendation": "Use --conllu-engine stanza for UD-compliant output",
                            "url": url,
                        },
                    )
                    with open(conllu_path, "w", encoding="utf-8") as f:
                        f.write(doc_to_conllu(doc))
            logger.debug(
                "finished_batch_item",
                extra={
                    "trace_id": trace,
                    "run_id": trace,
                    "url": url,
                    "output": json_path,
                    "language": ctx["language"],
                },
            )
            click.echo(json_path)
        except Exception as exc:
            failures += 1
            logger.error(
                "batch_item_error",
                extra={"trace_id": trace, "run_id": trace, "url": url, "error": str(exc)},
            )

    sys.exit(EXIT_OK if failures == 0 else EXIT_GENERIC)

