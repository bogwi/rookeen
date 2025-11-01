from __future__ import annotations

import os
import threading
from typing import Any

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Cache of initialized Stanza pipelines keyed by normalized language code
_PIPELINES: dict[str, Any] = {}
_PIPELINES_LOCK = threading.Lock()


def _normalize_lang(lang: str | None) -> str:
    if not lang or not lang.strip():
        return "en"
    # Normalize like "en", "en-US" -> "en"
    return lang.strip().lower().split("-")[0]


def ensure_stanza_pipeline(lang: str, auto_download: bool = True) -> Any:
    """
    Ensure a cached Stanza pipeline for the given language and return it.

    The pipeline includes processors required for UD-compliant output: tokenize, mwt,
    pos, lemma, depparse. Downloads models on-demand when `auto_download` is True.

    Environment variables:
    - ROOKEEN_STANZA_USE_GPU: if set to "1" truthy, requests GPU usage for pipeline
    - ROOKEEN_STANZA_VERBOSE: if set to "1" truthy, enables Stanza verbose output

    Args:
        lang: IETF/ISO language code (e.g., "en", "en-US"). Only the base part is used.
        auto_download: Whether to attempt an automatic stanza.download for models.

    Returns:
        A Stanza Pipeline instance.

    Raises:
        RuntimeError: If Stanza is not installed or pipeline initialization fails.
    """
    normalized_lang = _normalize_lang(lang)

    with _PIPELINES_LOCK:
        if normalized_lang in _PIPELINES:
            return _PIPELINES[normalized_lang]

        try:
            import stanza
        except Exception as exc:  # pragma: no cover - import resolution
            raise RuntimeError(
                "stanza is required for UD-native export. Install with `pip install stanza` or `pip install rookeen[ud]`."
            ) from exc

        processors = os.getenv(
            "ROOKEEN_STANZA_PROCESSORS",
            "tokenize,mwt,pos,lemma,depparse",
        )
        use_gpu_env = os.getenv("ROOKEEN_STANZA_USE_GPU", "0").strip().lower()
        use_gpu = use_gpu_env in {"1", "true", "yes", "on"}
        verbose_env = os.getenv("ROOKEEN_STANZA_VERBOSE", "0").strip().lower()
        verbose = verbose_env in {"1", "true", "yes", "on"}

        if auto_download:
            try:
                stanza.download(normalized_lang, processors=processors, verbose=False)
            except Exception as exc:
                # Best-effort; proceed to pipeline initialization which may still succeed
                logger.debug(
                    "stanza.download failed or skipped",
                    extra={
                        "lang": normalized_lang,
                        "processors": processors,
                        "error": str(exc),
                    },
                )

        try:
            pipeline = stanza.Pipeline(
                lang=normalized_lang,
                processors=processors,
                tokenize_pretokenized=False,
                use_gpu=use_gpu,
                verbose=verbose,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize Stanza pipeline for lang '{normalized_lang}'."
            ) from exc

        _PIPELINES[normalized_lang] = pipeline
        return pipeline


def reset_stanza_pipelines() -> None:
    """Clear the cached Stanza pipelines. Useful for tests."""
    with _PIPELINES_LOCK:
        _PIPELINES.clear()


def _escape(text: str | None) -> str:
    if not text or not text.strip():
        return "_"
    return text.replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()


def text_to_conllu(text: str, lang: str, auto_download: bool = True) -> str:
    """
    Convert raw text to UD-valid CoNLL-U using Stanza.

    This function uses Stanza's built-in CoNLL-U conversion utilities to ensure
    proper formatting and compliance with UD standards, including correct handling
    of SpaceAfter annotations in the MISC field.

    Args:
        text: Raw input text.
        lang: Language code (e.g., "en", "en-US").
        auto_download: Attempt to download Stanza models if missing.

    Returns:
        CoNLL-U formatted string ending with a double newline.
    """
    if text is None:
        raise ValueError("text must be a non-None string")

    nlp = ensure_stanza_pipeline(lang, auto_download=auto_download)
    doc = nlp(text)

    # Use Stanza's built-in CoNLL-U conversion
    try:
        import os
        import tempfile

        from stanza.utils.conll import CoNLL

        # Use Stanza's write_doc2conll to generate UD-compliant CoNLL-U
        # This handles SpaceAfter annotations and other UD compliance issues automatically
        with tempfile.NamedTemporaryFile(mode="w", suffix=".conllu", delete=False) as temp_file:
            temp_filename = temp_file.name

        try:
            CoNLL.write_doc2conll(doc, temp_filename)

            # Read the generated CoNLL-U content
            with open(temp_filename, encoding="utf-8") as f:
                conll_content = f.read()

            # Post-process to fix feature sorting (Stanza doesn't sort features alphabetically)
            conll_content = _fix_feature_sorting(conll_content)

            # Ensure proper ending with double newline
            if not conll_content.endswith("\n\n"):
                conll_content += "\n"

            return conll_content

        finally:
            # Clean up temporary file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

    except ImportError:
        # Fallback to manual construction if CoNLL utilities are not available
        logger.warning(
            "stanza.utils.conll not available, falling back to manual CoNLL-U construction"
        )
        return _manual_conllu_construction(doc, text)


def _fix_feature_sorting(conll_content: str) -> str:
    """
    Post-process CoNLL-U content to sort morphological features alphabetically.

    The UD format requires features to be sorted alphabetically, but Stanza's
    output doesn't always follow this requirement.
    """
    lines = conll_content.split("\n")
    fixed_lines = []

    for line in lines:
        if line.startswith("#") or not line.strip():
            # Comments and empty lines pass through unchanged
            fixed_lines.append(line)
        else:
            # Parse CoNLL-U line
            fields = line.split("\t")
            if len(fields) == 10:
                # Field 5 is FEATS (morphological features)
                feats = fields[5]
                if feats and feats != "_":
                    # Sort features alphabetically (case-insensitive, as required by UD)
                    feature_pairs = feats.split("|")
                    feature_pairs.sort(key=lambda x: x.lower())
                    fields[5] = "|".join(feature_pairs)

                fixed_lines.append("\t".join(fields))
            else:
                # Malformed line, pass through unchanged
                fixed_lines.append(line)

    return "\n".join(fixed_lines)


def _manual_conllu_construction(doc: Any, original_text: str) -> str:
    """
    Fallback manual CoNLL-U construction for when Stanza's CoNLL utilities are not available.

    This is a simplified version that may not handle all edge cases as well as Stanza's
    built-in utilities, but provides basic CoNLL-U output.
    """
    lines: list[str] = []

    for sent_id, sent in enumerate(doc.sentences):
        # Sentence comments
        lines.append(f"# sent_id = {sent_id}")
        lines.append(f"# text = {_escape(getattr(sent, 'text', ''))}")

        # Multi-word tokens (MWT) emitted before word lines
        # Stanza token.id can be int or a (start, end) tuple for MWTs
        for token in sent.tokens:
            tok_id = token.id
            try:
                if isinstance(tok_id, tuple) and len(tok_id) == 2:
                    start, end = tok_id
                    # token.text contains the surface form spanning the range
                    lines.append(f"{start}-{end}\t{_escape(token.text)}\t_\t_\t_\t_\t_\t_\t_\t_")
            except Exception:
                # Ignore unexpected token id shapes
                pass

        # Word-level annotations (one per ID)
        for word in sent.words:
            wid = word.id
            form = getattr(word, "text", None)
            lemma = getattr(word, "lemma", None)
            upos = getattr(word, "upos", None)
            xpos = getattr(word, "xpos", None)
            feats = getattr(word, "feats", None)
            head = getattr(word, "head", None)
            deprel = getattr(word, "deprel", None)
            misc = getattr(word, "misc", None)

            lines.append(
                "\t".join(
                    [
                        str(wid),
                        _escape(form),
                        _escape(lemma),
                        _escape(upos),
                        _escape(xpos),
                        _escape(feats),
                        str(head if head is not None else 0),
                        _escape(deprel),
                        "_",
                        _escape(misc),
                    ]
                )
            )

        # Sentence separator
        lines.append("")

    out = "\n".join(lines)
    if not out.endswith("\n\n"):
        out += "\n"
    return out


__all__ = [
    "ensure_stanza_pipeline",
    "text_to_conllu",
    "reset_stanza_pipelines",
]
