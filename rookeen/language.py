from __future__ import annotations

import shutil
import subprocess
from typing import Any

from langdetect import DetectorFactory, detect_langs

# Ensure deterministic results from langdetect
DetectorFactory.seed = 0

try:
    import spacy
    from spacy.cli import download as spacy_download
except Exception:  # pragma: no cover - environment/import guard
    # Defer import errors until functions are called to provide clearer messages
    spacy = None
    spacy_download = None


SUPPORTED_LANGS = {"en", "de", "es", "fr"}

_LANG_TO_MODEL: dict[str, str] = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "es": "es_core_news_sm",
    "fr": "fr_core_news_sm",
}

_MODEL_CACHE: dict[str, spacy.Language] = {}


def normalize_lang(code: str) -> str:
    """Normalize language code to ISO 639-1 lower-case (e.g., en-US -> en).

    Falls back to best-effort mappings for common variants.
    """
    if not code:
        return "en"
    base = code.replace(" ", "").replace("_", "-").lower().split("-")[0]
    # Handle 3-letter codes and common aliases
    three_to_two = {
        "eng": "en",
        "deu": "de",
        "ger": "de",
        "spa": "es",
        "fra": "fr",
        "fre": "fr",
    }
    if len(base) == 3 and base in three_to_two:
        base = three_to_two[base]
    return base if len(base) == 2 else "en"


def model_name_for(code: str) -> str:
    """Return spaCy model package name for a given ISO 639-1 language code."""
    lang = normalize_lang(code)
    if lang not in _LANG_TO_MODEL:
        raise ValueError(f"Unsupported language '{code}'. Supported: {sorted(SUPPORTED_LANGS)}")
    return _LANG_TO_MODEL[lang]


def detect_language(text: str) -> tuple[str, float]:
    """Detect language code and confidence using langdetect.

    - Normalizes the returned code to ISO 639-1
    - Penalizes confidence for very short inputs (<200 chars)
    - Maps unknowns to a best guess within SUPPORTED_LANGS (defaults to 'en')
    """
    text = (text or "").strip()
    if not text:
        return "en", 0.0

    try:
        # detect_langs provides probabilities; choose the top candidate
        candidates = detect_langs(text)
        candidates.sort(key=lambda p: p.prob, reverse=True)
        top = candidates[0]
        code = normalize_lang(top.lang)
        confidence = float(top.prob)
    except Exception:
        # Any detection failure falls back to English with low confidence
        return "en", 0.3

    # Penalize very short texts
    if len(text) < 200:
        confidence = max(0.0, min(1.0, confidence * 0.8))

    # If detection not within supported set, map to closest guess (first two letters)
    if code not in SUPPORTED_LANGS:
        # Best-effort mapping by prefix; default to 'en'
        prefix = code[:2]
        code = prefix if prefix in SUPPORTED_LANGS else "en"
        confidence = min(confidence, 0.6)

    return code, confidence


def _pipify_model_name(model_pkg: str) -> str:
    """Convert spaCy model code 'en_core_web_sm' to pip package 'en-core-web-sm'."""
    return model_pkg.replace("_", "-")


def _install_spacy_model(model_pkg: str) -> bool:
    """Install a spaCy model package using available mechanisms.

    Tries spaCy's built-in downloader first (which uses pip), and if that
    fails due to missing pip or other issues, falls back to `uv pip install`.
    Returns True if installation succeeded, False otherwise.
    """
    # Ensure pip exists in the environment so spaCy downloader can work
    try:
        import pip  # noqa: F401

        pip_available = True
    except Exception:
        pip_available = False

    if not pip_available:
        uv_exe = shutil.which("uv")
        if uv_exe is not None:
            try:
                proc = subprocess.run(
                    [uv_exe, "pip", "install", "pip"], capture_output=True, text=True
                )
                pip_available = proc.returncode == 0
            except BaseException:
                pip_available = False

    # Prefer uv to install the model into the current project environment
    uv_exe = shutil.which("uv")
    if uv_exe is None:
        # If uv is unavailable, attempt spaCy downloader as a last resort
        try:
            if spacy_download is not None:
                spacy_download(model_pkg)
                return True
        except BaseException:
            return False
        return False

    pip_name = _pipify_model_name(model_pkg)
    try:
        proc = subprocess.run(
            [uv_exe, "pip", "install", pip_name],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return True
    except BaseException:
        pass

    # Fallback to spaCy downloader if uv pip failed
    try:
        if spacy_download is not None and pip_available:
            spacy_download(model_pkg)
            return True
    except BaseException:
        return False
    return False


def get_spacy_model(lang_code: str, auto_download: bool = False) -> Any:
    """Load and cache a spaCy language model for the given language code.

    If the model is not installed and auto_download is True, it will attempt to
    download the appropriate small model. Otherwise, raises a descriptive error.
    """
    if spacy is None:
        raise RuntimeError("spaCy is not installed. Please add 'spacy' to dependencies.")

    lang = normalize_lang(lang_code)
    if lang not in SUPPORTED_LANGS:
        raise ValueError(
            f"Unsupported language '{lang_code}'. Supported languages: {sorted(SUPPORTED_LANGS)}"
        )

    if lang in _MODEL_CACHE:
        return _MODEL_CACHE[lang]

    model_pkg = _LANG_TO_MODEL[lang]

    try:
        nlp = spacy.load(model_pkg)
    except Exception as load_err:
        if auto_download:
            installed = _install_spacy_model(model_pkg)
            if installed:
                try:
                    nlp = spacy.load(model_pkg)
                except Exception as second_err:
                    raise RuntimeError(
                        f"Installed '{model_pkg}' but failed to load it. Error: {second_err}"
                    ) from second_err
            else:
                raise RuntimeError(
                    "Failed to install spaCy model '"
                    + model_pkg
                    + "'. Try installing manually: `uv pip install "
                    + model_pkg
                    + "` (note pip package is '"
                    + _pipify_model_name(model_pkg)
                    + "') or `python -m spacy download "
                    + model_pkg
                    + "`."
                ) from load_err
        else:
            raise RuntimeError(
                "spaCy model '"
                + model_pkg
                + "' is not installed. Install it with: `uv pip install "
                + model_pkg
                + "` or call get_spacy_model(..., auto_download=True)."
            ) from load_err

    _MODEL_CACHE[lang] = nlp
    return nlp
