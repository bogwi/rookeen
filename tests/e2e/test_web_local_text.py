from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from .helpers import (
    assert_analyzer_has_model_metadata,
    assert_json_key,
    build_cli_cmd,
    extract_json_from_stdout,
    find_analyzer,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "test_data" / "web"
SNAPSHOT_PATH = DATA_DIR / "noisy_html_en.snapshot.json"


def _round_floats(value: Any, ndigits: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, ndigits)
    if isinstance(value, list):
        return [_round_floats(item, ndigits) for item in value]
    if isinstance(value, dict):
        return {key: _round_floats(val, ndigits) for key, val in value.items()}
    return value


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    analyzers: dict[str, Any] = {}
    for analyzer in payload.get("analyzers", []):
        name = analyzer.get("name")
        if not name:
            continue
        sanitized_results = _round_floats(analyzer.get("results", {}))
        analyzers[name] = {
            "confidence": analyzer.get("confidence"),
            "metadata": _round_floats(analyzer.get("metadata", {})),
            "results": sanitized_results,
        }

    source = payload.get("source", {}).copy()
    source.pop("fetched_at", None)

    return {
        "language": {
            "code": payload.get("language", {}).get("code"),
            "model": payload.get("language", {}).get("model"),
        },
        "content": {
            "title": payload.get("content", {}).get("title"),
            "char_count": payload.get("content", {}).get("char_count"),
            "word_count": payload.get("content", {}).get("word_count"),
        },
        "source": source,
        "analyzers": analyzers,
    }


def test_web_local_plain_text() -> None:
    text = (DATA_DIR / "wiki_nlp_en.txt").read_text(encoding="utf-8")
    cmd = build_cli_cmd(
        "analyze",
        "--stdin",
        "--lang",
        "en",
        "--stdout",
    )
    proc = subprocess.run(cmd, input=text, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    payload = extract_json_from_stdout(proc.stdout)

    # Basic structure and language
    assert_json_key(payload, "tool")
    assert_json_key(payload, "language.code")
    assert payload["language"]["code"] == "en"

    content = payload.get("content", {})
    assert content == {
        "title": "<stdin>",
        "char_count": 406,
        "word_count": 54,
    }

    # Lexical stats should be non-trivial
    lex = find_analyzer(payload, "lexical_stats")
    total_tokens = int(lex.get("results", {}).get("total_tokens", 0))
    assert total_tokens == 37

    lex_results = lex["results"]
    assert lex_results["unique_lemmas"] == 26
    assert lex_results["sentences"] == 1
    assert pytest.approx(lex_results["type_token_ratio"], rel=1e-4) == 0.7027
    top_lemmas = {lemma: count for lemma, count in lex_results["top_lemmas"]}
    assert top_lemmas.get("language") == 5
    assert len(top_lemmas) >= 15

    # Metadata present on core analyzers
    for name in ("pos", "ner", "lexical_stats", "readability", "keywords"):
        assert_analyzer_has_model_metadata(payload, name)

    ner = find_analyzer(payload, "ner")
    assert ner["results"]["total_entities"] == 3
    assert "ORG" in ner["results"]["counts_by_label"]

    keywords = find_analyzer(payload, "keywords")
    kw_entries = keywords["results"]["keywords"]
    assert isinstance(kw_entries, list) and len(kw_entries) >= 10
    assert any(word.lower() == "language" for word, _score in kw_entries)

    readability = find_analyzer(payload, "readability")
    readability_results = readability["results"]
    assert readability_results["supported"] is True
    assert pytest.approx(readability_results["flesch_kincaid_grade"], rel=1e-3) == 28.7438

    timing = payload.get("timing", {})
    assert timing and timing["total_seconds"] > 0
    
def test_web_local_noisy_html_stdin() -> None:
    raw_html = (DATA_DIR / "noisy_html_en.txt").read_text(encoding="utf-8")

    cmd = build_cli_cmd(
        "analyze",
        "--stdin",
        "--lang",
        "en",
        "--stdout",
    )
    proc = subprocess.run(cmd, input=raw_html, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    payload = extract_json_from_stdout(proc.stdout)

    # Ensure content was processed (word/char counts present)
    content = payload.get("content", {})
    assert content["word_count"] >= 900
    assert content["char_count"] >= 13000

    # HTML noise should not leak literal tags in results
    assert "<script" not in payload["source"]["value"]

    lex = find_analyzer(payload, "lexical_stats")
    lex_results = lex["results"]
    assert lex_results["total_tokens"] >= 500
    assert lex_results["sentences"] >= 20
    top_lemmas = {lemma for lemma, _count in lex_results["top_lemmas"][:50]}
    assert {"bert", "model", "nlp"}.issubset({lemma.lower() for lemma in top_lemmas})

    # NLP analyzers should surface relevant data despite markup
    ner = find_analyzer(payload, "ner")
    assert ner["results"]["total_entities"] >= 20

    keywords = find_analyzer(payload, "keywords")
    kw_words = {word.lower() for word, _score in keywords["results"].get("keywords", [])}
    assert "bert" in kw_words
    assert "nlp" in kw_words
    assert "div" in kw_words  # Noise still present, ensures YAKE sees tags

    # Readability present
    readability = find_analyzer(payload, "readability")
    assert readability["results"]["supported"] is True
    assert readability["results"]["flesch_reading_ease"] < 0

    current = _normalize_payload(payload)
    with SNAPSHOT_PATH.open(encoding="utf-8") as fh:
        expected = json.load(fh)
    assert current == expected

