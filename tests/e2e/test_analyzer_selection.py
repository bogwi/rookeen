from __future__ import annotations

from pathlib import Path

import pytest

from .helpers import (
    assert_analyzer_has_model_metadata,
    assert_json_key,
    assert_lang,
    build_cli_cmd,
    find_analyzer,
    read_json,
    run_cli,
    unique_output_base,
)

URL = "https://en.wikipedia.org/w/index.php?title=Natural_language_processing&oldid=1201524046"

# Networked Wikipedia; exclude from default runs
pytestmark = [pytest.mark.external]


def test_analyzer_selection_default() -> None:
    """Test default analyzer behavior (all available analyzers)."""
    out_base: Path = unique_output_base("analyzer-selection-default")
    out_json = str(out_base) + ".json"

    args = build_cli_cmd(
        "analyze",
        URL,
        "--format",
        "json",
        "--models-auto-download",
        "--robots",
        "ignore",
        "-o",
        out_json,
    )
    proc = run_cli(args)
    assert proc.returncode == 0, (
        f"CLI failed: {proc.returncode}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
    )

    payload = read_json(out_json)

    # Check basic structure
    assert_lang(payload, "en")
    assert_json_key(payload, "tool")
    assert_json_key(payload, "version")
    assert_json_key(payload, "source")
    assert_json_key(payload, "language")
    assert_json_key(payload, "content")
    assert_json_key(payload, "analyzers")

    analyzers = payload.get("analyzers", [])
    assert len(analyzers) >= 5  # Should have at least core analyzers

    # Check that all expected analyzers are present
    analyzer_names = {a["name"] for a in analyzers}
    expected_core = {"dependency", "keywords", "lexical_stats", "ner", "pos", "readability"}
    assert expected_core.issubset(analyzer_names), (
        f"Missing core analyzers. Found: {analyzer_names}"
    )

    # Validate core analyzer results
    for name in expected_core:
        analyzer = find_analyzer(payload, name)
        assert "results" in analyzer
        assert "metadata" in analyzer
        assert_analyzer_has_model_metadata(payload, name)


def test_analyzer_selection_specific_enable() -> None:
    """Test selective analyzer enabling."""
    out_base: Path = unique_output_base("analyzer-selection-enable")
    out_json = str(out_base) + ".json"

    args = build_cli_cmd(
        "analyze",
        URL,
        "--format",
        "json",
        "--models-auto-download",
        "--robots",
        "ignore",
        "--enable",
        "pos",
        "--enable",
        "ner",
        "-o",
        out_json,
    )
    proc = run_cli(args)
    assert proc.returncode == 0, (
        f"CLI failed: {proc.returncode}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
    )

    payload = read_json(out_json)
    analyzers = payload.get("analyzers", [])

    # Should have exactly pos, ner, and dependency (dependency is always included)
    analyzer_names = {a["name"] for a in analyzers}
    expected = {"pos", "ner", "dependency"}
    assert analyzer_names == expected, f"Expected {expected}, got {analyzer_names}"

    # Validate POS analyzer structure
    pos = find_analyzer(payload, "pos")
    pos_results = pos.get("results", {})
    assert "upos_counts" in pos_results
    assert "upos_ratios" in pos_results
    assert "top_lemmas_by_upos" in pos_results
    assert_analyzer_has_model_metadata(payload, "pos")

    # Validate NER analyzer structure
    ner = find_analyzer(payload, "ner")
    ner_results = ner.get("results", {})
    assert "supported" in ner_results
    assert_analyzer_has_model_metadata(payload, "ner")


def test_analyzer_selection_specific_disable() -> None:
    """Test selective analyzer disabling."""
    out_base: Path = unique_output_base("analyzer-selection-disable")
    out_json = str(out_base) + ".json"

    args = build_cli_cmd(
        "analyze",
        URL,
        "--format",
        "json",
        "--models-auto-download",
        "--robots",
        "ignore",
        "--disable",
        "keywords",
        "--disable",
        "readability",
        "-o",
        out_json,
    )
    proc = run_cli(args)
    assert proc.returncode == 0, (
        f"CLI failed: {proc.returncode}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
    )

    payload = read_json(out_json)
    analyzers = payload.get("analyzers", [])
    analyzer_names = {a["name"] for a in analyzers}

    # Should NOT contain disabled analyzers
    assert "keywords" not in analyzer_names
    assert "readability" not in analyzer_names

    # Should still contain other core analyzers
    assert "dependency" in analyzer_names
    assert "lexical_stats" in analyzer_names
    assert "ner" in analyzer_names
    assert "pos" in analyzer_names


def test_analyzer_selection_embeddings() -> None:
    """Test embeddings analyzer functionality."""
    out_base: Path = unique_output_base("analyzer-selection-embeddings")
    out_json = str(out_base) + ".json"

    args = build_cli_cmd(
        "analyze",
        URL,
        "--format",
        "json",
        "--models-auto-download",
        "--robots",
        "ignore",
        "--enable-embeddings",
        "--embeddings-backend",
        "miniLM",
        "-o",
        out_json,
    )
    proc = run_cli(args)
    assert proc.returncode == 0, (
        f"CLI failed: {proc.returncode}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
    )

    payload = read_json(out_json)
    analyzers = payload.get("analyzers", [])
    analyzer_names = {a["name"] for a in analyzers}

    # Should contain embeddings analyzer
    assert "embeddings" in analyzer_names

    # Validate embeddings analyzer structure
    embeddings = find_analyzer(payload, "embeddings")
    embeddings_results = embeddings.get("results", {})

    # Check required fields
    assert embeddings_results.get("supported") is True
    assert "model" in embeddings_results
    assert "dim" in embeddings_results
    assert "vector" in embeddings_results

    # Validate vector properties
    vector = embeddings_results["vector"]
    assert isinstance(vector, list)
    assert len(vector) == embeddings_results["dim"]

    # MiniLM-specific assertions (since we explicitly select MiniLM)
    assert embeddings_results["dim"] == 384
    assert "sentence-transformers" in embeddings_results["model"]
    assert "all-MiniLM-L6-v2" in embeddings_results["model"]


def test_analyzer_selection_sentiment() -> None:
    """Test sentiment analyzer functionality."""
    out_base: Path = unique_output_base("analyzer-selection-sentiment")
    out_json = str(out_base) + ".json"

    args = build_cli_cmd(
        "analyze",
        URL,
        "--format",
        "json",
        "--models-auto-download",
        "--robots",
        "ignore",
        "--enable-sentiment",
        "-o",
        out_json,
    )
    proc = run_cli(args)
    assert proc.returncode == 0, (
        f"CLI failed: {proc.returncode}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
    )

    payload = read_json(out_json)
    analyzers = payload.get("analyzers", [])
    analyzer_names = {a["name"] for a in analyzers}

    # Should contain sentiment analyzer
    assert "sentiment" in analyzer_names

    # Validate sentiment analyzer structure
    sentiment = find_analyzer(payload, "sentiment")
    sentiment_results = sentiment.get("results", {})

    # Check required fields
    assert sentiment_results.get("supported") is True
    assert "label" in sentiment_results
    assert "score" in sentiment_results
    assert "method" in sentiment_results
    assert "scores" in sentiment_results

    # Validate sentiment scores structure
    scores = sentiment_results["scores"]
    assert isinstance(scores, dict)
    assert "neg" in scores
    assert "neu" in scores
    assert "pos" in scores
    assert "compound" in scores

    # Validate score ranges
    for _key, value in scores.items():
        assert isinstance(value, int | float)
        assert 0.0 <= value <= 1.0

    # Validate label
    label = sentiment_results["label"]
    assert label in ["positive", "negative", "neutral"]

    # Validate method
    method = sentiment_results["method"]
    assert method in ["vader", "textblob", "spacy"]


def test_analyzer_selection_combined_flags() -> None:
    """Test combining multiple analyzer selection flags."""
    out_base: Path = unique_output_base("analyzer-selection-combined")
    out_json = str(out_base) + ".json"

    args = build_cli_cmd(
        "analyze",
        URL,
        "--format",
        "json",
        "--models-auto-download",
        "--robots",
        "ignore",
        "--enable",
        "pos",
        "--enable",
        "ner",
        "--disable",
        "keywords",
        "--enable-embeddings",
        "--enable-sentiment",
        "-o",
        out_json,
    )
    proc = run_cli(args)
    assert proc.returncode == 0, (
        f"CLI failed: {proc.returncode}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
    )

    payload = read_json(out_json)
    analyzers = payload.get("analyzers", [])
    analyzer_names = {a["name"] for a in analyzers}

    # Should contain: pos, ner, dependency (always included), embeddings, sentiment
    expected = {"pos", "ner", "dependency", "embeddings", "sentiment"}
    assert expected.issubset(analyzer_names), (
        f"Missing expected analyzers. Expected: {expected}, Found: {analyzer_names}"
    )

    # Should NOT contain disabled analyzers
    assert "keywords" not in analyzer_names

    # Validate all expected analyzers have proper structure
    for name in expected:
        analyzer = find_analyzer(payload, name)
        assert "results" in analyzer
        assert "metadata" in analyzer
        assert_analyzer_has_model_metadata(payload, name)


def test_analyzer_selection_json_structure() -> None:
    """Test that JSON output has all expected top-level fields and structure."""
    out_base: Path = unique_output_base("analyzer-selection-structure")
    out_json = str(out_base) + ".json"

    args = build_cli_cmd(
        "analyze",
        URL,
        "--format",
        "json",
        "--models-auto-download",
        "--robots",
        "ignore",
        "--enable-embeddings",
        "--enable-sentiment",
        "-o",
        out_json,
    )
    proc = run_cli(args)
    assert proc.returncode == 0, (
        f"CLI failed: {proc.returncode}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
    )

    payload = read_json(out_json)

    # Validate top-level structure
    required_top_level = ["tool", "version", "source", "language", "content", "analyzers", "timing"]
    for field in required_top_level:
        assert_json_key(payload, field)

    # Validate source structure
    source = payload["source"]
    assert source["type"] == "url"
    assert URL in source["value"]
    assert "fetched_at" in source
    assert "domain" in source

    # Validate language structure
    language = payload["language"]
    assert "code" in language
    assert "confidence" in language
    assert "model" in language

    # Validate content structure
    content = payload["content"]
    assert "title" in content
    assert "char_count" in content
    assert "word_count" in content

    # Validate analyzers structure
    analyzers = payload["analyzers"]
    assert isinstance(analyzers, list)
    assert len(analyzers) > 0

    # Validate each analyzer has required fields
    for analyzer in analyzers:
        assert "name" in analyzer
        assert "processing_time" in analyzer
        assert "confidence" in analyzer
        assert "results" in analyzer
        assert "metadata" in analyzer

        # Validate metadata structure
        metadata = analyzer["metadata"]
        assert "language" in metadata
        assert "model" in metadata

        # Validate language in metadata
        meta_lang = metadata["language"]
        assert "code" in meta_lang
        assert "confidence" in meta_lang
