from __future__ import annotations

import os
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

pytestmark = [pytest.mark.external]

# Path to test data files
TEST_DATA_DIR = Path(__file__).resolve().parents[1] / "test_data"
SAMPLE_FILE = TEST_DATA_DIR / "sample.txt"
MULTILINGUAL_FILE = TEST_DATA_DIR / "multilingual_test.txt"
URLS_FILE = TEST_DATA_DIR / "urls.txt"


def test_file_analysis_sample() -> None:
    """Test file-based analysis using the sample.txt test data file."""
    out_base: Path = unique_output_base("file-analysis-sample")
    out_json = str(out_base) + ".json"

    args = build_cli_cmd(
        "analyze-file",
        str(SAMPLE_FILE),
        "--format",
        "json",
        "--models-auto-download",
        "--lang",
        "en",
        "-o",
        out_json,
    )
    proc = run_cli(args)
    assert proc.returncode == 0, (
        f"File analysis failed: {proc.returncode}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
    )

    payload = read_json(out_json)

    # Language correctness
    assert_lang(payload, "en")
    conf = float(assert_json_key(payload, "language.confidence"))
    assert conf >= 0.6

    # Content verification - should match our sample file
    content = payload.get("content", {})
    assert "title" in content  # File-based analysis should still have title field
    assert content.get("word_count", 0) > 0
    assert content.get("char_count", 0) > 0

    # Lexical stats should be reasonable for a short file
    lex = find_analyzer(payload, "lexical_stats")
    total_tokens = int(lex.get("results", {}).get("total_tokens", 0))
    assert 3 <= total_tokens <= 6  # Sample file has about 4 tokens

    # Verify analyzers are present and have metadata
    for name in ("pos", "ner", "lexical_stats", "readability", "keywords", "dependency"):
        assert_analyzer_has_model_metadata(payload, name)


def test_multilingual_file_detection() -> None:
    """Test language detection across multiple languages using multilingual_test.txt."""
    out_base: Path = unique_output_base("multilingual-detection")
    out_json = str(out_base) + ".json"

    # Don't specify language - let auto-detection work
    args = build_cli_cmd(
        "analyze-file",
        str(MULTILINGUAL_FILE),
        "--format",
        "json",
        "--models-auto-download",
        "-o",
        out_json,
    )
    proc = run_cli(args)
    assert proc.returncode == 0, (
        f"Multilingual analysis failed: {proc.returncode}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
    )

    payload = read_json(out_json)

    # Auto-detection should pick up the dominant language (English is first)
    # The exact language detected may vary, but it should be one of the supported languages
    detected_code = payload.get("language", {}).get("code")
    supported_codes = {"en", "fr", "es", "de", "ja", "zh", "ru", "ar", "th"}
    assert detected_code in supported_codes, (
        f"Detected language {detected_code} not in supported set"
    )

    conf = float(assert_json_key(payload, "language.confidence"))
    assert conf >= 0.4  # Lower threshold for mixed content

    # Content should reflect the multilingual nature
    content = payload.get("content", {})
    assert content.get("word_count", 0) > 20  # Multiple languages = more words

    # Verify basic analyzer functionality still works
    lex = find_analyzer(payload, "lexical_stats")
    total_tokens = int(lex.get("results", {}).get("total_tokens", 0))
    assert total_tokens > 25  # Multilingual content should have substantial token count


def test_batch_processing_urls() -> None:
    """Test batch processing using the urls.txt test data file."""
    # Create temporary directory for batch outputs
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        args = build_cli_cmd(
            "batch",
            str(URLS_FILE),
            "--output-dir",
            temp_dir,
            "--format",
            "json",
            "--models-auto-download",
            "--robots",
            "ignore",
        )
        proc = run_cli(args)
        assert proc.returncode == 0, (
            f"Batch processing failed: {proc.returncode}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
        )

        # CLI should output paths to generated JSON files
        out_paths = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip().endswith(".json")]
        assert len(out_paths) == 5, f"Expected 5 output files, got {len(out_paths)}"

        # Verify each output file
        expected_langs = ["en", "en", "de", "es", "fr"]  # Based on URLs in urls.txt
        for i, path in enumerate(out_paths):
            assert os.path.exists(path), f"Output file not created: {path}"

            payload = read_json(path)

            # Check language matches expected based on URL
            detected_lang = payload.get("language", {}).get("code")
            expected_lang = expected_langs[i]
            assert detected_lang == expected_lang, (
                f"File {i}: expected {expected_lang}, got {detected_lang}"
            )

            # Verify basic structure
            assert_json_key(payload, "tool")
            assert_json_key(payload, "version")
            assert_json_key(payload, "source")
            assert_json_key(payload, "language")
            assert_json_key(payload, "content")
            assert_json_key(payload, "analyzers")

            # Content should be substantial for Wikipedia articles
            content = payload.get("content", {})
            assert content.get("word_count", 0) > 100
            assert content.get("char_count", 0) > 500

            # Verify analyzers have metadata
            analyzers = payload.get("analyzers", [])
            assert len(analyzers) >= 5  # Should have core analyzers

            for analyzer in analyzers:
                assert "name" in analyzer
                assert "metadata" in analyzer
                assert "model" in analyzer["metadata"]


def test_combined_testdata_workflow() -> None:
    """Comprehensive test combining file analysis, language detection, and batch processing."""
    # Test 1: File analysis with explicit language override
    out_base1: Path = unique_output_base("combined-file-en")
    out_json1 = str(out_base1) + ".json"

    args1 = build_cli_cmd(
        "analyze-file",
        str(SAMPLE_FILE),
        "--format",
        "json",
        "--models-auto-download",
        "--lang",
        "en",  # Explicit override
        "-o",
        out_json1,
    )
    proc1 = run_cli(args1)
    assert proc1.returncode == 0, "Combined test file analysis failed"

    # Test 2: Multilingual file with language detection disabled
    out_base2: Path = unique_output_base("combined-multilingual")
    out_json2 = str(out_base2) + ".json"

    args2 = build_cli_cmd(
        "analyze-file",
        str(MULTILINGUAL_FILE),
        "--format",
        "json",
        "--models-auto-download",
        "--lang",
        "en",  # Force English processing despite mixed content
        "-o",
        out_json2,
    )
    proc2 = run_cli(args2)
    assert proc2.returncode == 0, "Combined test multilingual analysis failed"

    # Test 3: Batch processing with selective analyzers
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        args3 = build_cli_cmd(
            "batch",
            str(URLS_FILE),
            "--output-dir",
            temp_dir,
            "--format",
            "json",
            "--models-auto-download",
            "--robots",
            "ignore",
            "--enable",
            "pos",
            "--enable",
            "lexical_stats",
            "--disable",
            "keywords",
            "--disable",
            "readability",
        )
        proc3 = run_cli(args3)
        assert proc3.returncode == 0, "Combined test batch processing failed"

        # Verify selective analyzer behavior
        out_paths = [ln.strip() for ln in proc3.stdout.splitlines() if ln.strip().endswith(".json")]
        assert len(out_paths) == 5

        for path in out_paths:
            payload = read_json(path)
            analyzer_names = {a["name"] for a in payload.get("analyzers", [])}

            # Should have enabled analyzers
            assert "pos" in analyzer_names
            assert "lexical_stats" in analyzer_names
            assert "dependency" in analyzer_names  # Always included

            # Should NOT have disabled analyzers
            assert "keywords" not in analyzer_names
            assert "readability" not in analyzer_names
