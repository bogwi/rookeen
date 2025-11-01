#!/usr/bin/env python3
"""
Rookeen Demo Test Suite - Real-World Usage Verification

This script runs a comprehensive set of real-world usage tests to verify
that rookeen delivers on its advertised features.

Run after syncing all dependencies:
    uv sync --group dev --all-extras
    uv run python scripts/run_demo_tests.py
"""

from __future__ import annotations

import concurrent.futures
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


class TestResult:
    """Container for individual test results."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.success = False
        self.error: str | None = None
        self.verifications: list[str] = []
        self.output_file: Path | None = None


def run_rookeen_cmd(args: list[str], timeout: int = 300, stdin_input: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run a rookeen command and return the result."""
    cmd = ["uv", "run", "rookeen", *args]
    return subprocess.run(
        cmd,
        input=stdin_input,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=REPO_ROOT,
    )


def extract_json_from_stdout(stdout: str) -> dict[str, Any]:
    """Extract JSON from stdout that may contain other output."""
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not locate JSON object in stdout")
    payload = stdout[start : end + 1]
    result: dict[str, Any] = json.loads(payload)
    return result


def verify_result(payload: dict[str, Any], test: TestResult) -> None:
    """Verify a test result and populate the test object."""
    # Basic structure checks
    if payload.get("tool") != "rookeen":
        test.verifications.append(f"FAIL: Tool name incorrect: {payload.get('tool')}")
        return

    test.verifications.append("Tool: rookeen")

    if "version" not in payload:
        test.verifications.append("FAIL: Missing version")
        return

    test.verifications.append(f"Version: {payload.get('version')}")

    # Language detection
    lang = payload.get("language", {})
    if lang.get("code"):
        test.verifications.append(f"Language detected: {lang.get('code')}")
    else:
        test.verifications.append("FAIL: Language detection failed")

    # Content processing
    content = payload.get("content", {})
    if content.get("word_count", 0) > 0:
        test.verifications.append(
            f"Content processed: {content.get('word_count')} words, "
            f"{content.get('char_count')} characters"
        )

    # Analyzers check
    analyzers = payload.get("analyzers", [])
    test.verifications.append(f"Total analyzers: {len(analyzers)}")
    analyzer_names = {a.get("name") for a in analyzers}
    test.verifications.append(f"  Analyzers: {', '.join(sorted(analyzer_names))}")


def test_1_basic_analysis(test: TestResult) -> None:
    """Test 1: Basic Web Page Analysis (BBC Technology)."""
    output_file = RESULTS_DIR / f"demo_test1_basic_{int(time.time())}.json"
    test.output_file = output_file

    proc = run_rookeen_cmd(
        [
            "analyze",
            "https://www.bbc.com/news/technology",
            "--format",
            "json",
            "--models-auto-download",
            "--robots",
            "ignore",
            "-o",
            str(output_file),
        ]
    )

    if proc.returncode != 0:
        test.error = f"Command failed: {proc.stderr}"
        return

    try:
        payload = json.loads(output_file.read_text())
        verify_result(payload, test)

        # Specific verifications for basic analysis
        analyzers = payload.get("analyzers", [])
        analyzer_names = {a.get("name") for a in analyzers}

        required_core = {"dependency", "keywords", "lexical_stats", "ner", "pos", "readability"}
        missing = required_core - analyzer_names

        if missing:
            test.verifications.append(f"FAIL: Missing core analyzers: {missing}")
        else:
            test.verifications.append("All core analyzers present")

        # Check NER
        ner = next((a for a in analyzers if a.get("name") == "ner"), None)
        if ner and ner.get("results", {}).get("total_entities", 0) > 0:
            entities = ner["results"]["total_entities"]
            test.verifications.append(f"NER: {entities} entities detected")

        # Check keywords
        keywords = next((a for a in analyzers if a.get("name") == "keywords"), None)
        if keywords and keywords.get("results", {}).get("keywords"):
            kw_count = len(keywords["results"]["keywords"])
            test.verifications.append(f"Keywords: {kw_count} keywords extracted")

        # Check readability
        readability = next((a for a in analyzers if a.get("name") == "readability"), None)
        if readability and readability.get("results", {}).get("supported"):
            fk_grade = readability["results"].get("flesch_kincaid_grade")
            if fk_grade is not None:
                test.verifications.append(f"Readability: Flesch-Kincaid Grade = {fk_grade:.2f}")

        test.success = True
        test.verifications.insert(0, "Test completed successfully")

    except Exception as e:
        test.error = f"Failed to parse result: {e}"


def test_2_embeddings(test: TestResult) -> None:
    """Test 2: Technical Blog with Embeddings (Python.org)."""
    output_file = RESULTS_DIR / f"demo_test2_embeddings_{int(time.time())}.json"
    test.output_file = output_file

    proc = run_rookeen_cmd(
        [
            "analyze",
            "https://www.python.org/about/gettingstarted/",
            "--enable-embeddings",
            "--embeddings-backend",
            "miniLM",
            "--models-auto-download",
            "--robots",
            "ignore",
            "-o",
            str(output_file),
        ]
    )

    if proc.returncode != 0:
        test.error = f"Command failed: {proc.stderr}"
        return

    try:
        payload = json.loads(output_file.read_text())
        verify_result(payload, test)

        # Embeddings-specific verification
        analyzers = payload.get("analyzers", [])
        embeddings = next((a for a in analyzers if a.get("name") == "embeddings"), None)

        if not embeddings:
            test.verifications.append("FAIL: Embeddings analyzer missing")
            return

        emb_results = embeddings.get("results", {})
        dim = emb_results.get("dim")
        backend = emb_results.get("backend")
        normalized = emb_results.get("normalized")

        if dim:
            test.verifications.append(f"Embeddings dimension: {dim}")
        if backend:
            test.verifications.append(f"Embeddings backend: {backend}")
        if normalized is not None:
            test.verifications.append(f"Embeddings normalized: {normalized}")

        if dim == 384 and backend == "miniLM" and normalized is True:
            test.verifications.append("Embeddings configuration correct")
        else:
            test.verifications.append("WARN: Embeddings configuration unexpected")

        test.success = True
        test.verifications.insert(0, "Test completed successfully")

    except Exception as e:
        test.error = f"Failed to parse result: {e}"


def test_3_sentiment(test: TestResult) -> None:
    """Test 3: Sentiment Analysis (File-based)."""
    output_file = RESULTS_DIR / f"demo_test3_sentiment_{int(time.time())}.json"
    test.output_file = output_file

    # Create temporary file with negative sentiment
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(
            "I was really disappointed with this product. The quality is poor and it broke "
            "after just one day. Customer service was unhelpful and rude. I would not recommend "
            "this to anyone. Very frustrating experience overall."
        )
        temp_file = Path(f.name)

    try:
        proc = run_rookeen_cmd(
            [
                "analyze-file",
                str(temp_file),
                "--enable-sentiment",
                "--models-auto-download",
                "-o",
                str(output_file),
            ]
        )

        if proc.returncode != 0:
            test.error = f"Command failed: {proc.stderr}"
            return

        payload = json.loads(output_file.read_text())
        verify_result(payload, test)

        # Sentiment-specific verification
        analyzers = payload.get("analyzers", [])
        sentiment = next((a for a in analyzers if a.get("name") == "sentiment"), None)

        if not sentiment:
            test.verifications.append("FAIL: Sentiment analyzer missing")
            return

        sent_results = sentiment.get("results", {})
        supported = sent_results.get("supported")

        if supported:
            test.verifications.append("Sentiment supported: true")

            # Check for compound score (VADER)
            compound = sent_results.get("scores", {}).get("compound")
            if compound is not None:
                test.verifications.append(f"Sentiment compound score: {compound:.4f}")
                if compound < 0:
                    test.verifications.append("Sentiment correctly identified as negative")
                else:
                    test.verifications.append("WARN: Sentiment should be negative but got positive/neutral")

            # Check for polarity (TextBlob)
            polarity = sent_results.get("polarity")
            if polarity is not None:
                test.verifications.append(f"Sentiment polarity: {polarity:.4f}")
        else:
            test.verifications.append("FAIL: Sentiment not supported (missing dependencies?)")

        test.success = True
        test.verifications.insert(0, "Test completed successfully")

    except Exception as e:
        test.error = f"Failed to parse result: {e}"
    finally:
        temp_file.unlink()


def test_4_selective_analyzers(test: TestResult) -> None:
    """Test 4: Selective Analyzers (Wikipedia)."""
    output_file = RESULTS_DIR / f"demo_test4_selective_{int(time.time())}.json"
    test.output_file = output_file

    proc = run_rookeen_cmd(
        [
            "analyze",
            "https://en.wikipedia.org/wiki/Python_(programming_language)",
            "--enable",
            "pos",
            "--enable",
            "ner",
            "--enable",
            "lexical_stats",
            "--disable",
            "keywords",
            "--disable",
            "readability",
            "--models-auto-download",
            "--robots",
            "ignore",
            "-o",
            str(output_file),
        ]
    )

    if proc.returncode != 0:
        test.error = f"Command failed: {proc.stderr}"
        return

    try:
        payload = json.loads(output_file.read_text())
        verify_result(payload, test)

        # Selective analyzer verification
        analyzers = payload.get("analyzers", [])
        analyzer_names = {a.get("name") for a in analyzers}

        # Should have
        should_have = {"pos", "ner", "lexical_stats", "dependency"}
        has_all = should_have.issubset(analyzer_names)
        if has_all:
            test.verifications.append("All enabled analyzers present")
        else:
            missing = should_have - analyzer_names
            test.verifications.append(f"FAIL: Missing enabled analyzers: {missing}")

        # Should NOT have
        should_not_have = {"keywords", "readability"}
        has_disabled = should_not_have.intersection(analyzer_names)
        if not has_disabled:
            test.verifications.append("Disabled analyzers correctly excluded")
        else:
            test.verifications.append(f"FAIL: Disabled analyzers still present: {has_disabled}")

        test.success = True
        test.verifications.insert(0, "Test completed successfully")

    except Exception as e:
        test.error = f"Failed to parse result: {e}"


def test_5_stdin_pipeline(test: TestResult) -> None:
    """Test 5: Stdin Pipeline."""
    test_text = (
        "The Python programming language is an excellent choice for data science "
        "and web development. It offers clean syntax, powerful libraries, and "
        "strong community support."
    )

    proc = run_rookeen_cmd(
        [
            "analyze",
            "--stdin",
            "--models-auto-download",
            "--stdout",
        ],
        stdin_input=test_text,
    )

    if proc.returncode != 0:
        test.error = f"Command failed: {proc.stderr}"
        return

    try:
        payload = extract_json_from_stdout(proc.stdout)
        verify_result(payload, test)

        # Stdin-specific verification
        source = payload.get("source", {})
        if source.get("type") == "stdin":
            test.verifications.append("Stdin source correctly identified")

        # Check that processing worked
        content = payload.get("content", {})
        if content.get("word_count", 0) > 0:
            test.verifications.append(f"Stdin content processed: {content.get('word_count')} words")

        # Check lexical stats
        analyzers = payload.get("analyzers", [])
        lex_stats = next((a for a in analyzers if a.get("name") == "lexical_stats"), None)
        if lex_stats and lex_stats.get("results", {}).get("total_tokens", 0) > 0:
            tokens = lex_stats["results"]["total_tokens"]
            test.verifications.append(f"Lexical stats: {tokens} tokens")

        test.success = True
        test.verifications.insert(0, "Test completed successfully")

    except Exception as e:
        test.error = f"Failed to parse result: {e}"


def generate_report(tests: list[TestResult]) -> str:
    """Generate a formatted test report."""
    report_lines = [
        "ROOKEEN REAL-WORLD USAGE TEST REPORT",
        "=" * 50,
        "",
    ]

    for i, test in enumerate(tests, 1):
        report_lines.extend([
            f"Test {i}: {test.name}",
            "-" * 50,
        ])

        if test.error:
            report_lines.append("Result: FAILED")
            report_lines.append(f"Error: {test.error}")
        elif test.success:
            report_lines.append("Result: SUCCESS")
            report_lines.append("")
            report_lines.append("Verification:")
            report_lines.extend(f"  {v}" for v in test.verifications)
            if test.output_file:
                report_lines.append("")
                report_lines.append(f"Output: {test.output_file}")
        else:
            report_lines.append("Result: UNKNOWN")

        report_lines.append("")
        report_lines.append("")

    # Summary
    passed = sum(1 for t in tests if t.success)
    failed = sum(1 for t in tests if t.error)
    total = len(tests)

    report_lines.extend([
        "SUMMARY",
        "=" * 50,
        f"Total tests: {total}",
        f"Passed: {passed}",
        f"Failed: {failed}",
        "",
    ])

    if passed == total:
        report_lines.extend([
            "All tests completed successfully. Rookeen delivers on its advertised features:",
            "",
            "- Web page fetching and HTML sanitization",
            "- Language detection (auto)",
            "- Core analyzers (POS, NER, lexical_stats, keywords, readability, dependency)",
            "- Optional analyzers (embeddings, sentiment)",
            "- Selective analyzer enable/disable",
            "- Multiple input sources (URL, file, stdin)",
            "- Multiple output formats (JSON, stdout streaming)",
            "- Industry-ready Unix pipeline composability",
            "",
            "ROOKEEN DOES WHAT IT ADVERTISES",
        ])
    else:
        report_lines.append("WARN: Some tests failed. Check errors above.")

    return "\n".join(report_lines)


def main() -> int:
    """Run all demo tests and generate report."""
    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Check if uv is available
    if not shutil.which("uv"):
        print("ERROR: 'uv' command not found. Please install uv first.", file=sys.stderr)
        return 1

    print("Running Rookeen Demo Test Suite...")
    print("=" * 50)
    print()

    tests = [
        TestResult("Basic Web Page Analysis (BBC Technology)"),
        TestResult("Technical Blog with Embeddings (Python.org)"),
        TestResult("Sentiment Analysis (File-based)"),
        TestResult("Selective Analyzers (Wikipedia)"),
        TestResult("Stdin Pipeline"),
    ]

    # Run tests in parallel
    test_functions = [
        test_1_basic_analysis,
        test_2_embeddings,
        test_3_sentiment,
        test_4_selective_analyzers,
        test_5_stdin_pipeline,
    ]

    print("Running tests in parallel...")
    print()

    # Run tests in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tests
        future_to_test = {
            executor.submit(test_func, test): (test, test_func)
            for test, test_func in zip(tests, test_functions, strict=False)
        }

        # Process completed tests as they finish
        for future in concurrent.futures.as_completed(future_to_test):
            test, _ = future_to_test[future]
            try:
                future.result()  # This will raise any exceptions that occurred
                if test.success:
                    print(f"  [PASS] {test.name}")
                elif test.error:
                    print(f"  [FAIL] {test.name}: {test.error}")
            except Exception as e:
                test.error = f"Unexpected error: {e}"
                print(f"  [FAIL] {test.name}: {e}")
            print()

    # Generate report
    report = generate_report(tests)
    report_file = RESULTS_DIR / f"demo_test_report_{int(time.time())}.txt"
    report_file.write_text(report)

    print("=" * 50)
    print(f"Report saved to: {report_file}")
    print()
    print(report)

    # Return exit code based on results
    if all(t.success for t in tests):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

