#!/usr/bin/env python3
"""
Comprehensive test suite for JSON schema validation.
Tests both valid and invalid cases to ensure robust validation.
"""

import json
import subprocess
import sys
from pathlib import Path


def run_validation_script(script_path: str, files: list[str], schema: str | None = None) -> tuple[bool, str, str]:
    """Run validation script and return (success, stdout, stderr)."""
    cmd = [sys.executable, script_path, *files]
    if schema:
        cmd.extend(["--schema", schema])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def test_basic_validation() -> None:
    """Test basic validation functionality."""
    print("Testing basic validation functionality...")

    # Test URL processing
    test_url = (
        "https://en.wikipedia.org/w/index.php?title=Natural_language_processing&oldid=1201524046"
    )

    # Test simple validator with URL
    success, stdout, stderr = run_validation_script("scripts/validate_for_ci.py", [test_url])
    assert success, (
        "Basic validation failed\n"
        f"stdout: {stdout}\n"
        f"stderr: {stderr}"
    )


def test_detailed_validation() -> None:
    """Test detailed validation with error reporting."""
    print("Testing detailed validation...")

    # Test valid URLs
    valid_urls = [
        "https://en.wikipedia.org/w/index.php?title=Natural_language_processing&oldid=1201524046"
    ]

    success, stdout, stderr = run_validation_script("scripts/validate_for_ci.py", valid_urls)
    assert success and "All files passed validation" in stdout, (
        "Detailed validation failed for valid URLs\n"
        f"stdout: {stdout}\n"
        f"stderr: {stderr}"
    )

    # Test invalid files
    invalid_files = [
        "test_data/invalid_missing_required.json",
        "test_data/invalid_wrong_types.json",
        "test_data/invalid_analyzer_missing_required.json",
    ]

    success, stdout, stderr = run_validation_script(
        "scripts/validate_for_ci.py", invalid_files
    )
    assert (not success) and "failed validation" in stdout, (
        "Detailed validation should reject invalid files\n"
        f"stdout: {stdout}\n"
        f"stderr: {stderr}"
    )


def test_schema_validation() -> None:
    """Test schema file validation."""
    print("Testing schema file validation...")

    schema_path = "schemas/rookeen_v1.json"

    try:
        with open(schema_path) as f:
            schema = json.load(f)

        # Basic schema structure checks
        required_fields = ["tool", "version", "language", "analyzers"]
        assert all(field in schema.get("required", []) for field in required_fields), (
            "Schema missing required top-level fields"
        )

        # Check analyzer schemas
        analyzers = schema["properties"]["analyzers"]["items"]["properties"]["results"]["oneOf"]
        analyzer_names = [
            "lexical_stats",
            "pos",
            "ner",
            "readability",
            "keywords",
            "sentiment",
            "dependency",
            "embeddings",
        ]

        schema_names = []
        for analyzer_schema in analyzers:
            if "required" in analyzer_schema:
                # Check if it's one of our known analyzers by required fields
                required = analyzer_schema["required"]
                if "total_tokens" in required:
                    schema_names.append("lexical_stats")
                elif "upos_counts" in required:
                    schema_names.append("pos")
                elif "counts_by_label" in required:
                    schema_names.append("ner")
                elif "flesch_reading_ease" in required:
                    schema_names.append("readability")
                elif "keywords" in required:
                    schema_names.append("keywords")
                elif "label" in required:
                    schema_names.append("sentiment")
                elif "dep_counts" in required:
                    schema_names.append("dependency")
                elif "vector" in required:
                    schema_names.append("embeddings")

        assert len(schema_names) >= 8, (
            f"Schema missing analyzers: {set(analyzer_names) - set(schema_names)}"
        )

    except Exception as e:
        raise AssertionError(f"Schema validation failed: {e}") from e


def test_real_data_validation() -> None:
    """Test validation against real Rookeen output from URLs."""
    print("Testing validation against real data from URLs...")

    # Test URLs from the test suite
    test_urls = [
        "https://en.wikipedia.org/w/index.php?title=Natural_language_processing&oldid=1201524046",
        "https://en.wikipedia.org/wiki/Cat",
    ]

    success, stdout, stderr = run_validation_script(
        "scripts/validate_for_ci.py", [*test_urls, "--verbose"]
    )
    assert success, (
        "Real data validation failed\n"
        f"stdout: {stdout}\n"
        f"stderr: {stderr}"
    )


def main() -> int:
    """Run all validation tests."""
    print("Starting JSON Schema Validation Test Suite")
    print("=" * 60)

    tests = [
        ("Basic Validation", test_basic_validation),
        ("Detailed Validation", test_detailed_validation),
        ("Schema Structure", test_schema_validation),
        ("Real Data Validation", test_real_data_validation),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        try:
            # Consider a test PASSED if it runs without raising an exception
            test_func()
            passed += 1
            print(f"{test_name}: PASSED")
        except AssertionError as e:
            print(f"{test_name}: FAILED - {e}")
        except Exception as e:
            print(f"{test_name}: ERROR - {e}")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("All tests passed! JSON validation is working correctly.")
        return 0
    print(f"{total - passed} test(s) failed. Please review the validation setup.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
