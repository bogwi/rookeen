#!/usr/bin/env python3
"""
Unified CI/CD-friendly validation script for Rookeen JSON outputs.
Provides comprehensive validation with detailed error reporting and appropriate exit codes.
Can process URLs directly or validate existing files.
Uses schemas/rookeen_v1.json for all validation.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from jsonschema import SchemaError, ValidationError, validate


def load_schema(schema_path: str = "schemas/rookeen_v1.json") -> Any:
    """Load JSON schema from file."""
    try:
        with open(schema_path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Schema file not found: {schema_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in schema file: {e}")
        sys.exit(1)


def process_url_to_json(url: str, analyzer_flags: list[str] | None = None) -> str | None:
    """Process a URL with Rookeen and return path to generated JSON file."""
    # Create temporary file for output
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        # Build CLI command
        cmd = [
            sys.executable,
            "-m",
            "rookeen.cli",
            "analyze",
            url,
            "--format",
            "json",
            "--models-auto-download",
            "--robots",
            "ignore",
            "-o",
            temp_path,
        ]

        # Add analyzer flags if provided
        if analyzer_flags:
            cmd.extend(analyzer_flags)

        # Run the command
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )

        if result.returncode != 0:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return None

        return temp_path

    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return None


def cleanup_temp_file(file_path: str) -> None:
    """Clean up temporary file if it exists."""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception:
        pass  # Ignore cleanup errors in CI


def validate_single_file(file_path: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Validate a single JSON file against the schema."""
    result: dict[str, Any] = {"file": file_path, "valid": False, "errors": [], "analyzer_count": 0, "analyzers": []}

    try:
        with open(file_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        result["errors"].append(f"File not found: {file_path}")
        return result
    except json.JSONDecodeError as e:
        result["errors"].append(f"Invalid JSON: {e}")
        return result

    try:
        validate(data, schema)
        result["valid"] = True

        # Extract metadata for reporting
        if "analyzers" in data:
            result["analyzer_count"] = len(data["analyzers"])
            result["analyzers"] = [
                analyzer.get("name", "unknown") for analyzer in data["analyzers"]
            ]

    except ValidationError as e:
        result["errors"].append(f"Schema validation failed: {e.message}")
        if e.absolute_path:
            result["errors"].append(f"Failed at: {' -> '.join(str(x) for x in e.absolute_path)}")
        if e.instance is not None:
            result["errors"].append(f"Invalid value: {e.instance}")
    except SchemaError as e:
        result["errors"].append(f"Schema error: {e.message}")
    except Exception as e:
        result["errors"].append(f"Unexpected error: {e}")

    return result


def print_validation_report(
    results: list[dict[str, Any]], verbose: bool = False, quiet: bool = False
) -> bool:
    """Print a comprehensive validation report."""
    total_files = len(results)
    valid_files = sum(1 for r in results if r["valid"])
    invalid_files = total_files - valid_files

    if not quiet:
        print("Rookeen JSON Schema Validation Report")
        print("=" * 50)
        print(f"Summary: {valid_files}/{total_files} files valid")
        print()

        if invalid_files > 0:
            print("Validation Errors:")
            print("-" * 30)
            for result in results:
                if not result["valid"]:
                    display_name = result.get("original_url", result["file"])
                    print(f"{display_name}:")
                    for error in result["errors"]:
                        print(f"   • {error}")
                    print()

        if verbose and valid_files > 0:
            print("Valid Files Details:")
            print("-" * 30)
            for result in results:
                if result["valid"]:
                    analyzers = ", ".join(result["analyzers"]) if result["analyzers"] else "none"
                    display_name = result.get("original_url", Path(result["file"]).name)
                    print(f"{display_name}: {result['analyzer_count']} analyzers ({analyzers})")
            print()

    if valid_files == total_files:
        if not quiet:
            print("All files passed validation!")
        return True
    if not quiet:
        print(f"{invalid_files} file(s) failed validation.")
    return False


def main() -> None:
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate Rookeen JSON outputs against schema")
    parser.add_argument("inputs", nargs="+", help="URLs to process or JSON files to validate")
    parser.add_argument(
        "--schema", "-s", default="schemas/rookeen_v1.json", help="Path to JSON schema file"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed information about valid files"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Only show summary, suppress detailed errors"
    )
    parser.add_argument(
        "--enable",
        action="append",
        dest="analyzer_flags",
        help="Enable specific analyzers (can be used multiple times)",
    )
    parser.add_argument(
        "--disable",
        action="append",
        dest="disable_flags",
        help="Disable specific analyzers (can be used multiple times)",
    )
    parser.add_argument(
        "--enable-embeddings", action="store_true", help="Enable embeddings analyzer"
    )
    parser.add_argument("--enable-sentiment", action="store_true", help="Enable sentiment analyzer")

    args = parser.parse_args()

    # Build analyzer flags
    analyzer_flags = []
    if args.analyzer_flags:
        for analyzer in args.analyzer_flags:
            analyzer_flags.extend(["--enable", analyzer])
    if args.disable_flags:
        for analyzer in args.disable_flags:
            analyzer_flags.extend(["--disable", analyzer])
    if args.enable_embeddings:
        analyzer_flags.append("--enable-embeddings")
    if args.enable_sentiment:
        analyzer_flags.append("--enable-sentiment")

    # Load schema
    schema = load_schema(args.schema)

    # Process inputs (URLs or files)
    results = []
    temp_files = []

    for input_item in args.inputs:
        if input_item.startswith(("http://", "https://")):
            # Process URL
            if not args.quiet:
                print(f"Processing URL: {input_item}")
            temp_file = process_url_to_json(input_item, analyzer_flags)
            if temp_file:
                temp_files.append(temp_file)
                result = validate_single_file(temp_file, schema)
                result["is_url"] = True
                result["original_url"] = input_item
                results.append(result)
            else:
                results.append(
                    {
                        "file": input_item,
                        "valid": False,
                        "errors": ["Failed to process URL"],
                        "analyzer_count": 0,
                        "analyzers": [],
                        "is_url": True,
                        "original_url": input_item,
                    }
                )
        else:
            # Validate existing file
            result = validate_single_file(input_item, schema)
            result["is_url"] = False
            results.append(result)

    # Print report
    success = print_validation_report(results, args.verbose, args.quiet)

    # Clean up temporary files
    for temp_file in temp_files:
        cleanup_temp_file(temp_file)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
