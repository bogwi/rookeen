#!/usr/bin/env python3
"""
Performance benchmark harness for Rookeen.

Runs analyses on fixed inputs, capturing timing and return codes.
Results are stored as CSV for trend analysis.

INDUSTRY-STANDARD UNIX PIPELINE COMPOSABILITY:
- --quiet: Suppress all non-essential output (for pipelines)
- --json: Output only JSON results to stdout
- --format: Choose output format (json, csv, table)
- Proper exit codes for automation
- Structured output for downstream processing
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from subprocess import DEVNULL, PIPE, run
from typing import Any

"""Benchmark cases configuration.

Each case is a dict with keys:
  - name: case identifier
  - file: local HTML/text file to analyze (offline)
  - lang: language code
  - analyzers: list of analyzers to enable via --enable flags
  - extra_args: optional list of additional CLI args (e.g., embeddings backend)
"""

CASES: list[dict[str, Any]] = [
    {
        "name": "en_wiki",
        "file": "bench/nlp_wiki_oldid_1201524046.html",
        "lang": "en",
        "analyzers": [],
        "extra_args": [],
    },
    {
        "name": "en_wiki_embeddings",
        "file": "bench/nlp_wiki_oldid_1201524046.html",
        "lang": "en",
        "analyzers": ["embeddings"],
        "extra_args": [],
    },
    {
        "name": "en_wiki_sentiment",
        "file": "bench/nlp_wiki_oldid_1201524046.html",
        "lang": "en",
        "analyzers": ["sentiment"],
        "extra_args": [],
    },
    {
        "name": "en_wiki_full",
        "file": "bench/nlp_wiki_oldid_1201524046.html",
        "lang": "en",
        "analyzers": ["embeddings", "sentiment"],
        "extra_args": [],
    },
]

# Add embeddings backend-specific benchmarks
CASES.append(
    {
        "name": "en_wiki_bge_m3",
        "file": "bench/nlp_wiki_oldid_1201524046.html",
        "lang": "en",
        "analyzers": ["embeddings"],
        "extra_args": ["--embeddings-backend", "bge-m3", "--embeddings-model", "BAAI/bge-m3"],
    }
)

# Only include OpenAI case if API key is available
if os.getenv("OPENAI_API_KEY") or os.getenv("ROOKEEN_OPENAI_API_KEY"):
    CASES.append(
        {
            "name": "en_wiki_openai_te3",
            "url": "https://en.wikipedia.org/w/index.php?title=Natural_language_processing&oldid=1201524046",
            "lang": "en",
            "analyzers": ["embeddings"],
            "extra_args": ["--embeddings-backend", "openai-te3"],
        }
    )


def run_benchmark(
    name: str,
    file: str,
    lang: str,
    analyzers: list[str],
    quiet: bool = False,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Run a single benchmark case and return results.

    Args:
        name: Benchmark case name
        url: URL to analyze
        lang: Language code
        analyzers: List of analyzers to enable
        quiet: If True, suppress progress output
    """
    if not quiet:
        print(f"Running benchmark: {name}", file=sys.stderr)

    # Build command
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "rookeen.cli",
        "analyze-file",
        file,
        "--lang",
        lang,
        "-o",
        f"results/_bench_{name}.json",
    ]

    # Add analyzer flags
    # Optional analyzers must be enabled via dedicated flags so they are registered.
    # Map friendly names to the correct CLI switches.
    for analyzer in analyzers:
        if analyzer == "embeddings":
            cmd.append("--enable-embeddings")
        elif analyzer == "sentiment":
            cmd.append("--enable-sentiment")
        else:
            cmd.extend(["--enable", analyzer])

    # If this is an embeddings-only case, disable other analyzers to isolate cost
    # (keep dependency which is auto-included when parser is present)
    if analyzers == ["embeddings"]:
        cmd.extend([
            "--disable", "keywords",
            "--disable", "lexical_stats",
            "--disable", "ner",
            "--disable", "pos",
            "--disable", "readability",
        ])

    # Add any extra CLI args (e.g., embeddings backend)
    if extra_args:
        cmd.extend(extra_args)

    # Run benchmark (capture output based on quiet mode)
    t0 = time.time()
    stdout_dest = DEVNULL if quiet else PIPE
    stderr_dest = DEVNULL if quiet else PIPE
    p = run(cmd, stdout=stdout_dest, stderr=stderr_dest, text=True)
    dt = time.time() - t0

    result = {
        "timestamp": datetime.now().isoformat(),
        "case": name,
        "source_file": file,
        "language": lang,
        "analyzers": ",".join(analyzers) if analyzers else "default",
        "return_code": p.returncode,
        "seconds": round(dt, 3),
        "success": p.returncode == 0,
    }

    # Include error details only if there was an error
    if p.returncode != 0:
        if not quiet:
            print(f"  FAILED (exit code {p.returncode})", file=sys.stderr)
        if p.stdout:
            result["error_stdout"] = p.stdout[:500]  # Truncate long output
        if p.stderr:
            result["error_stderr"] = p.stderr[:500]
    elif not quiet:
        print(".3f", file=sys.stderr)

    return result


def output_table(results: list[dict[str, Any]], file: Any = sys.stdout) -> None:
    """Output results in human-readable table format."""
    if not results:
        print("No results to display", file=file)
        return

    # Table headers
    headers = ["Case", "Language", "Analyzers", "Return Code", "Time (s)", "Status"]
    col_widths = [
        max(len(h), max(len(str(r.get(k.lower().replace(" ", "_"), ""))) for r in results))
        for h, k in zip(
            headers, ["case", "language", "analyzers", "return_code", "seconds", "success"], strict=False
        )
    ]

    # Header row
    header_line = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths, strict=False))
    separator = "-+-".join("-" * w for w in col_widths)
    print(header_line, file=file)
    print(separator, file=file)

    # Data rows
    for result in results:
        status = "✓" if result.get("success", False) else "✗"
        row = [
            result.get("case", ""),
            result.get("language", ""),
            result.get("analyzers", ""),
            str(result.get("return_code", "")),
            ".3f",
            status,
        ]
        print(" | ".join(f"{cell:<{w}}" for cell, w in zip(row, col_widths, strict=False)), file=file)


def output_csv(results: list[dict[str, Any]], file: Any = sys.stdout) -> None:
    """Output results in CSV format."""
    if not results:
        return

    fieldnames = [
        "timestamp",
        "case",
        "url",
        "language",
        "analyzers",
        "return_code",
        "seconds",
        "success",
    ]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        # Remove error fields for CSV (they're optional)
        csv_row = {k: v for k, v in result.items() if k in fieldnames}
        writer.writerow(csv_row)


def save_files(results: list[dict[str, Any]], quiet: bool) -> tuple[str, str]:
    """Save results to files and return paths. Returns (csv_path, json_path)."""
    os.makedirs("bench/results", exist_ok=True)

    # Save as JSON (for compatibility with existing validation)
    json_path = "bench/results/latest.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save as CSV for trend analysis
    csv_path = f"bench/results/{int(time.time())}.csv"
    with open(csv_path, "w", newline="") as f:
        output_csv(results, f)

    if not quiet:
        print(f"Results saved to: {csv_path}", file=sys.stderr)
        print(f"JSON results: {json_path}", file=sys.stderr)

    return csv_path, json_path


def main() -> None:
    """Run all benchmark cases with industry-standard Unix pipeline composability."""
    parser = argparse.ArgumentParser(
        description="Rookeen Performance Benchmark Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Interactive mode (default)
  %(prog)s

  # Quiet mode for scripts
  %(prog)s --quiet

  # Pipeline-friendly JSON output
  %(prog)s --json | jq '.[0].return_code == 0'

  # CSV output for data analysis
  %(prog)s --format csv > results.csv

  # Table format for human reading
  %(prog)s --format table

  # Check if all benchmarks passed
  %(prog)s --json --quiet | jq 'all(.success)'
        """,
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress all non-essential output (progress, file saves)",
    )

    parser.add_argument(
        "--json", action="store_true", help="Output only JSON results to stdout (implies --quiet)"
    )

    parser.add_argument(
        "--format",
        choices=["json", "csv", "table"],
        default="json",
        help="Output format (default: json)",
    )

    parser.add_argument("--no-save", action="store_true", help="Don't save results to files")

    args = parser.parse_args()

    # --json implies --quiet and overrides format
    if args.json:
        args.quiet = True
        args.format = "json"

    # Run benchmarks
    results = []
    for case in CASES:
        name = case["name"]
        file = case["file"]
        lang = case["lang"]
        analyzers = case.get("analyzers", [])
        extra_args = case.get("extra_args", [])
        try:
            result = run_benchmark(name, file, lang, analyzers, quiet=args.quiet, extra_args=extra_args)
            results.append(result)
        except Exception as e:
            if not args.quiet:
                print(f"ERROR in {name}: {e}", file=sys.stderr)
            results.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "case": name,
                    "source_file": file,
                    "language": lang,
                    "analyzers": ",".join(analyzers) if analyzers else "default",
                    "return_code": -1,
                    "seconds": 0,
                    "success": False,
                    "error": str(e),
                }
            )

    # Save files unless disabled
    if not args.no_save:
        save_files(results, args.quiet)

    # Print summary unless quiet
    if not args.quiet:
        successful = sum(1 for r in results if r.get("success", False))
        # total_time = sum(r.get("seconds", 0) for r in results)
        # avg_time = total_time / len(results) if results else 0

        print(f"\nSummary: {successful}/{len(results)} successful", file=sys.stderr)
        print(".3f", file=sys.stderr)

    # Output results in requested format
    if args.format == "json":
        print(json.dumps(results, indent=2 if not args.quiet else None))
    elif args.format == "csv":
        output_csv(results)
    elif args.format == "table":
        output_table(results)

    # Set exit code based on success
    successful_count = sum(1 for r in results if r.get("success", False))
    sys.exit(0 if successful_count == len(results) else 1)


if __name__ == "__main__":
    main()
