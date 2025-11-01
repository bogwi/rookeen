#!/usr/bin/env python3
"""
Rookeen Pipeline Tools - Industry-Ready Unix Pipeline Utilities

This module provides additional CLI tools that demonstrate Unix pipeline
best practices and enable seamless integration with shell pipelines.

Key Features:
- Pure structured output to stdout
- Human-readable messages to stderr
- Proper exit codes for automation
- JSON/CSV output formats
- Pipeline composition support
"""

import argparse
import json
import sys
from typing import Any


def _infer_backend_from_result(result: dict[str, Any]) -> str | None:
    """Best-effort inference of embeddings backend from a benchmark result.

    Heuristics based on benchmark case naming from run_bench.py:
    - en_wiki_bge_m3 → bge-m3
    - en_wiki_openai_te3 → openai-te3
    - en_wiki_embeddings or cases including embeddings without specific suffix → miniLM (default)
    """
    case = str(result.get("case", "")).lower()
    analyzers = str(result.get("analyzers", "")).lower()
    if "bge_m3" in case or "bge-m3" in case:
        return "bge-m3"
    if "openai_te3" in case or "openai-te3" in case:
        return "openai-te3"
    if "embeddings" in analyzers:
        return "miniLM"
    return None


def analyze_benchmark_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze benchmark results and return summary statistics."""
    if not results:
        return {"error": "No results to analyze"}

    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]
    times = [r.get("seconds", 0) for r in results]

    # Per-backend breakdown (best-effort)
    backend_groups: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        b = _infer_backend_from_result(r)
        if b is None:
            continue
        backend_groups.setdefault(b, []).append(r)

    by_backend: dict[str, dict[str, Any]] = {}
    for b, grp in backend_groups.items():
        grp_times = [x.get("seconds", 0) for x in grp]
        grp_success = [x for x in grp if x.get("success", False)]
        by_backend[b] = {
            "count": len(grp),
            "avg_time": (sum(grp_times) / len(grp_times)) if grp_times else 0,
            "success_rate": (len(grp_success) / len(grp)) if grp else 0,
        }

    return {
        "total": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / len(results) if results else 0,
        "avg_time": sum(times) / len(times) if times else 0,
        "min_time": min(times) if times else 0,
        "max_time": max(times) if times else 0,
        "total_time": sum(times),
        "fastest_case": min(successful, key=lambda x: x.get("seconds", 0))["case"]
        if successful
        else None,
        "slowest_case": max(successful, key=lambda x: x.get("seconds", 0))["case"]
        if successful
        else None,
        "by_backend": by_backend,
    }


def cmd_analyze_benchmarks(args: argparse.Namespace) -> None:
    """Analyze benchmark results from stdin or file."""
    # Read input
    if args.file:
        try:
            with open(args.file) as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading file {args.file}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            data = json.load(sys.stdin)
        except Exception as e:
            print(f"Error reading from stdin: {e}", file=sys.stderr)
            sys.exit(1)

    # Analyze results
    if isinstance(data, list):
        analysis = analyze_benchmark_results(data)
    else:
        analysis = analyze_benchmark_results([data])

    # Output in requested format
    if args.format == "json":
        print(json.dumps(analysis, indent=2))
    elif args.format == "csv":
        print("metric,value")
        for key, value in analysis.items():
            if isinstance(value, int | float):
                print(f"{key},{value}")
            else:
                print(f"{key},{value}")
    else:  # table
        print("Benchmark Analysis Summary")
        print("=" * 30)
        for _key, value in analysis.items():
            if isinstance(value, float):
                print("25")
            else:
                print("25")


def cmd_filter_benchmarks(args: argparse.Namespace) -> None:
    """Filter benchmark results based on criteria."""
    try:
        data = json.load(sys.stdin)
    except Exception as e:
        print(f"Error reading from stdin: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, list):
        data = [data]

    filtered = []

    for result in data:
        include = True

        # Apply filters
        if args.success_only and not result.get("success", False):
            include = False
        if args.failed_only and result.get("success", False):
            include = False
        if args.min_time is not None and result.get("seconds", 0) < args.min_time:
            include = False
        if args.max_time is not None and result.get("seconds", 0) > args.max_time:
            include = False
        if args.language and result.get("language") != args.language:
            include = False
        if args.analyzer and args.analyzer not in result.get("analyzers", ""):
            include = False
        if args.embeddings_backend:
            inferred = _infer_backend_from_result(result)
            if inferred is None or inferred.lower() != args.embeddings_backend.lower():
                include = False

        if include:
            filtered.append(result)

    # Output filtered results
    if args.count:
        print(len(filtered))
    else:
        print(json.dumps(filtered, indent=2))


def cmd_compare_benchmarks(args: argparse.Namespace) -> None:
    """Compare two sets of benchmark results."""
    try:
        # Read first dataset
        if args.file1 == "-":
            data1 = json.load(sys.stdin)
        else:
            with open(args.file1) as f:
                data1 = json.load(f)

        # Read second dataset
        if args.file2 == "-":
            data2 = json.load(sys.stdin)
        else:
            with open(args.file2) as f:
                data2 = json.load(f)
    except Exception as e:
        print(f"Error reading input: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data1, list):
        data1 = [data1]
    if not isinstance(data2, list):
        data2 = [data2]

    # Create comparison
    comparison: dict[str, Any] = {
        "dataset1": analyze_benchmark_results(data1),
        "dataset2": analyze_benchmark_results(data2),
        "improvement": {},
    }

    # Calculate improvements
    for metric in ["avg_time", "success_rate"]:
        val1 = comparison["dataset1"].get(metric, 0)
        val2 = comparison["dataset2"].get(metric, 0)
        if metric == "avg_time":
            # Lower is better for time
            improvement = ((val1 - val2) / val1) * 100 if val1 > 0 else 0
        else:
            # Higher is better for rates
            improvement = ((val2 - val1) / val1) * 100 if val1 > 0 else 0
        comparison["improvement"][metric] = improvement

    print(json.dumps(comparison, indent=2))


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Rookeen Pipeline Tools - Unix Pipeline Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Analyze benchmark results
  uv run python bench/run_bench.py --json --quiet | python scripts/pipeline_tools.py analyze

  # Filter successful benchmarks only
  uv run python bench/run_bench.py --json --quiet | python scripts/pipeline_tools.py filter --success-only

  # Count failed benchmarks
  uv run python bench/run_bench.py --json --quiet | python scripts/pipeline_tools.py filter --failed-only --count

  # Compare benchmark runs
  python scripts/pipeline_tools.py compare bench/results/latest.json bench/results/previous.json

  # Filter by performance criteria
  uv run python bench/run_bench.py --json --quiet | python scripts/pipeline_tools.py filter --max-time 3.0
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze benchmark results")
    analyze_parser.add_argument("--file", "-f", help="Input file (default: stdin)")
    analyze_parser.add_argument(
        "--format", choices=["json", "csv", "table"], default="json", help="Output format"
    )
    analyze_parser.set_defaults(func=cmd_analyze_benchmarks)

    # Filter command
    filter_parser = subparsers.add_parser("filter", help="Filter benchmark results")
    filter_parser.add_argument(
        "--success-only", action="store_true", help="Only successful benchmarks"
    )
    filter_parser.add_argument("--failed-only", action="store_true", help="Only failed benchmarks")
    filter_parser.add_argument("--min-time", type=float, help="Minimum execution time")
    filter_parser.add_argument("--max-time", type=float, help="Maximum execution time")
    filter_parser.add_argument("--language", help="Filter by language")
    filter_parser.add_argument("--analyzer", help="Filter by analyzer (substring match)")
    filter_parser.add_argument(
        "--embeddings-backend",
        help="Filter to results using a specific embeddings backend (miniLM, bge-m3, openai-te3)",
    )
    filter_parser.add_argument("--count", action="store_true", help="Output count only")
    filter_parser.set_defaults(func=cmd_filter_benchmarks)

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two benchmark result sets")
    compare_parser.add_argument("file1", help="First benchmark results file (or - for stdin)")
    compare_parser.add_argument("file2", help="Second benchmark results file (or - for stdin)")
    compare_parser.set_defaults(func=cmd_compare_benchmarks)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
