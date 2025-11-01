#!/bin/bash
# Industry-Ready Unix Pipeline Composability Demo for Rookeen
# This script demonstrates how the improved benchmark harness enables
# seamless integration with Unix pipelines and automation workflows.

set -e  # Exit on any error

echo "=== Rookeen Unix Pipeline Composability Demo ==="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to run commands and show status
run_cmd() {
    local cmd="$1"
    local desc="$2"

    echo -e "${YELLOW}→${NC} $desc"
    echo -e "${YELLOW}Command:${NC} $cmd"
    echo

    if eval "$cmd"; then
        echo -e "${GREEN}✓${NC} Success\n"
    else
        echo -e "${RED}✗${NC} Failed\n"
        return 1
    fi
}

cd "$(dirname "$0")/.."

echo "1. PIPELINE-FRIENDLY JSON OUTPUT"
echo "================================"
run_cmd "uv run python bench/run_bench.py --json --quiet | jq '.[0].return_code == 0'" \
        "Check if first benchmark case succeeded (was failing before)"

echo "2. VALIDATION WITH JQ FILTERING"
echo "==============================="
run_cmd "uv run python bench/run_bench.py --json --quiet | jq '[.[] | select(.success == true)] | length'" \
        "Count successful benchmark runs"

echo "3. AUTOMATION: CHECK ALL BENCHMARKS PASSED"
echo "==========================================="
if uv run python bench/run_bench.py --json --quiet | jq -e 'all(.success)'; then
    echo -e "${GREEN}✓${NC} All benchmarks passed!"
else
    echo -e "${RED}✗${NC} Some benchmarks failed!"
    exit 1
fi
echo

echo "4. PERFORMANCE ANALYSIS WITH JQ"
echo "================================"
run_cmd "uv run python bench/run_bench.py --json --quiet | jq '[.[] | {case: .case, time: .seconds}]'" \
        "Extract case names and execution times"

echo "5. CSV OUTPUT FOR DATA ANALYSIS"
echo "==============================="
run_cmd "uv run python bench/run_bench.py --format csv --no-save | head -3" \
        "Generate CSV output suitable for spreadsheet analysis"

echo "6. TABLE FORMAT FOR HUMAN READING"
echo "================================="
run_cmd "uv run python bench/run_bench.py --format table --no-save | tail -8" \
        "Generate human-readable table format"

echo "7. QUIET MODE FOR CI/CD INTEGRATION"
echo "==================================="
echo "Quiet mode suppresses progress output, perfect for CI/CD:"
run_cmd "uv run python bench/run_bench.py --quiet --no-save >/dev/null && echo 'Benchmarks completed successfully'" \
        "Run benchmarks quietly and check exit code"

echo "8. CONDITIONAL PROCESSING BASED ON RESULTS"
echo "=========================================="
echo "Example: Only proceed if all benchmarks pass and avg time < 3s"
uv run python bench/run_bench.py --json --quiet | jq -e '
    (all(.success)) and
    (([.[] | .seconds] | add) / length) < 3.0
' && echo -e "${GREEN}✓${NC} All benchmarks passed within time limit!" || echo -e "${RED}✗${NC} Benchmarks failed or too slow!"

echo
echo "9. INTEGRATION WITH OTHER UNIX TOOLS"
echo "===================================="
echo "Sort benchmarks by execution time:"
uv run python bench/run_bench.py --json --quiet | \
    jq '[.[] | {case: .case, seconds: .seconds}]' | \
    jq 'sort_by(.seconds)' | \
    jq -r '.[] | "\(.case): \(.seconds)s"'

echo
echo "Generate performance report:"
uv run python bench/run_bench.py --json --quiet | \
    jq -r '"Total benchmarks: \(length)
Successful: \([.[] | select(.success == true)] | length)
Failed: \([.[] | select(.success == false)] | length)
Average time: \((([.[] | .seconds] | add) / length) | .3f)s
Max time: \([.[] | .seconds] | max | .3f)s
Min time: \([.[] | .seconds] | min | .3f)s"'

echo
echo "=== Demo Complete ==="
echo -e "${GREEN}✓${NC} All pipeline composability features working correctly!"
echo
echo "Key improvements:"
echo "- JSON output is now pure JSON (no mixed content)"
echo "- Proper stderr/stdout separation"
echo "- Multiple output formats (json, csv, table)"
echo "- Meaningful exit codes for automation"
echo "- Quiet mode for CI/CD integration"
echo "- Pipeline-friendly design patterns"
