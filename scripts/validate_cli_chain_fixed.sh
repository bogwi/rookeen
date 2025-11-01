#!/bin/bash

# Rookeen CLI Chain Validation Script
# Tests CLI error handling, validation, and edge cases
# Run from repo root: bash scripts/validate_cli_chain_fixed.sh
#
# Note: set -e is NOT used here because this script intentionally tests error conditions

# Change to repo root directory
cd "$(dirname "$0")/.."

echo "=== Rookeen CLI Chain Validation (Fixed) ==="
echo

run_cmd() {
    local cmd="$1"
    local desc="$2"
    echo "$desc"
    
    # Capture both output and exit code
    local output
    local exit_code
    
    # Run command and capture output
    output=$(eval "$cmd" 2>&1)
    exit_code=$?
    
    # Show first few lines of output
    echo "$output" | head -3
    
    # Show exit code
    echo "Exit code: $exit_code"
    echo
}

echo "Test 1: CLI Parsing Errors"
run_cmd 'uv run rookeen analyze' "1a. Missing arguments (normal):"
run_cmd 'uv run rookeen --errors-json analyze' "1b. Missing arguments (JSON):"
run_cmd 'uv run rookeen --invalid-global analyze' "1c. Invalid global options (normal):"
run_cmd 'uv run rookeen --errors-json --invalid-global analyze' "1d. Invalid global options (JSON):"
run_cmd 'uv run rookeen analyze --invalid-option' "1e. Invalid command options (normal):"
run_cmd 'uv run rookeen --errors-json analyze --invalid-option' "1f. Invalid command options (JSON):"

echo "Test 2: File I/O Errors"
run_cmd 'uv run rookeen analyze-file nonexistent.txt' "2a. Non-existent input file (normal):"
run_cmd 'uv run rookeen --errors-json analyze-file nonexistent.txt' "2b. Non-existent input file (JSON):"
run_cmd 'uv run rookeen batch nonexistent.txt' "2c. Non-existent batch file (normal):"
run_cmd 'timeout 10 uv run rookeen --errors-json batch nonexistent.txt' "2d. Non-existent batch file (JSON):"

echo "Test 3: Network/Fetch Errors"
run_cmd 'timeout 5 uv run rookeen analyze "http://invalid-domain-12345.com" --models-auto-download' "3a. Invalid domain (normal):"
run_cmd 'timeout 5 uv run rookeen --errors-json analyze "http://invalid-domain-12345.com" --models-auto-download' "3b. Invalid domain (JSON):"

echo "Test 4: Successful Operations"
run_cmd 'uv run rookeen analyze-file tests/test_data/sample.txt --stdout --lang en 2>/dev/null | jq -r ".source.type" 2>/dev/null || echo "Command succeeded but JSON parsing failed"' "4a. Analyze file successfully:"
run_cmd 'uv run rookeen analyze-file tests/test_data/sample.txt --stdout --lang en 2>/dev/null | jq -r ".language.code" 2>/dev/null || echo "Command succeeded but JSON parsing failed"' "4b. Analyze with stdout and pipe to jq:"

echo "Test 5: Mixed Pipeline Scenarios"
echo "5a. Batch with empty file:"
echo "" > empty_urls.txt
run_cmd 'uv run rookeen batch empty_urls.txt' "   Batch with empty file:"
rm -f empty_urls.txt

echo "5b. Batch with invalid URLs:"
echo -e "http://invalid1.com\nhttp://invalid2.com" > invalid_urls.txt
run_cmd 'timeout 10 uv run rookeen batch invalid_urls.txt --output-dir /tmp/test_output' "   Batch with invalid URLs:"
rm -f invalid_urls.txt
rm -rf /tmp/test_output 2>/dev/null || true

echo "Test 6: Flag Combinations and Edge Cases"
run_cmd 'uv run rookeen --errors-json analyze --flag1 --flag2 --flag3' "6a. Multiple invalid flags:"
run_cmd 'uv run rookeen --errors-json analyze-file tests/test_data/sample.txt --lang invalid_lang' "6b. Valid flags with invalid values:"
run_cmd 'uv run rookeen --errors-json analyze-file tests/test_data/sample.txt --stdout --output test.json' "6c. Conflicting output options:"

echo "=== Validation Complete ==="
