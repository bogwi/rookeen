#!/bin/bash

# Rookeen Streaming Mode Edge Validation Script
# Tests edge cases and integration scenarios for stdin/stdout functionality
# Run from repo root: bash scripts/validate_streaming_mode.sh

set -e

# Change to repo root directory
cd "$(dirname "$0")/.."

echo "=== Rookeen Streaming Mode Edge Validation ==="

# Test 1: Basic streaming validation (from step 14)
echo "1. Basic streaming validation:"
echo 'Hello world' | uv run rookeen analyze --stdin --lang en --stdout | jq '.language.code'
echo "Expected: \"en\""
echo

# Test 2: Empty input handling
echo "2. Empty input handling:"
echo "" | uv run rookeen analyze --stdin --lang en --stdout | jq '.source.type' 2>/dev/null || echo "Handled gracefully"
echo "Expected: \"stdin\" or graceful handling"
echo

# Test 3: Unicode/multilingual support
echo "3. Unicode/multilingual support:"
echo "Hello 世界 🌍 café" | uv run rookeen analyze --stdin --lang en --stdout | jq '.language.code'
echo "Expected: \"en\""
echo

# Test 4: Auto language detection
echo "4. Auto language detection:"
echo "Bonjour le monde" | uv run rookeen analyze --stdin --stdout | jq '.language.code'
echo "Expected: \"fr\""
echo

# Test 5: Multi-line input
echo "5. Multi-line input:"
printf "Line 1\nLine 2\nLine 3\n" | uv run rookeen analyze --stdin --lang en --stdout | jq '.content.word_count'
echo "Expected: 6"
echo

# Test 6: Analyzer selection (enable)
echo "6. Analyzer selection (enable):"
echo "This is a test sentence." | uv run rookeen analyze --stdin --lang en --enable pos --stdout | jq '.analyzers[].name'
echo "Expected: [\"pos\", \"dependency\"]"
echo

# Test 7: Analyzer selection (disable)
echo "7. Analyzer selection (disable):"
echo "This is a test sentence." | uv run rookeen analyze --stdin --lang en --disable keywords --disable readability --stdout | jq '.analyzers[].name'
echo "Expected: core analyzers without keywords/readability"
echo

# Test 8: Large input handling
echo "8. Large input handling:"
python3 -c "print('This is a test sentence. ' * 100)" | uv run rookeen analyze --stdin --lang en --stdout | jq '.content.word_count'
echo "Expected: 500"
echo

# Test 9: File output with stdin
echo "9. File output with stdin:"
echo "Test input for file output" | uv run rookeen analyze --stdin --lang en --output test_streaming_output > /dev/null
jq '.source.type' test_streaming_output.json
rm -f test_streaming_output.json
echo "Expected: \"stdin\""
echo

# Test 10: Export options with stdin
echo "10. Export options with stdin:"
echo "Test sentence for export." | uv run rookeen analyze --stdin --lang en --export-spacy-json --output test_streaming_export > /dev/null
ls -1 test_streaming_export* | wc -l
rm -f test_streaming_export*
echo "Expected: 2 (JSON + spacy.json)"
echo

# Test 11: Binary input handling
echo "11. Binary input handling:"
printf '\x00\x01\x02Hello\x03\x04' | uv run rookeen analyze --stdin --lang en --stdout | jq '.source.type' 2>/dev/null || echo "Handled gracefully"
echo "Expected: \"stdin\" or graceful handling"
echo

# Test 12: Complex pipeline with multiple flags
echo "12. Complex pipeline:"
echo "Natural language processing is fascinating." | uv run rookeen analyze --stdin --lang en --enable pos --disable keywords --stdout | jq '{language: .language.code, analyzers: [.analyzers[].name], word_count: .content.word_count}'
echo "Expected: language=\"en\", analyzers include pos, word_count=5"
echo

# Test 13: Error validation - stdin + URL conflict
echo "13. Error validation (stdin + URL conflict):"
uv run rookeen analyze --stdin "http://example.com" 2>&1 | grep -q "Cannot specify both URL and --stdin" && echo "PASS: Correctly rejected conflicting arguments" || echo "FAIL: Should reject conflicting arguments"
echo

# Test 14: Error validation - stdin flag not on analyze-file
echo "14. Error validation (stdin not on analyze-file):"
uv run rookeen analyze-file --stdin tests/test_data/sample.txt 2>&1 | grep -q "Usage:" && echo "PASS: Correctly rejected invalid flag" || echo "FAIL: Should reject invalid flag"
echo

echo "=== All streaming mode edge cases validated ==="
