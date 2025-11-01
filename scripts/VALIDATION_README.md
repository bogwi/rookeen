# JSON Schema Validation for Rookeen

This document describes the JSON schema validation system implemented for Rookeen outputs, providing contract-first development and ensuring data consistency.

## Overview

The validation system consists of:
- **JSON Schema**: Formal specification of Rookeen output format (`schemas/rookeen_v1.json`)
- **Validation Scripts**: Multiple tools for validating JSON files
- **Test Suite**: Comprehensive tests covering various scenarios
- **Test Data**: Sample files for validation testing

## JSON Schema Structure

The schema (`schemas/rookeen_v1.json`) validates:

### Required Top-Level Fields
- `tool`: Must be "rookeen"
- `version`: Version string
- `language`: Language detection results
- `analyzers`: Array of analyzer results

### Language Object
```json
{
  "code": "en",
  "confidence": 0.95,
  "model": "en_core_web_sm"
}
```

### Analyzer Types Supported
1. **lexical_stats** - Token counts, sentence stats, top lemmas
2. **pos** - Part-of-speech tagging with UPOS counts and ratios
3. **ner** - Named entity recognition with entity counts and examples
4. **readability** - Multiple readability metrics
5. **keywords** - YAKE-based keyword extraction
6. **sentiment** - Sentiment analysis with scores
7. **dependency** - Dependency parsing statistics
8. **embeddings** - Sentence embeddings with vectors

## Unified Validation Script

### CI Validator (`scripts/validate_for_ci.py`)
Comprehensive CI/CD-friendly validation script that combines all validation functionality:
- Detailed error reporting with schema validation failures
- PASS/FAIL output per input with appropriate exit codes
- Support for processing URLs directly or validating existing files
- Analyzer selection flags for custom validation scenarios
- Automatic cleanup of temporary files
- Verbose and quiet modes for different CI/CD needs

```bash
# Basic validation of existing JSON files
uv run python scripts/validate_for_ci.py results/*.json

# Process URLs directly (robots.txt ignored for testing)
uv run python scripts/validate_for_ci.py "https://example.com/article"

# Mix URLs and files
uv run python scripts/validate_for_ci.py results/data.json "https://example.com/article"

# Verbose output with analyzer details
uv run python scripts/validate_for_ci.py results/*.json --verbose

# Quiet mode for CI/CD pipelines
uv run python scripts/validate_for_ci.py results/*.json --quiet

# Custom analyzer selection for URLs
uv run python scripts/validate_for_ci.py "https://example.com/article" --enable pos --enable ner --enable-embeddings

# Disable specific analyzers
uv run python scripts/validate_for_ci.py "https://example.com/article" --disable keywords --disable readability
```

### Test Suite (`scripts/test_validation.py`)
Runs comprehensive tests including:
- Basic functionality tests with URL processing
- Error handling tests for invalid data
- Schema structure validation
- Real data validation against live URLs

```bash
uv run python scripts/test_validation.py
```

## Test Data

The `test_data/` directory contains sample files for testing:

- `valid_minimal.json` - Minimal valid output
- `valid_comprehensive.json` - Complete output with all analyzers
- `invalid_missing_required.json` - Missing required fields
- `invalid_wrong_types.json` - Wrong data types
- `invalid_analyzer_missing_required.json` - Analyzer missing required fields

## Key Features

### URL Processing (Result-Agnostic)
- **Process URLs directly**: Scripts can analyze web pages on-demand without requiring existing result files
- **Automatic cleanup**: Temporary JSON files are created and cleaned up automatically
- **Robots.txt handling**: Automatically ignores robots.txt for testing purposes
- **Flexible input**: Mix URLs and existing JSON files in the same command

### Analyzer Selection
- **Enable specific analyzers**: Use `--enable analyzer_name` flags
- **Disable analyzers**: Use `--disable analyzer_name` flags
- **Optional analyzers**: Support for `--enable-embeddings` and `--enable-sentiment`
- **Validation coverage**: All analyzer combinations are validated against the schema

### Comprehensive Testing
- **Test data**: Includes valid and invalid examples for all scenarios
- **Real-world validation**: Tests against actual Rookeen processing output
- **Error scenarios**: Validates error handling for malformed data and processing failures
- **CI/CD ready**: Appropriate exit codes and automation-friendly output

## Usage Examples

### Quick Validation
```bash
# Validate all result files
uv run python scripts/validate_for_ci.py results/*.json

# Get detailed report with verbose output
uv run python scripts/validate_for_ci.py results/*.json --verbose

# Quiet mode for minimal output
uv run python scripts/validate_for_ci.py results/*.json --quiet
```

### CI/CD Integration
```bash
# In CI pipeline - validate existing files
uv run python scripts/validate_for_ci.py results/*.json --quiet || exit 1

# Test with specific URLs (robots.txt ignored)
uv run python scripts/validate_for_ci.py "https://en.wikipedia.org/w/index.php?title=Natural_language_processing&oldid=1201524046"

# Validate multiple files and URLs
uv run python scripts/validate_for_ci.py results/*.json "https://example.com/article"
```

### Development Testing
```bash
# Run full test suite
uv run python scripts/test_validation.py

# Test specific files
uv run python scripts/validate_for_ci.py test_data/valid_*.json --verbose
uv run python scripts/validate_for_ci.py test_data/invalid_*.json
```

## Error Types Handled

The validation system catches and reports:

1. **Missing required fields** - Top-level or analyzer-specific required fields
2. **Wrong data types** - String vs number, array vs object, etc.
3. **Invalid values** - Out-of-range numbers, invalid enums
4. **Malformed JSON** - JSON parsing errors
5. **Schema conflicts** - Multiple analyzer schemas matching incorrectly

## Schema Features

### Discriminating Fields
Each analyzer type has unique required fields to prevent schema conflicts:
- `lexical_stats`: `total_tokens`
- `pos`: `upos_counts`
- `ner`: `counts_by_label`
- `readability`: `flesch_reading_ease`
- `keywords`: `method` + `keywords`
- `sentiment`: `label`
- `dependency`: `dep_counts`
- `embeddings`: `vector`

### Flexible Validation
- Supports optional fields throughout
- Handles different sentiment analysis formats
- Validates array structures with proper typing
- Pattern matching for dynamic keys (POS tags, entity labels)

## Integration with Rookeen Pipeline

The validation system integrates seamlessly with Rookeen:

1. **Output Generation**: All Rookeen CLI commands produce schema-compliant JSON
2. **Validation**: Can validate any Rookeen output file
3. **Error Handling**: Clear error messages for debugging
4. **Extensibility**: Easy to add new analyzer types to schema

## Best Practices

1. **Always validate outputs** in development and CI/CD
2. **Use detailed validator** for debugging validation issues
3. **Run test suite** after schema changes
4. **Keep test data updated** when adding new analyzer features
5. **Document schema changes** in this README

## Troubleshooting

### Common Issues

**"Failed at: analyzers -> 0 -> results"**
- Analyzer results don't match any known schema
- Check analyzer name and required fields

**"Invalid value: X"**
- Field has wrong data type or out-of-range value
- Check schema constraints for that field

**"Schema validation failed"**
- General validation error
- Use detailed validator for specific error information

### Debugging Steps

1. Run with verbose output: `uv run python scripts/validate_for_ci.py file.json --verbose`
2. Check the specific error path and message in the detailed report
3. Compare with test data examples
4. Verify analyzer output format matches schema expectations

## Future Enhancements

- Add schema versioning support
- Implement schema migration tools
- Add performance benchmarking for large files
- Create web-based schema validator
- Add schema documentation generation
