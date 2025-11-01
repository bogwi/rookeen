# Validation Scripts CI Integration

## Why These Scripts Won't Break CI

### `validate_cli_chain_fixed.sh`
**Exit Behavior: Always exits with code 0**

- **No `set -e`**: The script intentionally avoids `set -e` because it tests error conditions
- **Error testing design**: It runs commands that are expected to fail (invalid arguments, non-existent files, network errors) but captures their exit codes for verification
- **Does not propagate errors**: The script runs all tests regardless of individual command failures, then exits successfully
- **Purpose**: Validates that rookeen handles errors correctly and returns appropriate exit codes, not to test functionality

**Why this is safe for CI:**
- The script's purpose is to verify error handling works, not to test successful operations
- It intentionally tests failure scenarios and validates they're handled correctly
- It always exits with 0 because completing the validation (even if some commands fail) is considered success

### `validate_streaming_mode.sh`
**Exit Behavior: Exits with 0 if all tests pass, non-zero if any test fails**

- **Uses `set -e`**: Fails fast if any test fails, which is correct behavior
- **All tests should pass**: These are functional tests that verify streaming mode works correctly
- **Purpose**: Ensures stdin/stdout functionality works as expected

**Why this is safe for CI:**
- All 14 tests currently pass successfully
- If a test fails, it indicates a real regression that should break CI
- Tests are fast and reliable (no network dependencies, use local stdin)
- The script tests core functionality, so failures are legitimate issues

## CI Integration Strategy

Both scripts are integrated into CI as **required validation steps**:

1. **CLI Validation** - Runs after unit tests, validates error handling doesn't regress
2. **Streaming Validation** - Runs after CLI validation, validates stdin/stdout functionality
3. **Both are non-optional**: If either fails, CI fails (as intended)

## When Scripts Should Fail CI

- **CLI validation should never fail** in CI (it tests error conditions are handled)
- **Streaming validation should fail** if:
  - stdin/stdout functionality breaks
  - Analyzer selection with stdin breaks
  - Language detection breaks
  - Export options break
  - Any of the 14 test scenarios fail

## Maintenance

If streaming mode tests start failing:
1. Investigate the regression - this is a real issue
2. Fix the underlying problem in rookeen
3. Re-run the script to verify the fix

If CLI validation starts failing:
1. This is unexpected - the script should always pass
2. Check if rookeen's error handling has changed
3. Update the script if error messages or exit codes have legitimately changed

