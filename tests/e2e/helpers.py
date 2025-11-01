from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_cli_cmd(*sub_args: str) -> list[str]:
    """Build a robust command to invoke the CLI.

    Preference order:
    1) `uv run rookeen ...` (project-managed env and console script)
    2) `rookeen ...` (console script on PATH)
    3) `python -m rookeen ...` (module entry)
    4) `python -m rookeen.cli ...` (legacy module path)
    """
    uv_exe = shutil.which("uv")
    if uv_exe:
        return [uv_exe, "run", "rookeen", *sub_args]

    # Console script available on PATH
    rook_exe = shutil.which("rookeen")
    if rook_exe:
        return [rook_exe, *sub_args]

    # Module entry (requires package installed in current interpreter)
    return [sys.executable, "-m", "rookeen", *sub_args]


def run_cli(args: list[str], timeout: int = 300, retries: int = 2) -> subprocess.CompletedProcess:
    """Run a command non-interactively and return the CompletedProcess.

    The command should be fully specified (including `uv run` if desired).

    Args:
        args: Command arguments
        timeout: Timeout in seconds for each attempt (default: 300)
        retries: Number of retry attempts (default: 2)
    """
    last_exception: subprocess.TimeoutExpired | Exception | None = None
    for attempt in range(retries + 1):
        try:
            return subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            last_exception = e
            if attempt < retries:
                time.sleep(2**attempt)  # Exponential backoff
                continue
            raise e
        except Exception as e:
            last_exception = e
            if attempt < retries:
                time.sleep(2**attempt)  # Exponential backoff
                continue
            raise e

    # This should not be reached, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected error in run_cli")


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def assert_json_key(dct: dict[str, Any], keypath: str) -> Any:
    """Assert a dotted key path exists and return the value.

    Example: keypath="language.code"
    """
    parts = keypath.split(".")
    cur: Any = dct
    for part in parts:
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
            continue
        raise AssertionError(f"Missing key path: {keypath}")
    return cur


def extract_json_from_stdout(stdout: str) -> Any:
    """Extract the JSON payload from CLI stdout that may contain installer noise.

    Some dependencies (e.g., spaCy model auto-download) can print to stdout
    even when the CLI is configured to stream only JSON. This helper extracts
    the JSON object by locating the first '{' and the last '}' and parsing
    the enclosed substring.
    """
    # Prefer matching at the known start of our payload if present
    marker = '"tool": "rookeen"'
    start = stdout.find('{')
    if marker in stdout:
        mpos = stdout.find(marker)
        # backtrack to the preceding '{'
        brace_pos = stdout.rfind('{', 0, mpos + 1)
        if brace_pos != -1:
            start = brace_pos
    end = stdout.rfind('}')
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not locate JSON object in stdout")
    payload = stdout[start : end + 1]
    return json.loads(payload)


def assert_lang(dct: dict[str, Any], expected_code: str) -> None:
    code = assert_json_key(dct, "language.code")
    if code != expected_code:
        raise AssertionError(f"language.code expected {expected_code}, got {code}")


def assert_min_count(dct: dict[str, Any], keypath: str, min_value: int) -> None:
    value = assert_json_key(dct, keypath)
    try:
        numeric = int(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise AssertionError(f"Value at {keypath} is not an int-like: {value}") from exc
    if numeric < min_value:
        raise AssertionError(f"Value at {keypath} expected >= {min_value}, got {numeric}")


def unique_output_base(prefix: str) -> Path:
    """Create a time-based unique base under results/ for outputs."""
    ensure_results_dir()
    ts = int(time.time())
    safe_prefix = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in prefix)
    return RESULTS_DIR / f"{safe_prefix}_{ts}"


def find_analyzer(payload: dict[str, Any], name: str) -> dict[str, Any]:
    analyzers = payload.get("analyzers", [])
    for a in analyzers:
        if isinstance(a, dict) and a.get("name") == name:
            return a
    raise AssertionError(f"Analyzer not found: {name}")


def assert_analyzer_has_model_metadata(payload: dict[str, Any], name: str) -> None:
    analyzer = find_analyzer(payload, name)
    md = analyzer.get("metadata", {})
    model = md.get("model")
    if not model or not isinstance(model, str):
        raise AssertionError(f"Analyzer {name} missing metadata.model")
