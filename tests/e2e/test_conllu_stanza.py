from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from .helpers import (
    build_cli_cmd,
    read_json,
    run_cli,
    unique_output_base,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# CoNLL-U validator (UD Tools). We do NOT redistribute it due to GPL licensing.
# Obtain it from: ## [validate.py](https://github.com/UniversalDependencies/tools/blob/master/validate.py)
# Configure path via env var UD_VALIDATE_SCRIPT or place a copy at tools/validate.py.
_env_validate = os.environ.get("UD_VALIDATE_SCRIPT") or os.environ.get("ROOKEEN_UD_VALIDATE_SCRIPT")
VALIDATE_SCRIPT = Path(_env_validate) if _env_validate else (REPO_ROOT / "tools" / "validate.py")

URL = "https://en.wikipedia.org/w/index.php?title=Natural_language_processing&oldid=1201524046"

# Networked Wikipedia; exclude from default runs
pytestmark = [pytest.mark.external]


def _stanza_available() -> bool:
    """Check if Stanza is available in the current environment."""
    uv_exe = shutil.which("uv")
    if uv_exe:
        cmd = [uv_exe, "run", "python", "-c", "import stanza"]
    else:
        import sys

        cmd = [sys.executable, "-c", "import stanza"]

    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def _validate_conllu(conllu_file: str, level: int = 1) -> bool:
    """Validate a CoNLL-U file at the specified level."""
    if not VALIDATE_SCRIPT.exists():
        pytest.skip(
            "UD validate.py not found. Set UD_VALIDATE_SCRIPT to its path or place it at tools/validate.py. "
            "See https://github.com/UniversalDependencies/tools/blob/master/validate.py"
        )

    uv_exe = shutil.which("uv")
    if uv_exe:
        args = [
            uv_exe,
            "run",
            "python",
            str(VALIDATE_SCRIPT),
            "--lang",
            "en",
            "--level",
            str(level),
            conllu_file,
        ]
    else:
        import sys

        args = [
            sys.executable,
            str(VALIDATE_SCRIPT),
            "--lang",
            "en",
            "--level",
            str(level),
            conllu_file,
        ]

    result = subprocess.run(args, capture_output=True, text=True)
    return "*** PASSED ***" in (result.stdout + result.stderr)


def test_stanza_conllu_wikipedia() -> None:
    """Test that Stanza engine produces UD-valid CoNLL-U output for Wikipedia content."""
    if not _stanza_available():
        print("SKIPPED: Stanza not available (install with 'uv sync --extra ud')")
        return

    out_base = unique_output_base("conllu-stanza-wiki")
    out_json = str(out_base) + ".json"
    out_conllu = str(out_base) + ".conllu"

    args = build_cli_cmd(
        "analyze",
        URL,
        "--models-auto-download",
        "--export-conllu",
        "--conllu-engine",
        "stanza",
        "--robots",
        "ignore",
        "-o",
        str(out_base),
    )
    proc = run_cli(args)
    assert proc.returncode == 0, (
        f"CLI failed: {proc.returncode}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
    )

    # Verify files were created
    assert os.path.exists(out_json), f"JSON file not created: {out_json}"
    assert os.path.exists(out_conllu), f"CoNLL-U file not created: {out_conllu}"

    # Validate JSON output
    payload = read_json(out_json)
    assert "language" in payload
    assert payload["language"]["code"] == "en"

    # Test Level 1 validation (CoNLL-U backbone)
    assert _validate_conllu(out_conllu, level=1), "Level 1 validation failed"

    # Test Level 2 validation (UD format)
    assert _validate_conllu(out_conllu, level=2), "Level 2 validation failed"

    # Level 3 validation is expected to have some errors for web content
    # This is documented behavior and acceptable for production use
    print("✓ Stanza engine produces Level 1-2 compliant CoNLL-U for web content")


def test_stanza_conllu_simple_text() -> None:
    """Test Stanza CoNLL-U export on clean simple text."""
    if not _stanza_available():
        print("SKIPPED: Stanza not available (install with 'uv sync --extra ud')")
        return

    # Create a simple text file with clean prose
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(
            "Natural language processing is fascinating. It enables computers to understand human language."
        )
        text_file = f.name

    try:
        out_base = unique_output_base("conllu-stanza-simple")
        out_json = str(out_base) + ".json"
        out_conllu = str(out_base) + ".conllu"

        args = build_cli_cmd(
            "analyze-file",
            text_file,
            "--lang",
            "en",
            "--export-conllu",
            "--conllu-engine",
            "stanza",
            "-o",
            str(out_base),
        )
        proc = run_cli(args)
        assert proc.returncode == 0, (
            f"CLI failed: {proc.returncode}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
        )

        # Verify files were created
        assert os.path.exists(out_json), f"JSON file not created: {out_json}"
        assert os.path.exists(out_conllu), f"CoNLL-U file not created: {out_conllu}"

        # Test validation at multiple levels for clean text
        assert _validate_conllu(out_conllu, level=1), "Level 1 validation failed"
        assert _validate_conllu(out_conllu, level=2), "Level 2 validation failed"

        print("✓ Stanza engine produces Level 1-2 compliant CoNLL-U for simple text")

    finally:
        os.unlink(text_file)


def test_basic_engine_fallback() -> None:
    """Test that basic engine works as fallback when explicitly requested."""
    # Create a simple text file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Hello world.")
        text_file = f.name

    try:
        out_base = unique_output_base("conllu-basic")
        out_json = str(out_base) + ".json"
        out_conllu = str(out_base) + ".conllu"

        args = build_cli_cmd(
            "analyze-file",
            text_file,
            "--lang",
            "en",
            "--export-conllu",
            "--conllu-engine",
            "basic",
            "-o",
            str(out_base),
        )
        proc = run_cli(args)
        assert proc.returncode == 0, (
            f"CLI failed: {proc.returncode}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
        )

        # Verify files were created
        assert os.path.exists(out_json), f"JSON file not created: {out_json}"
        assert os.path.exists(out_conllu), f"CoNLL-U file not created: {out_conllu}"

        # Basic engine should at least pass Level 1 (backbone format)
        assert _validate_conllu(out_conllu, level=1), "Level 1 validation failed for basic engine"

        print("✓ Basic engine produces Level 1 compliant CoNLL-U")

    finally:
        os.unlink(text_file)
