from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from .helpers import build_cli_cmd, run_cli, unique_output_base

REPO_ROOT = Path(__file__).resolve().parents[2]

# CoNLL-U validator (UD Tools). We do NOT redistribute it due to GPL licensing.
# Obtain it from: ## [validate.py](https://github.com/UniversalDependencies/tools/blob/master/validate.py)
# Configure path via env var UD_VALIDATE_SCRIPT or place a copy at tools/validate.py.
_env_validate = os.environ.get("UD_VALIDATE_SCRIPT") or os.environ.get("ROOKEEN_UD_VALIDATE_SCRIPT")
VALIDATE_SCRIPT = Path(_env_validate) if _env_validate else (REPO_ROOT / "tools" / "validate.py")
DATA_DIR = Path(__file__).resolve().parents[1] / "test_data" / "ud"


def _stanza_available() -> bool:
    uv_exe = shutil.which("uv")
    cmd = [uv_exe, "run", "python", "-c", "import stanza"] if uv_exe else None
    if cmd is None:
        import sys
        cmd = [sys.executable, "-c", "import stanza"]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def _validate(path: str, level: int) -> bool:
    if not VALIDATE_SCRIPT.exists():
        pytest.skip(
            "UD validate.py not found. Set UD_VALIDATE_SCRIPT to its path or place it at tools/validate.py. "
            "See https://github.com/UniversalDependencies/tools/blob/master/validate.py"
        )

    uv_exe = shutil.which("uv")
    args = [
        uv_exe,
        "run",
        "python",
        str(VALIDATE_SCRIPT),
        "--lang",
        "en",
        "--level",
        str(level),
        path,
    ] if uv_exe else None
    if args is None:
        import sys
        args = [
            sys.executable,
            str(VALIDATE_SCRIPT),
            "--lang",
            "en",
            "--level",
            str(level),
            path,
        ]
    result = subprocess.run(args, capture_output=True, text=True)
    return "*** PASSED ***" in (result.stdout + result.stderr)


@pytest.mark.slow
def test_conllu_export_clean_and_web_local() -> None:
    if not _stanza_available():
        pytest.skip("Stanza not available (install with 'uv sync --extra ud')")

    # Clean text
    out_base = unique_output_base("conllu-local-clean")
    out_json = str(out_base) + ".json"
    out_conllu = str(out_base) + ".conllu"
    args = build_cli_cmd(
        "analyze-file",
        str(DATA_DIR / "clean_en.txt"),
        "--lang",
        "en",
        "--export-conllu",
        "--conllu-engine",
        "stanza",
        "-o",
        str(out_base),
    )
    proc = run_cli(args)
    assert proc.returncode == 0, proc.stderr
    assert os.path.exists(out_json)
    assert os.path.exists(out_conllu)
    assert _validate(out_conllu, 1)
    assert _validate(out_conllu, 2)

    # Web-like text
    out_base2 = unique_output_base("conllu-local-web")
    out_json2 = str(out_base2) + ".json"
    out_conllu2 = str(out_base2) + ".conllu"
    args2 = build_cli_cmd(
        "analyze-file",
        str(DATA_DIR / "web_en.txt"),
        "--lang",
        "en",
        "--export-conllu",
        "--conllu-engine",
        "stanza",
        "-o",
        str(out_base2),
    )
    proc2 = run_cli(args2)
    assert proc2.returncode == 0, proc2.stderr
    assert os.path.exists(out_json2)
    assert os.path.exists(out_conllu2)
    assert _validate(out_conllu2, 1)
    assert _validate(out_conllu2, 2)


