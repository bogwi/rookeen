from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from .helpers import (
    assert_analyzer_has_model_metadata,
    assert_json_key,
    build_cli_cmd,
    extract_json_from_stdout,
    find_analyzer,
)

pytestmark = [pytest.mark.slow]

DATA_FILE = (
    Path(__file__).resolve().parents[1] / "test_data" / "longform" / "gutenberg_en_excerpt.txt"
)


def test_longform_local_stability() -> None:
    text = DATA_FILE.read_text(encoding="utf-8")
    cmd = build_cli_cmd("analyze", "--stdin", "--lang", "en", "--stdout")
    proc = subprocess.run(cmd, input=text, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    payload = extract_json_from_stdout(proc.stdout)

    for key in ("tool", "version", "language", "content", "analyzers", "timing"):
        assert key in payload

    assert assert_json_key(payload, "language.code") == "en"

    lex = find_analyzer(payload, "lexical_stats")
    total_tokens = int(lex.get("results", {}).get("total_tokens", 0))
    assert total_tokens >= 800

    for name in ("pos", "ner", "lexical_stats", "readability", "keywords", "dependency"):
        assert_analyzer_has_model_metadata(payload, name)

