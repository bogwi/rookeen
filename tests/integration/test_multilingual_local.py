from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from rookeen.language import SUPPORTED_LANGS as SUPPORTED_LANGS
from tests.e2e.helpers import (
    assert_json_key,
    build_cli_cmd,
    extract_json_from_stdout,
    find_analyzer,
)

pytestmark = [pytest.mark.slow]


DATA_DIR = Path(__file__).resolve().parents[1] / "test_data" / "multilingual"
SUPPORTED = set(SUPPORTED_LANGS)


def _run_on_file(fname: str) -> Any:
    text = (DATA_DIR / fname).read_text(encoding="utf-8")
    cmd = build_cli_cmd("analyze", "--stdin", "--stdout")
    proc = subprocess.run(cmd, input=text, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    return extract_json_from_stdout(proc.stdout)


def test_multilingual_en() -> None:
    payload = _run_on_file("en_medium.txt")
    assert_json_key(payload, "language.code")
    assert payload["language"]["code"] in SUPPORTED
    lex = find_analyzer(payload, "lexical_stats")
    assert int(lex.get("results", {}).get("total_tokens", 0)) >= 50


def test_multilingual_de() -> None:
    payload = _run_on_file("de_medium.txt")
    assert payload["language"]["code"] in SUPPORTED


def test_multilingual_es() -> None:
    payload = _run_on_file("es_medium.txt")
    assert payload["language"]["code"] in SUPPORTED


def test_multilingual_fr() -> None:
    payload = _run_on_file("fr_medium.txt")
    assert payload["language"]["code"] in SUPPORTED


def test_multilingual_cjk() -> None:
    zh = _run_on_file("zh_medium.txt")
    ja = _run_on_file("ja_medium.txt")
    assert zh["language"]["code"] in SUPPORTED
    assert ja["language"]["code"] in SUPPORTED


def test_multilingual_rtl_and_cyrillic() -> None:
    ar = _run_on_file("ar_medium.txt")
    ua = _run_on_file("ua_medium.txt")
    assert ar["language"]["code"] in SUPPORTED
    assert ua["language"]["code"] in SUPPORTED

