from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from .helpers import build_cli_cmd, extract_json_from_stdout

pytestmark = [pytest.mark.slow]

DATA_DIR = Path(__file__).resolve().parents[1] / "test_data" / "sentiment"


def _sentiment_available() -> bool:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # noqa: F401

        return True
    except Exception:
        pass
    try:
        from textblob import TextBlob  # noqa: F401

        return True
    except Exception:
        return False


def test_sentiment_ranges() -> None:
    if not _sentiment_available():
        pytest.skip("sentiment libraries not available (install with 'uv sync --extra sentiment')")

    def analyze_text(path: Path) -> Any:
        text = path.read_text(encoding="utf-8")
        cmd = build_cli_cmd("analyze", "--stdin", "--stdout", "--enable-sentiment")
        proc = subprocess.run(cmd, input=text, text=True, capture_output=True)
        assert proc.returncode == 0, proc.stderr
        return extract_json_from_stdout(proc.stdout)

    pos = analyze_text(DATA_DIR / "pos.txt")
    neg = analyze_text(DATA_DIR / "neg.txt")
    neu = analyze_text(DATA_DIR / "neu.txt")

    def _get_compound(payload: dict) -> float:
        for a in payload.get("analyzers", []):
            if a.get("name") == "sentiment":
                res = a.get("results", {})
                if not res.get("supported"):
                    pytest.skip("sentiment not supported by available libraries")
                if "scores" in res and "compound" in res["scores"]:
                    return float(res["scores"]["compound"])
                if "polarity" in res:
                    return float(res["polarity"])
        raise AssertionError("sentiment analyzer missing")

    c_pos = _get_compound(pos)
    c_neg = _get_compound(neg)
    c_neu = _get_compound(neu)

    assert c_pos > 0.5
    assert c_neg < -0.5
    assert -0.2 <= c_neu <= 0.2

