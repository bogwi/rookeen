from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest

from .helpers import build_cli_cmd, extract_json_from_stdout

DATA_DIR = Path(__file__).resolve().parents[1] / "test_data" / "embeddings"


def _embed(text: str) -> list[float]:
    cmd = build_cli_cmd(
        "analyze",
        "--stdin",
        "--stdout",
        "--enable-embeddings",
        "--embeddings-backend",
        "miniLM",
        "--embeddings-preload",
        "--enable",
        "embeddings",
    )
    proc = subprocess.run(cmd, input=text, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    payload = extract_json_from_stdout(proc.stdout)
    for analyzer in payload.get("analyzers", []):
        if analyzer.get("name") == "embeddings":
            return [float(x) for x in analyzer.get("results", {}).get("vector", [])]
    raise AssertionError("embeddings analyzer missing")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@pytest.mark.smoke
def test_minilm_similarity_thresholds() -> None:
    similar_path = DATA_DIR / "pairs_similar.jsonl"
    dissimilar_path = DATA_DIR / "pairs_dissimilar.jsonl"

    with similar_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pair = json.loads(line)
            v1 = _embed(pair["text1"])
            v2 = _embed(pair["text2"])
            cosine = _cosine(v1, v2)
            assert cosine >= 0.60

    with dissimilar_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pair = json.loads(line)
            v1 = _embed(pair["text1"])
            v2 = _embed(pair["text2"])
            cosine = _cosine(v1, v2)
            assert cosine <= 0.30

