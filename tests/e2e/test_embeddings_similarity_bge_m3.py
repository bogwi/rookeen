from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from .helpers import build_cli_cmd, extract_json_from_stdout

pytestmark = [pytest.mark.slow]
DATA_DIR = Path(__file__).resolve().parents[1] / "test_data" / "embeddings"


def _backend_available() -> bool:
    return bool(shutil.which("uv"))


def _embed_bge(text: str) -> list[float]:
    cmd = build_cli_cmd(
        "analyze",
        "--stdin",
        "--stdout",
        "--enable-embeddings",
        "--embeddings-backend",
        "bge-m3",
        "--embeddings-model",
        "BAAI/bge-m3",
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


def test_bge_m3_similarity_thresholds() -> None:
    if not _backend_available():
        pytest.skip("embeddings backend not available")

    similar_path = DATA_DIR / "pairs_similar.jsonl"
    dissimilar_path = DATA_DIR / "pairs_dissimilar.jsonl"

    # BGE-M3 requires higher thresholds, see docs/BGE_M3_ANALYSIS.md for more details
    similar_treshold = 0.65
    
    # Dissimilar pairs for longer texts range 0.436-0.605, see docs/BGE_M3_ANALYSIS.md for more details
    disssimilar_treshold = 0.65
    
    with similar_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pair = json.loads(line)
            v1 = _embed_bge(pair["text1"])
            v2 = _embed_bge(pair["text2"])
            similarity = _cosine(v1, v2)
            assert similarity >= similar_treshold, (
                f"Similar pair failed threshold: {similarity:.6f} < {similar_treshold}. "
                f"Text1: {pair['text1'][:50]}..., Text2: {pair['text2'][:50]}..."
            )

    with dissimilar_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pair = json.loads(line)
            v1 = _embed_bge(pair["text1"])
            v2 = _embed_bge(pair["text2"])
            similarity = _cosine(v1, v2)
            assert similarity <= disssimilar_treshold, (
                f"Dissimilar pair failed threshold: {similarity:.6f} > {disssimilar_treshold}. "
                f"Text1: {pair['text1'][:50]}..., Text2: {pair['text2'][:50]}..."
            )

