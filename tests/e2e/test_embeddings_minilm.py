from __future__ import annotations

import math
import subprocess

from .helpers import build_cli_cmd, extract_json_from_stdout


def test_minilm_backend_stdin() -> None:
    cmd = build_cli_cmd(
        "analyze",
        "--stdin",
        "--stdout",
        "--enable-embeddings",
        "--embeddings-backend",
        "miniLM",
    )
    proc = subprocess.run(cmd, input="Hello world", text=True, capture_output=True)

    assert proc.returncode == 0, (
        f"CLI failed: {proc.returncode}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
    )
    data = extract_json_from_stdout(proc.stdout)
    emb = next(a for a in data["analyzers"] if a["name"] == "embeddings")
    v = emb["results"]["vector"]
    assert isinstance(v, list) and len(v) > 0
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-3
    # Optional provenance checks if present
    results = emb["results"]
    if "backend" in results:
        assert results["backend"] in {"miniLM", "bge-m3", "openai-te3"}
    if "dim" in results:
        assert isinstance(results["dim"], int) and results["dim"] > 0
        assert len(v) == results["dim"]
    if "normalized" in results:
        assert bool(results["normalized"]) is True


