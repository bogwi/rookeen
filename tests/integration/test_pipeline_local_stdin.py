import json
import subprocess
from pathlib import Path
from typing import Any

import pytest


def run_cli_with_stdin(text: str) -> Any:
    proc = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "rookeen.cli",
            "analyze",
            "--stdin",
            "--lang",
            "en",
            "--stdout",
        ],
        input=text.encode("utf-8"),
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8")
    return json.loads(proc.stdout.decode("utf-8"))


@pytest.mark.smoke
def test_pipeline_local_stdin_schema(fixture_texts):
    payload = run_cli_with_stdin(fixture_texts["en_short"])
    for k in ("tool", "version", "source", "language", "content", "analyzers", "timing"):
        assert k in payload
    assert payload["source"]["type"] == "stdin"


def test_schema_validation(fixture_texts):
    from jsonschema import validate

    payload = run_cli_with_stdin(fixture_texts["en_medium"])
    schema = json.loads(Path("schemas/rookeen_v1.json").read_text(encoding="utf-8"))
    validate(instance=payload, schema=schema)

