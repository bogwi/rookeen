from __future__ import annotations

from pathlib import Path

import pytest

from .helpers import (
    assert_analyzer_has_model_metadata,
    assert_json_key,
    assert_lang,
    build_cli_cmd,
    find_analyzer,
    read_json,
    run_cli,
    unique_output_base,
)

URL = "https://de.wikipedia.org/w/index.php?title=K%C3%BCnstliche_Intelligenz&oldid=247413934"

# Networked Wikipedia; exclude from default runs
pytestmark = [pytest.mark.external]


def test_de_wikipedia_oldid() -> None:
    out_base: Path = unique_output_base("de-wikipedia-org")
    out_json = str(out_base) + ".json"

    args = build_cli_cmd(
        "analyze",
        URL,
        "--format",
        "json",
        "--models-auto-download",
        "--lang",
        "de",
        "--robots",
        "ignore",
        "-o",
        out_json,
    )
    proc = run_cli(args)
    assert proc.returncode == 0, (
        f"CLI failed: {proc.returncode}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
    )

    payload = read_json(out_json)
    assert_lang(payload, "de")
    conf = float(assert_json_key(payload, "language.confidence"))
    assert conf >= 0.6

    pos = find_analyzer(payload, "pos")
    upos = pos.get("results", {}).get("upos_counts", {})
    assert isinstance(upos, dict)
    assert len(upos.keys()) >= 4

    ner = find_analyzer(payload, "ner")
    ner_results = ner.get("results", {})
    assert "supported" in ner_results
    if ner_results.get("supported"):
        total = int(ner_results.get("total_entities", 0))
        assert total >= 0
        assert isinstance(ner_results.get("counts_by_label", {}), dict)

    lex = find_analyzer(payload, "lexical_stats")
    total_tokens = int(lex.get("results", {}).get("total_tokens", 0))
    assert total_tokens > 200

    for name in ("pos", "ner", "lexical_stats", "readability", "keywords"):
        assert_analyzer_has_model_metadata(payload, name)
