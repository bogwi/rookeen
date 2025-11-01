from __future__ import annotations

import tempfile

import pytest

from .helpers import (
    assert_analyzer_has_model_metadata,
    assert_json_key,
    build_cli_cmd,
    find_analyzer,
    read_json,
    run_cli,
)

URLS = [
    "https://en.wikipedia.org/w/index.php?title=Natural_language_processing&oldid=1201524046",
    "https://de.wikipedia.org/w/index.php?title=K%C3%BCnstliche_Intelligenz&oldid=247413934",
    "https://es.wikipedia.org/w/index.php?title=Aprendizaje_autom%C3%A1tico&oldid=159956240",
    "https://fr.wikipedia.org/w/index.php?title=Traitement_automatique_du_langage_naturel&oldid=214340480",
]

# Mark entire module as external (networked Wikipedia URLs)
pytestmark = [pytest.mark.external]


def test_mixed_batch() -> None:
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
        for url in URLS:
            f.write(url + "\n")
        url_list_path = f.name

    args = build_cli_cmd(
        "batch",
        url_list_path,
        "--output-dir",
        "results",
        "--robots",
        "ignore",
        "--format",
        "json",
        "--models-auto-download",
    )
    proc = run_cli(args)
    assert proc.returncode == 0, (
        f"Batch CLI failed: {proc.returncode}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
    )

    # CLI prints one JSON path per processed URL on stdout
    out_paths = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip().endswith(".json")]
    assert len(out_paths) == len(URLS)

    for path in out_paths:
        payload = read_json(path)
        code = payload.get("language", {}).get("code")
        assert code in {"en", "de", "es", "fr"}
        conf = float(assert_json_key(payload, "language.confidence"))
        assert conf >= 0.6

        pos = find_analyzer(payload, "pos")
        upos = pos.get("results", {}).get("upos_counts", {})
        assert isinstance(upos, dict)
        assert len(upos.keys()) >= 4

        # Lightweight NER validation per file
        ner = find_analyzer(payload, "ner")
        ner_results = ner.get("results", {})
        assert "supported" in ner_results
        if ner_results.get("supported"):
            total = int(ner_results.get("total_entities", 0))
            assert total >= 0
            assert isinstance(ner_results.get("counts_by_label", {}), dict)

        for name in ("pos", "ner", "lexical_stats", "readability", "keywords"):
            assert_analyzer_has_model_metadata(payload, name)
