import asyncio
from pathlib import Path

from rookeen.analyzers.ner import NERAnalyzer
from rookeen.analyzers.pos import POSAnalyzer
from rookeen.language import get_spacy_model


def test_pos_upos_counts_present_local_fixture():
    nlp = get_spacy_model("en", auto_download=True)
    text = (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "test_data"
        / "en_short.txt"
    ).read_text(encoding="utf-8")
    doc = nlp(text)
    result = asyncio.run(POSAnalyzer().analyze(doc, "en"))
    assert "upos_counts" in result.results
    assert isinstance(result.results["upos_counts"], dict)


def test_ner_supported_flag_local_fixture():
    nlp = get_spacy_model("en", auto_download=True)
    text = (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "test_data"
        / "en_medium.txt"
    ).read_text(encoding="utf-8")
    doc = nlp(text)
    result = asyncio.run(NERAnalyzer(nlp=nlp).analyze(doc, "en"))
    assert "supported" in result.results

