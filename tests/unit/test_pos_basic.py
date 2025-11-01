import asyncio

from rookeen.analyzers.pos import POSAnalyzer
from rookeen.language import get_spacy_model


def test_pos_upos_counts_present(fixture_texts):
    nlp = get_spacy_model("en", auto_download=True)
    doc = nlp(fixture_texts["en_short"])
    result = asyncio.run(POSAnalyzer().analyze(doc, "en"))
    assert "upos_counts" in result.results
    assert isinstance(result.results["upos_counts"], dict)
