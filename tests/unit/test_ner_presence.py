import asyncio

from rookeen.analyzers.ner import NERAnalyzer
from rookeen.language import get_spacy_model


def test_ner_supported_flag(fixture_texts):
    nlp = get_spacy_model("en", auto_download=True)
    doc = nlp(fixture_texts["en_medium"])
    result = asyncio.run(NERAnalyzer().analyze(doc, "en"))
    assert "supported" in result.results
