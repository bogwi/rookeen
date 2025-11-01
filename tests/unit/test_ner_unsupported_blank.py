import asyncio

import spacy

from rookeen.analyzers.ner import NERAnalyzer


def test_ner_supported_false_with_blank_model(fixture_texts):
    # spaCy blank pipeline has no NER component
    nlp = spacy.blank("en")
    doc = nlp(fixture_texts["en_medium"])
    result = asyncio.run(NERAnalyzer().analyze(doc, "en"))
    assert result.results.get("supported") is False

