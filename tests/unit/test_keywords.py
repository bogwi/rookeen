import asyncio

import spacy

from rookeen.analyzers.keywords import KeywordAnalyzer


def test_keywords_returns_top_list(fixture_texts):
    analyzer = KeywordAnalyzer()
    nlp = spacy.blank("en")
    doc = nlp(fixture_texts["en_short"])
    result = asyncio.run(analyzer.analyze(doc, "en"))
    assert result.name == "keywords"
    assert "keywords" in result.results
    assert isinstance(result.results["keywords"], list)
