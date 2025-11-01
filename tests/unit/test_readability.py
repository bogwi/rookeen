import asyncio

import spacy

from rookeen.analyzers.readability import ReadabilityAnalyzer


def test_readability_has_expected_fields(fixture_texts):
    analyzer = ReadabilityAnalyzer()
    nlp = spacy.blank("en")
    doc = nlp(fixture_texts["en_medium"])
    result = asyncio.run(analyzer.analyze(doc, "en"))
    for key in (
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "smog_index",
        "automated_readability_index",
        "coleman_liau_index",
        "linsear_write_formula",
        "dale_chall_readability_score",
        "difficult_words",
        "text_standard",
    ):
        assert key in result.results, f"missing {key}"
