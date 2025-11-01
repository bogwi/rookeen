import asyncio

import spacy

from rookeen.analyzers.lexical_stats import LexicalStatsAnalyzer


def test_lexical_stats_expected_fields(fixture_texts):
    analyzer = LexicalStatsAnalyzer()
    nlp = spacy.blank("en")
    # Ensure sentence boundaries are available for doc.sents
    nlp.add_pipe("sentencizer")
    doc = nlp(fixture_texts["en_medium"])
    result = asyncio.run(analyzer.analyze(doc, "en"))

    for key in (
        "total_tokens",
        "unique_lemmas",
        "sentences",
        "avg_token_length",
        "avg_sentence_length_tokens",
        "type_token_ratio",
        "top_lemmas",
    ):
        assert key in result.results, f"missing {key}"

