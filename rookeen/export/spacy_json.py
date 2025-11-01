from __future__ import annotations

from typing import Any

from spacy.tokens import Doc


def doc_to_spacy_json(doc: Doc) -> dict[str, Any]:
    """Convert a spaCy Doc into a JSON-serializable structure.

    The structure includes document text, token-level attributes, and entities.
    This mirrors common expectations for downstream analytics and is stable across languages.
    """
    tokens: list[dict[str, Any]] = []
    for i, tok in enumerate(doc):
        tokens.append(
            {
                "id": i,
                "text": tok.text,
                "lemma": tok.lemma_,
                "pos": tok.pos_,
                "tag": tok.tag_,
                "dep": tok.dep_,
                "head": tok.head.i,
                "ent_type": tok.ent_type_,
                "whitespace": tok.whitespace_,
                "idx": tok.idx,
            }
        )

    ents: list[dict[str, Any]] = [
        {"start": ent.start, "end": ent.end, "label": ent.label_} for ent in doc.ents
    ]

    return {
        "text": doc.text,
        "tokens": tokens,
        "ents": ents,
    }
