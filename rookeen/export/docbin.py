from __future__ import annotations

from spacy.tokens import Doc, DocBin


def dump_docbin(doc: Doc, path: str) -> None:
    """Serialize a single spaCy Doc into a DocBin file on disk.

    The DocBin stores user data to preserve analyzer-specific extensions when present.
    """
    db = DocBin(store_user_data=True)
    db.add(doc)
    db.to_disk(path)
