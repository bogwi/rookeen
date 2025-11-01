from __future__ import annotations

from typing import Any

from spacy.tokens import Doc

# Mapping from spaCy dependency labels to UD dependency labels
# This is a simplified mapping for common cases
_SPACY_TO_UD_DEPREL: dict[str, str] = {
    "ROOT": "root",
    "prep": "case",
    "pobj": "obl",  # Note: may need context-aware mapping (obl vs nmod)
    "dobj": "obj",
    "iobj": "iobj",
    "nsubj": "nsubj",
    "nsubjpass": "nsubj:pass",
    "csubj": "csubj",
    "csubjpass": "csubj:pass",
    "aux": "aux",
    "auxpass": "aux:pass",
    "cop": "cop",
    "conj": "conj",
    "cc": "cc",
    "punct": "punct",
    "mark": "mark",
    "advmod": "advmod",
    "amod": "amod",
    "det": "det",
    "poss": "nmod:poss",
    "compound": "compound",
    "appos": "appos",
    "nummod": "nummod",
    "acl": "acl",
    "xcomp": "xcomp",
    "ccomp": "ccomp",
    "advcl": "advcl",
    "relcl": "acl:relcl",
    "intj": "discourse",  # Interjections in UD are typically discourse
}


def _get_morphological_features(token: Any, ud_pos: str | None = None) -> str:
    """
    Extract basic morphological features for UD FEATS column.
    Conservative: prefer spaCy's morph when available; otherwise emit "_".
    This avoids overcommitting incorrect features.
    """
    try:
        morph_str = str(token.morph)
        if morph_str and morph_str != "{}":
            feats: list[str] = []
            for feat_pair in morph_str.strip("{}").split("|"):
                if "=" in feat_pair:
                    key, value = feat_pair.split("=", 1)
                    if key not in ["PunctType", "PunctSide"]:
                        feats.append(f"{key}={value}")
            return "|".join(feats) if feats else "_"
    except Exception:
        pass
    return "_"


def _get_ud_deprel(token: Any, non_space_tokens: list[Any]) -> str:
    """
    Get UD-compliant dependency relation with context awareness.
    For example, obl vs nmod depends on the head's POS and context.
    """
    spacy_dep = token.dep_
    if spacy_dep in _SPACY_TO_UD_DEPREL:
        ud_dep = _SPACY_TO_UD_DEPREL[spacy_dep]
    else:
        ud_dep = spacy_dep.lower()

    # Context-aware corrections for prepositional phrases
    if ud_dep == "obl" and token.head:
        # Check if this is part of a PP modifying a noun
        # If obl's head is a preposition (ADP), check the preposition's head
        if token.head.pos_ == "ADP" and token.head.head:
            grandparent = token.head.head
            if grandparent.pos_ in ["NOUN", "PROPN", "PRON", "ADJ", "NUM"]:
                ud_dep = "nmod"

    # General rule: if obl modifies a noun directly, it should be nmod
    elif ud_dep == "obl" and token.head:
        head_pos = token.head.pos_
        if head_pos in ["NOUN", "PROPN", "PRON", "ADJ", "NUM"]:
            ud_dep = "nmod"

    return ud_dep


def _get_ud_pos_and_tag(token: Any) -> tuple[str, str]:
    """
    Get UD-compliant POS and XPOS tags.
    Override spaCy tags for better UD compliance.
    """
    # Default conservatively to spaCy's tags
    return token.pos_, token.tag_


def _escape_conllu_field(text: str) -> str:
    """
    Escape special characters in CoNLL-U fields.
    According to CoNLL-U spec, tabs and newlines should be avoided in fields.
    Also handle empty/whitespace-only strings.
    """
    if not text or not text.strip():
        return "_"
    # Replace tabs with spaces, as tabs are field separators
    # Strip leading/trailing whitespace
    return text.strip().replace("\t", " ").replace("\n", " ").replace("\r", " ")


def doc_to_conllu(doc: Doc) -> str:
    """
    Convert a spaCy Doc to CoNLL-U format.

    Args:
        doc: spaCy Doc object to convert

    Returns:
        String containing CoNLL-U formatted text
    """
    lines = []

    # Header comments warning about non-compliance of the basic exporter
    lines.append("# NOTE: Heuristic spaCy-based CoNLL-U serializer (basic engine)")
    lines.append("# This output is not guaranteed to be UD-valid for complex texts.")
    lines.append(
        "# Prefer the UD-native engine via --conllu-engine stanza for standards-compliant output."
    )

    # Handle case where document has no sentence segmentation
    sentences = list(doc.sents) if doc.has_annotation("SENT_START") else [doc]

    for sent_id, sent in enumerate(sentences):
        # Add sentence-level comments
        lines.append(f"# sent_id = {sent_id}")
        lines.append(f"# text = {_escape_conllu_field(sent.text)}")

        # Filter out space tokens using spaCy's is_space flag
        non_space_tokens = [t for t in sent if not t.is_space]

        # Build token data with corrected dependencies
        token_data = []
        for i, token in enumerate(non_space_tokens, start=1):
            # Get UD-compliant POS and tag
            ud_pos, ud_xpos = _get_ud_pos_and_tag(token)

            # Calculate head index - will be corrected below for PP structures
            if token.dep_ == "ROOT":
                head_idx = 0
            elif token.head and token.head in non_space_tokens:
                head_idx = non_space_tokens.index(token.head) + 1
            else:
                head_idx = 0

            # Get UD-compliant deprel
            ud_deprel = _get_ud_deprel(token, non_space_tokens)

            token_data.append(
                {
                    "index": i,
                    "token": token,
                    "head_idx": head_idx,
                    "ud_pos": ud_pos,
                    "ud_xpos": ud_xpos,
                    "ud_deprel": ud_deprel,
                }
            )

        # Fix prepositional phrase dependencies
        # When we have: noun <- prep <- ADP, ADP <- pobj <- noun
        # Change to: noun <- nmod <- noun, noun <- case <- ADP
        for data in token_data:
            token = data["token"]
            # Check for pobj that depends on ADP (preposition)
            if token.dep_ == "pobj" and token.head and token.head.pos_ == "ADP" and token.head.head:
                # Find the preposition in token_data
                prep_data = next((d for d in token_data if d["token"] == token.head), None)
                if prep_data:
                    # Change pobj to nmod, pointing to the preposition's head (the noun)
                    grandparent = token.head.head
                    if grandparent in non_space_tokens:
                        data["head_idx"] = non_space_tokens.index(grandparent) + 1
                        data["ud_deprel"] = "nmod"

                        # Change preposition deprel to case, pointing to the noun
                        prep_data["head_idx"] = data["index"]
                        prep_data["ud_deprel"] = "case"

        # Basic copular construction normalization (spaCy attr -> UD root+cop)
        # If the sentence has a ROOT that is an auxiliary/verb (e.g., "is")
        # and an 'attr' dependent predicate noun/adjective, promote the predicate
        # to root and attach the copula with deprel 'cop'. Reattach subjects and
        # other dependents of the copula to the promoted predicate.
        try:
            root_data = next((d for d in token_data if d["token"].dep_ == "ROOT"), None)
            attr_pred = None
            if root_data and root_data["token"].pos_ in ("AUX", "VERB"):
                for d in token_data:
                    if d["token"].dep_ == "attr" and d["token"].head == root_data["token"]:
                        attr_pred = d
                        break
            if root_data and attr_pred:
                original_root_idx = root_data["index"]
                promoted_idx = attr_pred["index"]
                # Promote predicate to root
                attr_pred["head_idx"] = 0
                attr_pred["ud_deprel"] = "root"
                # Attach copula to predicate
                root_data["head_idx"] = promoted_idx
                root_data["ud_deprel"] = "cop"
                # Reattach dependents that previously pointed to the copula root
                for d in token_data:
                    if d is root_data:
                        continue
                    if d["head_idx"] == original_root_idx:
                        d["head_idx"] = promoted_idx
        except Exception:
            # Best-effort normalization; ignore if anything unexpected occurs
            pass

        # Build the CoNLL-U lines
        for data in token_data:
            token = data["token"]

            # Build MISC field with SpaceAfter information
            misc_parts = []
            if not token.whitespace_:
                misc_parts.append("SpaceAfter=No")
            misc = "|".join(misc_parts) if misc_parts else "_"

            # Build CoNLL-U fields (10 columns)
            fields = [
                str(data["index"]),  # ID
                _escape_conllu_field(token.text),  # FORM
                _escape_conllu_field(token.lemma_),  # LEMMA
                data["ud_pos"],  # UPOS
                data["ud_xpos"],  # XPOS
                _get_morphological_features(token, data["ud_pos"]),  # FEATS
                str(data["head_idx"]),  # HEAD
                data["ud_deprel"],  # DEPREL
                "_",  # DEPS (enhanced, empty)
                misc,  # MISC
            ]

            lines.append("\t".join(fields))

        # Add blank line between sentences
        lines.append("")

    # Join with newlines, ensuring final empty line
    result = "\n".join(lines)
    if not result.endswith("\n\n"):
        result += "\n"
    return result
