from __future__ import annotations

import logging
from typing import Any

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as e:
    raise ImportError(
        "pyarrow is required for Parquet export. Install with 'pip install pyarrow' or 'pip install rookeen[parquet]'."
    ) from e


def analyzers_to_parquet(analyzers: list[dict[str, Any]], path: str) -> None:
    """
    Export a list of analyzer summary dicts to a Parquet file.

    Each analyzer dict should have keys: 'name', 'processing_time', 'confidence', 'results' (a dict), and optionally 'metadata' (a dict with 'model' and 'language').
    Only flat, scalar values from 'results' are exported (int, float, str, bool).
    If present, 'metadata.model' and 'metadata.language' are exported as columns.

    Args:
        analyzers: List of analyzer summary dicts.
        path: Output Parquet file path.
    Raises:
        ValueError: If analyzers is not a list of dicts.
        IOError: If writing to the file fails.
    """
    logger = logging.getLogger("rookeen.export.parquet")
    if not isinstance(analyzers, list):
        raise ValueError("'analyzers' must be a list of dicts.")
    rows: list[dict[str, Any]] = []
    for a in analyzers:
        # if not isinstance(a, dict):
        #     logger.warning("Skipping non-dict analyzer entry: %r", a)
        #     continue
        r = {
            "name": a.get("name"),
            "processing_time": a.get("processing_time"),
            "confidence": a.get("confidence"),
        }
        # Add metadata columns if present
        metadata = a.get("metadata", {})
        if isinstance(metadata, dict):
            r["metadata.model"] = metadata.get("model")
            lang = metadata.get("language")
            # language may be a dict or str
            if isinstance(lang, dict):
                r["metadata.language"] = lang.get("code")
            else:
                r["metadata.language"] = lang
        else:
            r["metadata.model"] = None
            r["metadata.language"] = None
        for k, v in a.get("results", {}).items():
            if isinstance(v, int | float | str | bool):
                r[f"results.{k}"] = v
        rows.append(r)
    if not rows:
        logger.warning("No valid analyzer rows to export.")
    try:
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, path)
        logger.info("Wrote %d analyzer rows to Parquet: %s", len(rows), path)
    except Exception as exc:
        logger.error("Failed to write Parquet: %s", exc)
        raise OSError(f"Failed to write Parquet file: {exc}") from exc
