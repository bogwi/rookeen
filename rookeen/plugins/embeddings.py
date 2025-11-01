from __future__ import annotations

import time
from typing import Any

from rookeen.analyzers.embeddings_backends import get_backend


async def run_embeddings(text: str) -> dict[str, Any]:
    """Legacy helper: delegate to MiniLM backend and return provenance + timing.

    This function intentionally does not return the embedding vector to keep
    payloads small, mirroring prior behavior. It surfaces provenance fields to
    avoid drift with the main analyzer implementation.
    """
    start = time.perf_counter()
    try:
        backend = get_backend("miniLM")
        # Ensure dependencies are available and model is materialized
        backend.load()
        prov = backend.provenance()
        return {
            "supported": True,
            **prov,
            "processing_time": time.perf_counter() - start,
        }
    except Exception as exc:
        return {
            "supported": False,
            "note": f"embeddings backend unavailable: {exc}",
            "processing_time": time.perf_counter() - start,
        }
