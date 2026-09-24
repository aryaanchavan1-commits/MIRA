"""Embedding lineage guard.

Records which embedding model produced the stored vectors. When the active
backend differs (e.g. hashing fallback → MiniLM after a model download), all
node embeddings are re-computed so the vector index never mixes incompatible
semantics. Prevents silent retrieval poisoning.
"""
from __future__ import annotations

import json
import logging
import os

from config.auto_config import DATA_DIR

logger = logging.getLogger("mira.embeddings_lineage")

_STATE_PATH = os.path.join(DATA_DIR, "embedding_state.json")


def recorded_model() -> str:
    try:
        with open(_STATE_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh).get("model", "")
    except Exception:
        return ""


def record_model(name: str) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(_STATE_PATH, "w", encoding="utf-8") as fh:
        json.dump({"model": name}, fh)


def ensure_consistent(workspace, backend_info_name: str) -> int:
    """Re-embed everything when the backend changed since the last run.

    Returns the number of nodes re-embedded (0 = lineage matches).
    """
    prev = recorded_model()
    if prev == backend_info_name:
        return 0
    # Never migrate good vectors to the hashing fallback. The fallback is a
    # transient degradation (OOM, AV lock); when the real model loads again
    # the lineage matches the record and nothing needs re-embedding.
    if backend_info_name.startswith("hashing:") and prev and not prev.startswith("hashing:"):
        logger.warning("embedding backend degraded to hashing fallback — "
                       "keeping %r vectors and lineage (no re-embedding)", prev)
        return 0
    if prev:
        logger.warning("embedding backend changed: %r -> %r — re-embedding all nodes",
                       prev, backend_info_name)
    n = workspace.reembed_missing()
    if n == 0:  # nothing missing means embeddings exist but from the old model
        for node in workspace.frame.nodes.values():
            node.embedding = None
        n = workspace.reembed_missing()
    record_model(backend_info_name)
    return n
