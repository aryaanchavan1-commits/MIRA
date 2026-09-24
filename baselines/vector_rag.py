"""Baseline A: naive Vector RAG (spec §28).

Same frame + stores as MIRA, but scoring = cosine similarity only. No rings,
no sectors, no graph, no paths. This is the control MIRA must beat.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from core.memory import MemoryFrame, MemoryNode
from storage.vector_store import VectorStore


@dataclass
class BaselineResult:
    query: str
    items: List[MemoryNode] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    latency_ms: float = 0.0
    system: str = "vector_rag"


class VectorRAG:
    system = "vector_rag"

    def __init__(self, frame: MemoryFrame, vector_store: VectorStore):
        self.frame = frame
        self.vs = vector_store
        self._index = {n.id: n for n in frame.nodes.values()}

    def retrieve(self, query_vec: np.ndarray, k: int = 8) -> BaselineResult:
        t0 = time.perf_counter()
        hits = self.vs.search(query_vec, k=k)
        res = BaselineResult(query="")
        for score, meta in hits:
            nid = meta.get("node_id")
            if nid in self._index:
                res.items.append(self._index[nid])
                res.scores.append(float(score))
        res.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return res
