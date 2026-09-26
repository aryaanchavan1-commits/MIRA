"""Baseline B: hybrid Vector + Graph RAG (spec §28).

Vector seeds → one-hop graph expansion → degree-weighted merge.
"""
from __future__ import annotations

import time
from typing import Dict

import numpy as np

from baselines.vector_rag import BaselineResult
from core.memory import MemoryFrame
from storage.graph_store import GraphStore
from storage.vector_store import VectorStore


class GraphRAG:
    system = "graph_rag"

    def __init__(self, frame: MemoryFrame, vector_store: VectorStore,
                 graph_store: GraphStore):
        self.frame = frame
        self.vs = vector_store
        self.gs = graph_store
        self._index = {n.id: n for n in frame.nodes.values()}
        self._degree = dict(graph_store.g.degree())

    def retrieve(self, query_vec: np.ndarray, k: int = 8,
                 expansion: int = 3) -> BaselineResult:
        t0 = time.perf_counter()
        scores: Dict[str, float] = {}
        # vector component
        for score, meta in self.vs.search(query_vec, k=k):
            nid = meta.get("node_id")
            if nid in self._index:
                scores[nid] = scores.get(nid, 0.0) + float(score)
        # graph expansion component: 1 hop from vector seeds, weighted by degree
        seeds = [nid for nid, _ in sorted(scores.items(),
                                          key=lambda kv: (-kv[1], kv[0]))[:3]]
        for seed in seeds:
            for nid in sorted(self.gs.neighborhood(seed, radius=1)):
                if nid in self._index and nid not in scores:
                    scores[nid] = 0.5 * min(1.0, self._degree.get(nid, 0) / 5.0)
        top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        res = BaselineResult(query="", system=self.system)
        for nid, s in top:
            res.items.append(self._index[nid])
            res.scores.append(float(s))
        res.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return res
