"""Spreading activation over the memory graph.

Bio-inspired retrieval mechanism (Collins & Loftus 1975; Anderson's ACT-R):
memories are nodes carrying activation energy; seed nodes inject energy;
energy spreads along edges, decaying with distance and attenuating per hop.
This is a **mechanism inspired by neural-style activation dynamics** — it is
NOT a simulation of a biological brain (spec §49 scientific honesty).

Implementation: dense numpy propagation on the graph's adjacency matrix.
For research scale (10^3–10^5 nodes) a dense hop matrix multiply per
iteration is the simplest correct thing.

ponytail: dense O(n^2) propagation per hop — fine at research scale; swap
in scipy.sparse csr @ vec (same code shape) if node counts exceed ~1e5.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("mira.activation")


class SpreadingActivation:
    """Activation dynamics over the memory graph.

    seed activation A0(node)  = w_sem * semantic_sim + w_imp * importance
    spread: A_{t+1} = decay*A_t + fan * (W_norm^T @ A_t)  per hop
    fan-out normalization = lesioned synapses don't over-fire when one node
    connects to everything (mimics synaptic scaling).
    """

    def __init__(self, frame, config: Optional[Dict] = None):
        self.frame = frame
        cfg = (config or {}).get("activation", {}) or {}
        self.hops = int(cfg.get("hops", 2))
        self.decay = float(cfg.get("decay", 0.5))          # retention per hop
        self.fan_norm = bool(cfg.get("fanout_normalize", True))
        self.w_sem = float(cfg.get("seed_semantic_weight", 1.0))
        self.w_imp = float(cfg.get("seed_importance_weight", 0.3))
        self.edge_weight = float(cfg.get("edge_weight_scale", 1.0))
        self._mat = None            # cached transposed normalized adjacency
        self._ids: List[str] = []
        self._idx: Dict[str, int] = {}

    # ------------------------------------------------------------------
    def _build(self) -> np.ndarray:
        """Column-normalized adjacency W[i, j] = weight of j -> i."""
        if self._mat is not None:
            return self._mat
        ids = list(self.frame.nodes.keys())
        self._ids = ids
        self._idx = {nid: k for k, nid in enumerate(ids)}
        n = len(ids)
        W = np.zeros((n, n), dtype=np.float32)
        for e in self.frame.edges:
            i, j = self._idx.get(e.source_id), self._idx.get(e.target_id)
            if i is None or j is None or i == j:
                continue
            w = max(0.01, float(e.weight)) * self.edge_weight
            W[i, j] = max(W[i, j], w)   # undirected influence, max of directions
            W[j, i] = max(W[j, i], w)
        if self.fan_norm:
            col = W.sum(axis=0, keepdims=True)
            col[col == 0] = 1.0
            W = W / col
        self._mat = W
        return W

    def invalidate(self) -> None:
        """Call after ingestion/graph changes."""
        self._mat = None

    # ------------------------------------------------------------------
    def activate(self, seed_scores: Dict[str, float]) -> List[Tuple[str, float]]:
        """Propagate seed activation; returns [(node_id, activation)] sorted,
        filtered to energy > 1e-4."""
        W = self._build()
        if not self._ids or not seed_scores:
            return []
        a = np.zeros(len(self._ids), dtype=np.float32)
        for nid, s in seed_scores.items():
            k = self._idx.get(nid)
            if k is not None:
                a[k] = max(a[k], float(s))   # multiple seeds: max, not sum
        if not a.any():
            return []
        for _ in range(self.hops):
            spread = W.T @ a   # fan-out normalization is baked into W at build
            a = np.maximum(a * self.decay, spread * (1.0 - self.decay))
        a = np.clip(a, 0.0, 1.0)
        out = [(self._ids[k], float(a[k])) for k in np.nonzero(a > 1e-4)[0]]
        out.sort(key=lambda t: -t[1])
        return out
