"""Spreading activation over the memory graph.

Bio-NN-inspired retrieval mechanism (Collins & Loftus 1975; Anderson's ACT-R):
memories are nodes carrying activation energy; seed nodes inject energy and
activation spreads along graph edges.
The default implementation is the existing continuous mode for compatibility.
The optional ``lif_like`` mode is a bounded, deterministic LIF-inspired
quantization/propagation slice; it is not a claim about biological systems.

The continuous mode keeps dense NumPy propagation for compatibility.  The
``lif_like`` path uses only one-dimensional state arrays and sparse adjacency
lists, so it never allocates an N x N matrix.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("mira.activation")


class SpreadingActivation:
    """Activation dynamics over the memory graph.

    ``continuous`` is the original dense propagation path.  ``lif_like`` is a
    bounded LIF-inspired path: each query starts with fresh membrane state,
    integrates seed/propagated current for a small number of ticks, spikes at
    a threshold, resets, and holds a refractory state.  Its graph representation
    is a bounded sparse neighbor list rather than a dense matrix.

    The output remains ``[(node_id, activation), ...]``.  In LIF mode only
    nodes that emitted at least one spike are returned; ``last_trace`` contains
    the per-query audit events.
    """

    # ponytail: fixed caps are a deliberate safety ceiling; raise only with profiling.
    _MAX_LIF_TICKS = 32
    _MAX_LIF_REFRACTORY = 32
    _MAX_LIF_NEIGHBORS = 32
    _MAX_TRACE_EVENTS = 10000

    def __init__(self, frame, config: Optional[Dict] = None,
                 mode: Optional[str] = None):
        self.frame = frame
        cfg = (config or {}).get("activation", {}) or {}
        raw_mode = mode if mode is not None else cfg.get("mode", "continuous")
        self.mode = str(raw_mode).strip().lower().replace("-", "_")
        if self.mode == "lif":
            self.mode = "lif_like"
        if self.mode not in {"continuous", "lif_like"}:
            raise ValueError("activation.mode must be 'continuous' or 'lif_like'")

        # Continuous-mode settings retain their historical names and behavior.
        self.hops = int(cfg.get("hops", 2))
        self.decay = self._finite(cfg.get("decay", 0.5), 0.5)
        self.fan_norm = bool(cfg.get("fanout_normalize", True))
        self.w_sem = self._finite(cfg.get("seed_semantic_weight", 1.0), 1.0)
        self.w_imp = self._finite(cfg.get("seed_importance_weight", 0.3), 0.3)
        self.edge_weight = self._finite(cfg.get("edge_weight_scale", 1.0), 1.0)

        # LIF settings are bounded even when a config file is edited by hand.
        self.lif_ticks = self._bounded_int(
            cfg, ("ticks", "simulation_ticks"), 3, 1, self._MAX_LIF_TICKS)
        self.lif_leak = self._bounded_float(
            cfg, ("leak", "leak_rate", "decay"), 0.5, 0.0, 1.0)
        self.lif_threshold = self._bounded_float(
            cfg, ("threshold",), 0.8, 0.0, 1.0)
        reset = self._bounded_float(
            cfg, ("reset_voltage", "reset"), 0.0, 0.0, self.lif_threshold)
        self.lif_reset = min(reset, self.lif_threshold)
        self.lif_refractory_ticks = self._bounded_int(
            cfg, ("refractory_ticks", "refractory"), 1, 0,
            self._MAX_LIF_REFRACTORY)
        self.lif_top_k_neighbors = self._bounded_int(
            cfg, ("top_k_neighbors", "neighbor_top_k", "max_neighbors"), 4, 0,
            self._MAX_LIF_NEIGHBORS)
        self.trace_max_events = self._bounded_int(
            cfg, ("max_trace_events", "trace_max_events"), 4096, 0,
            self._MAX_TRACE_EVENTS)
        self.lif_edge_weight_scale = self._bounded_float(
            cfg, ("edge_weight_scale",), 1.0, 0.0, 1.0)
        self.lif_w_sem = max(0.0, self._finite(
            cfg.get("seed_semantic_weight", 1.0), 1.0))
        self.lif_w_imp = max(0.0, self._finite(
            cfg.get("seed_importance_weight", 0.3), 0.3))

        self._mat: Optional[np.ndarray] = None  # continuous cache only
        self._neighbors: Optional[List[List[Tuple[int, float]]]] = None
        self._ids: List[str] = []
        self._idx: Dict[str, int] = {}
        self.last_trace: List[Dict[str, Any]] = []

    @staticmethod
    def _finite(value: Any, default: float) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
        return result if np.isfinite(result) else default

    @classmethod
    def _bounded_float(cls, cfg: Dict[str, Any], keys: Tuple[str, ...],
                      default: float, lo: float, hi: float) -> float:
        value = default
        for key in keys:
            if key in cfg:
                value = cls._finite(cfg.get(key), default)
                break
        return float(np.clip(value, lo, hi))

    @classmethod
    def _bounded_int(cls, cfg: Dict[str, Any], keys: Tuple[str, ...],
                     default: int, lo: int, hi: int) -> int:
        value = default
        for key in keys:
            if key in cfg:
                value = cls._finite(cfg.get(key), default)
                break
        return int(np.clip(int(value), lo, hi))

    # ------------------------------------------------------------------
    def _build(self) -> np.ndarray:
        """Build the original column-normalized dense adjacency matrix."""
        if self._mat is not None:
            return self._mat
        ids = sorted(self.frame.nodes.keys())
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

    def _build_sparse(self) -> List[List[Tuple[int, float]]]:
        """Build bounded top-neighbor lists without an N x N allocation."""
        if self._neighbors is not None:
            return self._neighbors
        ids = sorted(self.frame.nodes.keys(), key=lambda value: str(value))
        self._ids = ids
        self._idx = {nid: k for k, nid in enumerate(ids)}
        adjacency: List[Dict[int, float]] = [{} for _ in ids]
        for e in self.frame.edges:
            i = self._idx.get(e.source_id)
            j = self._idx.get(e.target_id)
            if i is None or j is None or i == j:
                continue
            try:
                weight = float(e.weight)
            except (TypeError, ValueError, OverflowError):
                continue
            if not np.isfinite(weight) or weight <= 0.0:
                continue
            weight = min(1.0, max(0.01, weight) * self.lif_edge_weight_scale)
            if weight <= 0.0:
                continue
            if weight > adjacency[i].get(j, 0.0):
                adjacency[i][j] = weight
                adjacency[j][i] = weight
        self._neighbors = []
        for k in range(len(ids)):
            ranked = sorted(
                adjacency[k].items(),
                key=lambda item: (-item[1], str(ids[item[0]])),
            )
            self._neighbors.append([
                (int(target), float(weight))
                for target, weight in ranked[:self.lif_top_k_neighbors]
            ])
        return self._neighbors

    @property
    def trace(self) -> List[Dict[str, Any]]:
        """Alias for the most recent per-query trace."""
        return self.last_trace

    def invalidate(self) -> None:
        """Call after ingestion/graph changes."""
        self._mat = None
        self._neighbors = None
        self._ids = []
        self._idx = {}
        self.last_trace = []

    # ------------------------------------------------------------------
    def activate(self, seed_scores: Dict[str, float],
                 trace: Optional[List[Dict[str, Any]]] = None) -> List[Tuple[str, float]]:
        """Run the configured mode and return deterministic ranked output.

        Continuous mode preserves the original dense implementation.  LIF mode
        emits only thresholded spikes and leaves an auditable event list in
        ``last_trace`` with ``tick``, ``node``, ``pre_voltage``,
        ``post_voltage``, ``threshold``, and ``spike`` fields.
        """
        local_trace = [] if trace is None else trace
        if self.mode == "lif_like":
            output = self._activate_lif(seed_scores, local_trace)
        else:
            local_trace.clear()
            output = self._activate_continuous(seed_scores)
        self.last_trace = list(local_trace)
        return output

    def _activate_continuous(self, seed_scores: Dict[str, float]) -> List[Tuple[str, float]]:
        W = self._build()
        if not self._ids or not seed_scores:
            return []
        a = np.zeros(len(self._ids), dtype=np.float32)
        for nid, s in sorted(seed_scores.items(), key=lambda item: str(item[0])):
            k = self._idx.get(nid)
            if k is not None:
                node = self.frame.nodes.get(nid)
                importance = float(getattr(node, "importance", 0.0)) if node else 0.0
                seeded = self.w_sem * float(s) + self.w_imp * importance
                a[k] = max(a[k], seeded)   # multiple seeds: max, not sum
        if not a.any():
            return []
        for _ in range(self.hops):
            spread = W.T @ a   # fan-out normalization is baked into W at build
            a = np.maximum(a * self.decay, spread * (1.0 - self.decay))
        a = np.clip(a, 0.0, 1.0)
        out = [(self._ids[k], float(a[k])) for k in np.nonzero(a > 1e-4)[0]]
        out.sort(key=lambda t: (-t[1], t[0]))
        return out

    def _activate_lif(self, seed_scores: Dict[str, float],
                      trace: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        neighbors = self._build_sparse()
        if not self._ids:
            return []

        n = len(self._ids)
        current = np.zeros(n, dtype=np.float64)
        tracked = np.zeros(n, dtype=bool)
        for nid, raw_score in sorted((seed_scores or {}).items(),
                                     key=lambda item: str(item[0])):
            k = self._idx.get(nid)
            if k is None:
                continue
            try:
                score = float(np.clip(float(raw_score), 0.0, 1.0))
            except (TypeError, ValueError, OverflowError):
                continue
            node = self.frame.nodes.get(nid)
            try:
                importance = float(getattr(node, "importance", 0.0))
            except (TypeError, ValueError, OverflowError):
                importance = 0.0
            if not np.isfinite(importance):
                importance = 0.0
            importance = float(np.clip(importance, 0.0, 1.0))
            injected = float(np.clip(
                self.lif_w_sem * score + self.lif_w_imp * importance, 0.0, 1.0))
            if injected > 0.0:
                current[k] = max(float(current[k]), injected)
                tracked[k] = True
        if not tracked.any():
            return []

        voltage = np.zeros(n, dtype=np.float64)
        refractory = np.zeros(n, dtype=np.int32)
        peak = np.zeros(n, dtype=np.float64)
        spiked = np.zeros(n, dtype=bool)

        for tick in range(self.lif_ticks):
            # Capture this tick's frontier before adding the next tick's
            # propagation, keeping the update synchronous and deterministic.
            touched = np.flatnonzero(tracked).tolist()
            next_current = np.zeros(n, dtype=np.float64)
            for k in touched:
                pre_voltage = float(np.clip(voltage[k], 0.0, 1.0))
                spike = False
                if refractory[k] > 0:
                    post_voltage = self.lif_reset
                    refractory[k] = max(0, int(refractory[k]) - 1)
                else:
                    integrated = float(np.clip(
                        pre_voltage * (1.0 - self.lif_leak) + current[k],
                        0.0, 1.0,
                    ))
                    spike = integrated > 0.0 and integrated >= self.lif_threshold
                    if spike:
                        post_voltage = self.lif_reset
                        refractory[k] = self.lif_refractory_ticks
                        peak[k] = max(float(peak[k]), integrated)
                        spiked[k] = True
                        for target, weight in neighbors[k]:
                            propagated = min(1.0, float(next_current[target])
                                            + integrated * weight)
                            next_current[target] = propagated
                            tracked[target] = True
                    else:
                        post_voltage = integrated
                voltage[k] = post_voltage
                self._record_trace(trace, tick, self._ids[k], pre_voltage,
                                   post_voltage, spike)
            current = next_current

        out = [
            (self._ids[int(k)], float(np.clip(peak[k], 0.0, 1.0)))
            for k in np.flatnonzero(spiked)
        ]
        out.sort(key=lambda item: (-item[1], str(item[0])))
        return out

    def _record_trace(self, trace: List[Dict[str, Any]], tick: int,
                      node: str, pre_voltage: float, post_voltage: float,
                      spike: bool) -> None:
        if len(trace) >= self.trace_max_events:
            return
        trace.append({
            "tick": int(tick),
            "node": node,
            "pre_voltage": float(pre_voltage),
            "post_voltage": float(post_voltage),
            "threshold": float(self.lif_threshold),
            "spike": bool(spike),
        })
