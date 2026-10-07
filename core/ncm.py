"""MIRA-NCM: Nested Constellation Memory — optional store-side extensions.

Two mechanisms, both OFF unless config ``ncm.enabled`` is true:

1. Constellations — sparse overlapping soft membership. k-means over node
   embeddings induces clusters; each node joins its top-K clusters with L1-
   normalized similarity weights. The retrieval term for a node is the overlap
   kernel against the query's own membership: sum_c w_q(c) * w_m(c) in [0, 1].
   K=1 degenerates to plain clustering — exactly the H2 ablation axis.

2. Versioning — persistent/versioned memory: every update appends a version
   keyed by stable hash, with immutable history, ``supersedes`` links and
   provenance. Never silently overwrites; "current" is the chain head.

Complexity (from this implementation, spec §40): fit is O(N·d·I) for I k-means
iterations (sklearn); assignment is one O(N·C) matmul against centroids, not
O(N²); per-node scoring is O(K) dict lookups. Budget caps: MAX_CONSTELLATIONS,
CONSTELLATION_TOP_K, MAX_DEPTH bound growth; fit degrades gracefully below
MIN_FIT_NODES embedded nodes.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.memory import MemoryFrame
from core.types import iso_now

# Spec §41 budget caps (overridable in app config under "ncm").
DEFAULTS = {
    "enabled": False,          # OFF by default: classic MIRA unchanged
    "top_k": 3,                # CONSTELLATION_TOP_K (K=1 is the ablation axis)
    "max_constellations": 64,  # MAX_CONSTELLATIONS
    "overlap_floor": 0.0,      # OVERLAP_THRESHOLD: weights below are cut
    "max_depth": 4,            # MAX_DEPTH (version-chain cap)
    "backend": "linear",       # membership kernel: 'linear' | 'born' (H6)
}

META_KEY = "constellations"    # stored in MemoryNode.metadata (persists)
QUERY_KEY = "__query__"
MIN_FIT_NODES = 8              # ponytail: no-op below this; upgrade = minibatch k-means


def ncm_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve NCM settings from the app config (defaults stand otherwise)."""
    out = dict(DEFAULTS)
    ncm = config.get("ncm", {}) if isinstance(config, dict) else {}
    for key in DEFAULTS:
        if key in ncm:
            out[key] = ncm[key]
    return out


class ConstellationIndex:
    """Sparse overlapping soft membership over node embeddings."""

    def __init__(self, top_k: int = 3, max_constellations: int = 64,
                 overlap_floor: float = 0.0):
        self.top_k = max(1, int(top_k))
        self.max_constellations = max(1, int(max_constellations))
        self.overlap_floor = float(overlap_floor)
        self.centroids: Optional[np.ndarray] = None
        self.members: Dict[str, Tuple[str, ...]] = {}     # node -> constellation ids
        self.weight_of: Dict[str, Tuple[float, ...]] = {} # node -> aligned weights
        self._vectors: Dict[str, np.ndarray] = {}         # node -> embedding (fit time)
        self._fit_info = "not fitted"

    # -- fit ------------------------------------------------------------
    def fit(self, frame: MemoryFrame) -> bool:
        """Induce constellations from node embeddings. True if fitted."""
        self.members, self.weight_of, self._vectors = {}, {}, {}
        for n in frame.nodes.values():
            if n.embedding is not None:
                self._vectors[n.id] = np.asarray(n.embedding, dtype=np.float32)
        if len(self._vectors) < MIN_FIT_NODES:
            self.centroids = None
            self._fit_info = f"too few embedded nodes ({len(self._vectors)})"
            return False
        ids = sorted(self._vectors)
        X = np.stack([self._vectors[i] for i in ids])
        k = int(min(self.max_constellations, math.isqrt(len(ids))))
        try:
            from sklearn.cluster import KMeans
            labels = KMeans(n_clusters=k, n_init=4, random_state=0).fit(X).labels_
        except Exception as exc:  # noqa: BLE001 — retrieval must not die on fit
            self.centroids = None
            self._fit_info = f"kmeans unavailable: {exc}"
            return False
        self.centroids = np.stack([X[labels == j].mean(axis=0) for j in range(k)])
        norms = np.linalg.norm(self.centroids, axis=1, keepdims=True)
        self.centroids = self.centroids / np.maximum(norms, 1e-9)
        for nid in ids:
            self._assign(nid)
        self._fit_info = f"k={k} n={len(ids)}"
        return True

    def _assign(self, nid: str, vec: Optional[np.ndarray] = None
                ) -> Tuple[Tuple[str, ...], Tuple[float, ...]]:
        """Top-K membership with L1-normalized weights. Stores node members;
        ad-hoc vectors (queries) are scored without storing."""
        v = self._vectors.get(nid) if vec is None else np.asarray(vec, dtype=np.float32)
        if v is None or self.centroids is None:
            return (), ()
        n = float(np.linalg.norm(v))
        sims = self.centroids @ (v / n) if n > 1e-9 else self.centroids @ v
        ids: List[str] = []
        raw: List[float] = []
        for j in np.argsort(-sims)[: self.top_k]:
            if sims[j] <= self.overlap_floor:
                break
            ids.append(f"c{int(j):03d}")
            raw.append(float(max(sims[j], 0.0)))
        s = sum(raw) or 1.0
        weights = tuple(w / s for w in raw)
        if vec is None and nid != QUERY_KEY:
            self.members[nid] = tuple(ids)
            self.weight_of[nid] = weights
        return tuple(ids), weights

    # -- scoring ----------------------------------------------------------
    def query_membership(self, qvec: np.ndarray) -> Tuple[Tuple[str, ...],
                                                          Tuple[float, ...]]:
        """Membership of a query vector: the same kernel as node assignment."""
        return self._assign(QUERY_KEY, vec=qvec)

    def node_score(self, nid: str, q_ids: Tuple[str, ...],
                   q_w: Tuple[float, ...]) -> float:
        """overlap(q, node) = sum_c w_q(c) * w_m(c), clipped to [0, 1]."""
        m_ids = self.members.get(nid)
        if not m_ids or not q_ids:
            return 0.0
        w_map = dict(zip(m_ids, self.weight_of.get(nid, ())))
        return float(np.clip(sum(q_w[i] * w_map.get(c, 0.0)
                                 for i, c in enumerate(q_ids)), 0.0, 1.0))

    # -- persistence via node metadata (survives SQLite round-trip) --------
    def attach_to_nodes(self, frame: MemoryFrame) -> int:
        """Write {ids, weights} into node metadata; returns nodes attached."""
        attached = 0
        for nid, ids in self.members.items():
            node = frame.nodes.get(nid)
            if node is None:
                continue
            node.metadata[META_KEY] = {"ids": list(ids),
                                       "weights": [round(w, 6) for w in
                                                   self.weight_of.get(nid, ())]}
            attached += 1
        return attached

    def info(self) -> Dict[str, Any]:
        return {"fitted": self.centroids is not None, "detail": self._fit_info,
                "top_k": self.top_k, "members": len(self.members)}


class VersionChain:
    """Persistent/versioned memory: append-only history keyed by stable hash."""

    def __init__(self, max_depth: int = 4):
        self.max_depth = max(1, int(max_depth))
        self.versions: Dict[str, List[Dict[str, Any]]] = {}

    @staticmethod
    def key(subject: str) -> str:
        """Chain key = hash of the subject (the thing whose facts change)."""
        return hashlib.sha256(subject.strip().lower().encode()).hexdigest()[:16]

    def update(self, subject: str, content: str, source: str = "",
               provenance: Optional[List[str]] = None) -> Dict[str, Any]:
        """Append a version; identical content does not duplicate an entry."""
        key = self.key(subject)
        hist = self.versions.setdefault(key, [])
        if hist and hist[-1]["content"] == content:
            return hist[-1]
        rec = {"content": content, "key": key, "at": iso_now(),
               "source": source, "provenance": list(provenance or []),
               "supersedes": hist[-1]["hash"] if hist else None,
               "n": len(hist) + 1,
               "hash": hashlib.sha256(
                   f"{key}|{len(hist) + 1}|{content}".encode()).hexdigest()[:16]}
        hist.append(rec)
        if len(hist) > self.max_depth:
            # ponytail: hard cap discards the oldest record; upgrade path =
            # spill to cold storage. Appends never mutate surviving entries'
            # content, only their ordinal after the drop.
            hist.pop(0)
            for i, r in enumerate(hist, start=1):
                r["n"] = i
        return rec

    def history(self, subject: str) -> List[Dict[str, Any]]:
        return list(self.versions.get(self.key(subject), []))

    def current(self, subject: str) -> Optional[Dict[str, Any]]:
        hist = self.versions.get(self.key(subject), [])
        return hist[-1] if hist else None

    def rollback(self, subject: str, n: int) -> Optional[Dict[str, Any]]:
        """Restore version n as the new head — append-only, never rewrite."""
        hist = self.versions.get(self.key(subject), [])
        if not (1 <= n <= len(hist)):
            return None
        target = dict(hist[n - 1])
        return self.update(subject, target["content"],
                           source=f"rollback:v{n}", provenance=[target["hash"]])

    def __len__(self) -> int:
        return sum(len(h) for h in self.versions.values())


# ---- retrieval wiring (used by core/retrieval.py) ---------------------------

def attach_constellations(frame: MemoryFrame, config: Dict[str, Any],
                          existing: Optional[ConstellationIndex] = None,
                          ) -> Optional[ConstellationIndex]:
    """Fit (or reuse) a ConstellationIndex. Returns None when NCM is disabled
    — the sentinel retrieval.py checks to keep classic MIRA unchanged."""
    cfg = ncm_config(config)
    if not cfg["enabled"]:
        return None
    if existing is not None:
        return existing
    idx = ConstellationIndex(top_k=cfg["top_k"],
                             max_constellations=cfg["max_constellations"],
                             overlap_floor=cfg["overlap_floor"])
    idx.fit(frame)
    idx.attach_to_nodes(frame)
    return idx


def constellation_score(idx: Optional[ConstellationIndex], nid: str,
                        qvec: np.ndarray) -> float:
    """Safe scoring entry: 0.0 when NCM is off or the node is unassigned."""
    if idx is None or idx.centroids is None or nid not in idx.members:
        return 0.0
    q_ids, q_w = idx.query_membership(qvec)
    return idx.node_score(nid, q_ids, q_w)
