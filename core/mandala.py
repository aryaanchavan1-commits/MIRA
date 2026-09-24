"""Mandala topology (spec §9-12).

Formal computational topology, not just a picture:
- ring      = depth from core (hierarchy level)
- sector    = semantic cluster (discovered from embeddings, not hard-coded)
- radial_distance = α·semantic + β·hierarchy + γ·graph + δ·temporal (§12)

All weights are experimental defaults — the ablation framework exists to
test whether they help (spec §52).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.memory import MemoryFrame, MemoryNode
from core.types import iso_now, parse_float, utcnow
from models.embeddings import EmbeddingBackend

logger = logging.getLogger("mira.mandala")

RING_LABELS = {
    0: "core",
    1: "major concepts",
    2: "subconcepts / entities / relations",
    3: "facts / evidence",
    4: "raw documents / detailed evidence",
}


class MandalaBuilder:
    def __init__(self, config: Dict[str, Any], embeddings: EmbeddingBackend):
        self.config = config
        self.embeddings = embeddings
        topo = config.get("topology", {})
        self.max_rings = int(topo.get("max_rings", 5))
        self.placement_strategy = topo.get("placement_strategy", "hybrid_mira")
        rd = config.get("radial_distance", {})
        self.w_semantic = parse_float(rd.get("alpha_semantic", 0.4), 0.4)
        self.w_hierarchy = parse_float(rd.get("beta_hierarchy", 0.25), 0.25)
        self.w_graph = parse_float(rd.get("gamma_graph", 0.2), 0.2)
        self.w_temporal = parse_float(rd.get("delta_temporal", 0.15), 0.15)

    # ------------------------------------------------------------------
    # ring assignment (strategy C: hierarchy)
    # ------------------------------------------------------------------
    def assign_rings(self, frame: MemoryFrame) -> None:
        roots = [n for n in frame.nodes.values() if n.parent_id is None]
        # BFS from roots; orphans at max ring
        for root in roots:
            root.ring = 0 if root.memory_type.value in ("concept", "entity") else 1
        frontier = roots
        ring = 0
        seen = {n.id for n in roots}
        while frontier:
            nxt: List[MemoryNode] = []
            for node in frontier:
                for cid in frame.children_of(node.id):
                    if cid in frame.nodes and cid not in seen:
                        child = frame.nodes[cid]
                        child.ring = min((node.ring or 0) + 1, self.max_rings - 1)
                        seen.add(cid)
                        nxt.append(child)
            frontier = nxt
            ring += 1
        for n in frame.nodes.values():
            if n.ring is None:
                n.ring = min(self.max_rings - 1, (n.depth or 0) + 2)
            n.depth = n.ring

    # ------------------------------------------------------------------
    # sector discovery (strategy A: embedding clustering)
    # ------------------------------------------------------------------
    def discover_sectors(self, frame: MemoryFrame) -> List[str]:
        nodes = [n for n in frame.nodes.values() if n.embedding is not None]
        if len(nodes) < 6:
            # too few for clustering: one sector
            for n in frame.nodes.values():
                n.sector = "general"
            return ["general"]
        X = np.stack([n.embedding for n in nodes]).astype("float32")
        # deterministic k for reproducibility; cap sectors at sqrt(n)
        k = max(2, min(8, int(len(nodes) ** 0.5)))
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(X)
            centroids = km.cluster_centers_
        except Exception as exc:
            logger.warning("sklearn unavailable (%s) — single sector fallback", exc)
            for n in frame.nodes.values():
                n.sector = "general"
            return ["general"]

        # name sectors by top distinctive token across cluster members
        names: List[str] = []
        for ci in range(k):
            members = [n for n, lab in zip(nodes, labels) if lab == ci]
            if not members:
                names.append(f"sector_{ci}")
                continue
            names.append(self._name_sector(members, ci, len(names)))
        for n, lab in zip(nodes, labels):
            n.sector = names[lab]
        for n in frame.nodes.values():
            if n.sector is None:
                n.sector = "general"
        return sorted(set(names))

    @staticmethod
    def _name_sector(members: List[MemoryNode], ci: int, ordinal: int) -> str:
        """Top TF token in member concepts, not a stopword → readable name."""
        from collections import Counter
        stop = {"the", "and", "of", "in", "to", "a", "for", "on", "with", "is",
                "are", "as", "by", "an", "or", "be", "this", "that", "it", "de"}
        counter: Counter = Counter()
        for m in members:
            for tok in m.concept.lower().split()[:12]:
                if len(tok) > 2 and tok not in stop:
                    counter[tok] += 1
        if counter:
            tok, _ = counter.most_common(1)[0]
            return tok[:24]
        return f"sector_{ordinal}"

    # ------------------------------------------------------------------
    # radial distance (§12): weighted combination of four distances
    # ------------------------------------------------------------------
    def radial_distance(self, node: MemoryNode, frame: MemoryFrame,
                        centrality: Optional[Dict[str, float]] = None) -> float:
        centrality = centrality or {}
        sem = float(np.linalg.norm(node.embedding)) if node.embedding is not None else 1.0
        # semantic_distance: 1 - max similarity to any ring-0/1 node (hub proximity)
        hub_sims = [float(np.dot(node.embedding, h.embedding))
                    for h in frame.nodes.values()
                    if h.embedding is not None and h.ring in (0, 1) and h.id != node.id]
        if hub_sims:
            sem = 1.0 - max(hub_sims)
        hier = (node.ring or (self.max_rings - 1)) / max(1, self.max_rings - 1)
        # graph distance: 1 - normalized degree centrality
        deg = centrality.get(node.id, 0.0)
        graph_term = 1.0 - min(1.0, deg * 5)   # 5 neighbors ≈ peripheral→central
        # temporal: age relative to newest node
        from datetime import datetime
        age_hours = max(0.0, (utcnow() - datetime.fromisoformat(node.created_at)).total_seconds() / 3600.0)
        temporal = min(1.0, age_hours / max(1.0, 24.0 * 30))  # 30-day horizon
        total = (self.w_semantic * sem + self.w_hierarchy * hier +
                 self.w_graph * graph_term + self.w_temporal * temporal)
        wsum = self.w_semantic + self.w_hierarchy + self.w_graph + self.w_temporal
        return round(total / wsum if wsum > 0 else total, 4)

    # ------------------------------------------------------------------
    # full build: rings → sectors → radial distances (§9-12)
    # ------------------------------------------------------------------
    def build(self, frame: MemoryFrame) -> Dict[str, Any]:
        self.assign_rings(frame)
        sectors = self.discover_sectors(frame)
        # graph centrality via storage layer
        from storage.graph_store import GraphStore
        gs = GraphStore()
        gs.build_from([n.to_row() for n in frame.nodes.values()],
                      [e.to_row() for e in frame.edges])
        cent = gs.centrality()
        for n in frame.nodes.values():
            n.radial_distance = self.radial_distance(n, frame, cent)
        return {"sectors": sectors, "n_nodes": len(frame.nodes),
                "strategy": self.placement_strategy}
