"""Placement strategies (spec §11).

Five strategies assign (ring, sector) to every node:
  A embedding_clusters  — rings from k-means distance bands, sectors from clusters
  B graph_centrality    — rings from centrality rank, sectors from graph communities
  C hierarchy           — rings from parent-child depth, sectors from top-level siblings
  D temporal            — rings from recency, sectors from time buckets
  E hybrid_mira         — hierarchy ring ∩ embedding sector, centrality pulls inward

E is the research candidate; A-D are ablation baselines for the placement
decision itself. All operate on an already-embedded MemoryFrame.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.memory import MemoryFrame, MemoryNode
from core.types import utcnow, datetime

logger = logging.getLogger("mira.placement")

STRATEGIES = ("embedding_clusters", "graph_centrality", "hierarchy", "temporal", "hybrid_mira")


def _kmeans(X: np.ndarray, k: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    k = max(2, min(k, len(X)))
    return KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)


def _token_name(members: List[MemoryNode], fallback: str) -> str:
    from collections import Counter
    stop = {"the", "and", "of", "in", "to", "a", "an", "for", "on", "with",
            "is", "are", "was", "were", "this", "that", "it", "its", "as",
            "by", "at", "or", "be", "can", "has", "have", "from", "not",
            "you", "your", "page", "also", "may", "one", "other", "which",
            "what", "when", "how", "who", "all", "more", "most", "use",
            "used", "using", "such", "these", "those", "their", "they",
            "example", "examples", "see", "references", "external", "links",
            "edit", "article", "wiki", "wikipedia", "main", "content"}
    c: Counter = Counter()
    for m in members:
        for tok in re.findall(r"[a-z][a-z\-]{2,}", m.concept.lower()):
            if tok not in stop:
                c[tok] += 1
    return c.most_common(1)[0][0][:24] if c else fallback


def _apply(frame: MemoryFrame, assignment: Dict[str, Tuple[int, str]]) -> None:
    for n in frame.nodes.values():
        ring, sector = assignment.get(n.id, (3, "general"))
        n.ring = int(np.clip(ring, 0, 4))
        n.sector = sector
        n.depth = n.ring


def place(frame: MemoryFrame, strategy: str,
          config: Optional[Dict] = None) -> Dict[str, Any]:
    """Assign ring+sector to all nodes using the chosen strategy, then
    compute the §12 radial distance for every node (all strategies share
    the same radial metric so ablations differ only in ring/sector)."""
    if strategy not in STRATEGIES:
        strategy = "hybrid_mira"
    nodes = sorted(frame.nodes.values(), key=lambda node: node.id)
    if not nodes:
        return {"strategy": strategy, "rings": {}, "sectors": []}

    if strategy == "embedding_clusters":
        info = _place_embedding(frame, nodes)
    elif strategy == "graph_centrality":
        info = _place_centrality(frame, nodes)
    elif strategy == "hierarchy":
        info = _place_hierarchy(frame, nodes)
    elif strategy == "temporal":
        info = _place_temporal(frame, nodes)
    else:
        info = _place_hybrid(frame, nodes)
    _apply_radial(frame, config or {})
    info["radial"] = {
        "computed": sum(1 for n in frame.nodes.values()
                        if n.radial_distance is not None),
        "weights": _radial_weights(config or {}),
    }
    return info


# --- §12: radial distance ---------------------------------------------------
def _radial_weights(config: Dict) -> Dict[str, float]:
    w = (config.get("radial", {}) or config.get("radial_distance", {}) or {})
    return {
        "alpha_semantic": float(w.get("alpha_semantic", 0.4)),
        "beta_hierarchy": float(w.get("beta_hierarchy", 0.2)),
        "gamma_graph": float(w.get("gamma_graph", 0.3)),
        "delta_temporal": float(w.get("delta_temporal", 0.1)),
    }


def _apply_radial(frame: MemoryFrame, config: Dict) -> None:
    """radial_distance = α·semantic_distance + β·hierarchy_distance
    + γ·graph_distance + δ·temporal_distance (§12, experimental weights).
    0 = core, 1 = rim. Every node gets a value."""
    from storage.graph_store import GraphStore
    ws_w = _radial_weights(config)
    embedded = sorted(
        (n for n in frame.nodes.values() if n.embedding is not None),
        key=lambda node: node.id,
    )
    if not embedded:
        return
    X = np.stack([n.embedding for n in embedded])
    cent = X.mean(axis=0)
    cent_n = np.linalg.norm(cent) or 1.0
    sem = {n.id: float(1.0 - np.dot(cent, e) / (cent_n * (np.linalg.norm(e) or 1.0)))
           for n, e in zip(embedded, X)}
    gs = GraphStore()
    gs.build_from([n.to_row() for n in frame.nodes.values()],
                  [e.to_row() for e in frame.edges])
    raw_cent = gs.centrality()
    cmax = max(raw_cent.values()) if raw_cent else 0.0
    now = utcnow()
    for n in frame.nodes.values():
        d_hier = (n.ring or 0) / 4.0
        d_graph = 1.0 - (raw_cent.get(n.id, 0.0) / cmax) if cmax > 0 else 1.0
        try:
            age_days = (now - datetime.fromisoformat(n.created_at)).total_seconds() / 86400.0
            d_temp = float(np.clip(age_days / 30.0, 0.0, 1.0))
        except Exception:
            d_temp = 1.0
        d_sem = float(np.clip(sem.get(n.id, 1.0), 0.0, 1.0))
        n.radial_distance = round(float(np.clip(
            ws_w["alpha_semantic"] * d_sem
            + ws_w["beta_hierarchy"] * d_hier
            + ws_w["gamma_graph"] * d_graph
            + ws_w["delta_temporal"] * d_temp, 0.0, 1.0)), 4)


# --- A: embedding clusters -------------------------------------------------
def _place_embedding(frame: MemoryFrame, nodes: List[MemoryNode]) -> Dict:
    embedded = [n for n in nodes if n.embedding is not None]
    assignment: Dict[str, Tuple[int, str]] = {}
    if len(embedded) >= 6:
        X = np.stack([n.embedding for n in embedded])
        k = max(2, min(8, int(len(nodes) ** 0.5)))
        labels = _kmeans(X, k)
        cent = X.mean(axis=0)
        dists = np.linalg.norm(X - cent, axis=1)
        # ring bands from distance quintiles
        bands = np.quantile(dists, [0.2, 0.4, 0.6, 0.8])
        names = {}
        for ci in range(k):
            members = [n for n, label in zip(embedded, labels) if label == ci]
            names[ci] = _token_name(members, f"sector_{ci}")
        for n, lab, d in zip(embedded, labels, dists):
            ring = int(np.sum(d > bands))
            assignment[n.id] = (ring, names[lab])
    for n in nodes:
        if n.id not in assignment:
            assignment[n.id] = (3, "general")
    _apply(frame, assignment)
    return {"strategy": "embedding_clusters", "sectors": sorted({s for _, s in assignment.values()})}


# --- B: graph centrality -----------------------------------------------------
def _place_centrality(frame: MemoryFrame, nodes: List[MemoryNode]) -> Dict:
    from storage.graph_store import GraphStore
    gs = GraphStore()
    gs.build_from([n.to_row() for n in nodes], [e.to_row() for e in frame.edges])
    cent = gs.centrality()
    ranked = sorted(nodes, key=lambda n: -cent.get(n.id, 0.0))
    assignment: Dict[str, Tuple[int, str]] = {}
    n_len = max(1, len(ranked))
    for i, n in enumerate(ranked):
        ring = int(i / n_len * 5)  # most central → ring 0
        assignment[n.id] = (ring, "core" if ring == 0 else f"community_{ring}")
    _apply(frame, assignment)
    return {"strategy": "graph_centrality", "sectors": sorted({s for _, s in assignment.values()})}


# --- C: hierarchy ------------------------------------------------------------
def _place_hierarchy(frame: MemoryFrame, nodes: List[MemoryNode]) -> Dict:
    assignment: Dict[str, Tuple[int, str]] = {}
    roots = [n for n in nodes if n.parent_id is None]
    frontier = [(r, 0) for r in roots]
    seen = set()
    while frontier:
        node, depth = frontier.pop(0)
        if node.id in seen:
            continue
        seen.add(node.id)
        sector = node.concept.split()[0][:24] if depth <= 1 else \
            assignment.get(node.parent_id, ("", "general"))[1]
        assignment[node.id] = (min(depth, 4), sector)
        for cid in frame.children_of(node.id):
            if cid in frame.nodes:
                frontier.append((frame.nodes[cid], depth + 1))
    for n in nodes:
        if n.id not in assignment:
            assignment[n.id] = (3, "general")
    _apply(frame, assignment)
    return {"strategy": "hierarchy", "sectors": sorted({s for _, s in assignment.values()})}


# --- D: temporal ---------------------------------------------------------------
def _place_temporal(frame: MemoryFrame, nodes: List[MemoryNode]) -> Dict:
    def age_hours(n: MemoryNode) -> float:
        try:
            return (utcnow() - datetime.fromisoformat(n.created_at)).total_seconds() / 3600.0
        except Exception:
            return 1e9
    ranked = sorted(nodes, key=age_hours)  # newest first → inner rings
    assignment: Dict[str, Tuple[int, str]] = {}
    n_len = max(1, len(ranked))
    for i, n in enumerate(ranked):
        ring = int(i / n_len * 5)
        bucket = ["today", "week", "month", "quarter", "older"][min(ring, 4)]
        assignment[n.id] = (ring, bucket)
    _apply(frame, assignment)
    return {"strategy": "temporal", "sectors": sorted({s for _, s in assignment.values()})}


# --- E: hybrid MIRA ------------------------------------------------------------
def _place_hybrid(frame: MemoryFrame, nodes: List[MemoryNode]) -> Dict:
    """§9 ring semantics from structure and type — not a BFS that collapses
    when parent_id is unset:
      ring 0 core (the single hub) · ring 1 major concepts (well-connected)
      ring 2 subconcepts/entities · ring 3 facts/evidence · ring 4 documents.
    Sectors via embedding clusters; centrality breaks ring ties inward."""
    from storage.graph_store import GraphStore
    gs = GraphStore()
    gs.build_from([n.to_row() for n in nodes], [e.to_row() for e in frame.edges])
    cent = gs.centrality()
    deg = {nid: len(gs.neighborhood(nid, radius=1)) for nid in cent}
    non_doc_deg = [d for n, d in deg.items()
                   if n in frame.nodes
                   and frame.nodes[n].memory_type.value != "document"]
    hub_cut = max(4, int(np.quantile(non_doc_deg, 0.8)) if non_doc_deg else 4)
    # ring 0 = best-connected NON-document node (documents outrank everything
    # in raw degree but belong on the rim per §9)
    non_doc_deg_map = {n: d for n, d in deg.items()
                       if n in frame.nodes
                       and frame.nodes[n].memory_type.value != "document"}
    core_id = max(non_doc_deg_map, key=non_doc_deg_map.get) if non_doc_deg_map else None

    # sector discovery via embeddings
    embedded = [n for n in nodes if n.embedding is not None]
    names: Dict[int, str] = {}
    labels: Dict[str, int] = {}
    if len(embedded) >= 6:
        X = np.stack([n.embedding for n in embedded])
        k = max(2, min(8, int(len(nodes) ** 0.5)))
        labs = _kmeans(X, k)
        for n, label in zip(embedded, labs):
            labels[n.id] = int(label)
        for ci in sorted(set(int(label) for label in labs)):
            members = [n for n, label in zip(embedded, labs) if label == ci]
            # name by the most central member's salient token, not raw counts
            members.sort(key=lambda m: -cent.get(m.id, 0.0))
            names[int(ci)] = _token_name(members[:5], f"sector_{ci}")

    assignment: Dict[str, Tuple[int, str]] = {}
    for n in nodes:
        t = n.memory_type.value
        if n.id == core_id:
            ring = 0  # core check first: type branches must not preempt it
        elif t == "document":
            ring = 4
        elif t in ("fact", "event", "evidence"):
            ring = 3
        elif deg.get(n.id, 0) >= hub_cut or t in ("concept", "relation"):
            ring = 1 if deg.get(n.id, 0) >= hub_cut else 2
        else:
            ring = 2 if deg.get(n.id, 0) >= 2 else 3
        sector = names.get(labels.get(n.id, -1), "general")
        assignment[n.id] = (ring, sector)
    _apply(frame, assignment)
    return {"strategy": "hybrid_mira", "sectors": sorted({s for _, s in assignment.values()})}


def _hierarchy_assignment(frame: MemoryFrame,
                          nodes: List[MemoryNode]) -> Dict[str, Tuple[int, str]]:
    assignment: Dict[str, Tuple[int, str]] = {}
    roots = [n for n in nodes if n.parent_id is None]
    frontier = [(r, 0) for r in roots]
    seen = set()
    while frontier:
        node, depth = frontier.pop(0)
        if node.id in seen:
            continue
        seen.add(node.id)
        assignment[node.id] = (min(depth, 4), "general")
        for cid in frame.children_of(node.id):
            if cid in frame.nodes:
                frontier.append((frame.nodes[cid], depth + 1))
    for n in nodes:
        if n.id not in assignment:
            assignment[n.id] = (3, "general")
    return assignment
