"""NetworkX graph store (spec §14).

Thin wrapper so NetworkX can be swapped later. Nodes are memory-node ids;
edges carry relation_type, weight, confidence. Node attributes (ring,
sector, etc.) live on the memory side and are attached here only for
convenience of centrality computations.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

import networkx as nx

logger = logging.getLogger("mira.graph")


class GraphStore:
    def __init__(self) -> None:
        self.g = nx.Graph()

    # ---- construction ----
    def add_node(self, node_id: str, **attrs: Any) -> None:
        self.g.add_node(node_id, **attrs)

    def add_edge(self, source_id: str, target_id: str, relation_type: str = "related",
                 weight: float = 1.0, confidence: float = 0.5,
                 provenance: Optional[List[str]] = None) -> None:
        if source_id == target_id:
            return
        provenance = list(provenance or [])
        # networkx.Graph has one undirected edge per pair. Keep all relation
        # labels, but make the selected (strongest) relation authoritative for
        # confidence and provenance as well as weight.
        if self.g.has_edge(source_id, target_id):
            cur = self.g[source_id][target_id]
            current_weight = float(cur.get("weight", 1.0))
            current_confidence = float(cur.get("confidence", 0.0))
            relations = set(cur.get("relations") or {"related"})
            relations.add(relation_type)
            cur["relations"] = relations
            cur["weight"] = max(current_weight, float(weight))
            if (float(weight), float(confidence)) > (current_weight, current_confidence):
                cur["relation_type"] = relation_type
                cur["confidence"] = confidence
                cur["provenance"] = provenance
        else:
            self.g.add_edge(source_id, target_id, weight=weight,
                            relation_type=relation_type,
                            relations={relation_type},
                            confidence=confidence,
                            provenance=provenance)

    def build_from(self, nodes: List[Dict], edges: List[Dict]) -> None:
        self.g.clear()
        for n in sorted(nodes, key=lambda item: str(item.get("id", ""))):
            self.add_node(n["id"], ring=n.get("ring"), sector=n.get("sector"),
                          memory_type=n.get("memory_type"))
        for e in sorted(edges, key=lambda item: (
                str(item.get("source_id", "")), str(item.get("target_id", "")),
                str(item.get("relation_type", "")))):
            self.add_edge(e["source_id"], e["target_id"],
                          relation_type=e.get("relation_type", "related"),
                          weight=e.get("weight", 1.0),
                          confidence=e.get("confidence", 0.5),
                          provenance=e.get("provenance"))

    # ---- queries (spec §14) ----
    def has_node(self, node_id: str) -> bool:
        return self.g.has_node(node_id)

    def shortest_path(self, a: str, b: str) -> List[str]:
        try:
            return nx.shortest_path(self.g, a, b)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def weighted_path(self, a: str, b: str) -> List[str]:
        """Least-cost path treating weight as strength (cost = 1/weight)."""
        try:
            return nx.shortest_path(self.g, a, b, weight=lambda u, v, d: 1.0 / max(d.get("weight", 1.0), 1e-6))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def neighborhood(self, node_id: str, radius: int = 1) -> Set[str]:
        if not self.g.has_node(node_id):
            return set()
        return set(nx.ego_graph(self.g, node_id, radius=radius).nodes)

    def degree(self, node_id: str) -> int:
        return self.g.degree(node_id) if self.g.has_node(node_id) else 0

    def centrality(self) -> Dict[str, float]:
        if self.g.number_of_nodes() == 0:
            return {}
        try:
            return nx.degree_centrality(self.g)  # O(n) — laptop-scale friendly
        except Exception:
            return {}

    def connected_components(self) -> List[Set[str]]:
        return [set(c) for c in nx.connected_components(self.g)]

    def subgraph(self, node_ids: Set[str]) -> "GraphStore":
        sub = GraphStore()
        sub.g = self.g.subgraph(node_ids).copy()
        return sub

    def edge_data(self, a: str, b: str) -> Dict:
        return self.g[a][b] if self.g.has_edge(a, b) else {}

    def edges_of(self, node_id: str) -> List[Dict]:
        """All edges touching node_id, normalized to source/target dicts."""
        if not self.g.has_node(node_id):
            return []
        out: List[Dict] = []
        for u, v, d in self.g.edges(node_id, data=True):
            out.append({"source_id": u, "target_id": v,
                        "relation_type": d.get("relation_type")
                        or sorted(d.get("relations", {"related"}))[0],
                        "weight": d.get("weight", 1.0),
                        "confidence": d.get("confidence", 0.5),
                        "provenance": list(d.get("provenance") or [])})
        return sorted(out, key=lambda item: (
            str(item["source_id"]), str(item["target_id"]),
            str(item["relation_type"])))

    # ---- persistence ----
    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as fh:
            pickle.dump(self.g, fh)

    @classmethod
    def load(cls, path: str) -> "GraphStore":
        import pickle
        store = cls()
        with open(path, "rb") as fh:
            store.g = pickle.load(fh)
        return store
