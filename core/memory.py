"""MIRA memory model (spec §7-8, §18-19).

MemoryNode / MemoryEdge as dataclasses; MemoryFrame is the in-memory bundle
(nodes+edges+stores) that the retrieval and placement layers operate on.
Rings, sectors and radial placement live in mandala.py / placement.py —
this module only defines the representation and CRUD/history semantics.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

from core.types import iso_now, new_id

logger = logging.getLogger("mira.memory")


class MemoryType(str, Enum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    FACT = "fact"
    ENTITY = "entity"
    EVENT = "event"
    DOCUMENT = "document"
    CONCEPT = "concept"
    RELATION = "relation"

    @classmethod
    def coerce(cls, value: Any) -> "MemoryType":
        try:
            return cls(str(value).lower())
        except ValueError:
            return cls.SEMANTIC  # extensible: unknown types map to semantic


@dataclass
class MemoryNode:
    concept: str
    memory_type: MemoryType = MemoryType.SEMANTIC
    summary: str = ""
    raw_text: str = ""
    id: str = field(default_factory=lambda: new_id("mem"))
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    ring: Optional[int] = None
    sector: Optional[str] = None
    depth: int = 0
    radial_distance: Optional[float] = None
    embedding: Any = None          # numpy array, not persisted in SQLite
    importance: float = 0.5
    confidence: float = 0.5
    created_at: str = field(default_factory=iso_now)
    updated_at: str = field(default_factory=iso_now)
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    source_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("embedding", None)
        d.pop("children", None)      # derived from parent_id, not stored
        d["memory_type"] = self.memory_type.value
        return d

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "MemoryNode":
        row = dict(row)
        row["memory_type"] = MemoryType.coerce(row.get("memory_type"))
        row.pop("status", None)
        emb = row.pop("embedding", None)
        node = cls(**{k: v for k, v in row.items() if k in cls.__dataclass_fields__})
        node.embedding = emb
        return node


@dataclass
class MemoryEdge:
    source_id: str
    target_id: str
    relation_type: str = "related"
    weight: float = 1.0
    confidence: float = 0.5
    provenance: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=iso_now)

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryCluster:
    """A sector: nodes grouped by embedding/graph similarity (spec §10)."""
    name: str
    node_ids: List[str] = field(default_factory=list)
    centroid: Any = None
    sector_index: int = 0          # angular position on the mandala
    coherence: float = 0.0


@dataclass
class MemoryRing:
    """Concentric band: ring 0 = core … ring N = raw evidence (spec §9)."""
    index: int
    label: str = ""
    node_ids: List[str] = field(default_factory=list)


class MemoryFrame:
    """In-memory working set: nodes + edges. Stores persist separately."""

    def __init__(self) -> None:
        self.nodes: Dict[str, MemoryNode] = {}
        self.edges: List[MemoryEdge] = []

    def add_node(self, node: MemoryNode) -> MemoryNode:
        self.nodes[node.id] = node
        return node

    def add_edge(self, edge: MemoryEdge) -> MemoryEdge:
        self.edges.append(edge)
        return edge

    def get_edge(self, source_id: str, target_id: str) -> Optional[MemoryEdge]:
        for e in self.edges:
            if ((e.source_id == source_id and e.target_id == target_id)
                    or (e.source_id == target_id and e.target_id == source_id)):
                return e
        return None

    def children_of(self, node_id: str) -> List[str]:
        return sorted(n.id for n in self.nodes.values() if n.parent_id == node_id)

    def neighbors(self, node_id: str) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for e in self.edges:
            if e.source_id == node_id:
                out.setdefault(e.relation_type, []).append(e.target_id)
            elif e.target_id == node_id:
                out.setdefault(e.relation_type, []).append(e.source_id)
        return out

    def stats(self) -> Dict[str, int]:
        by_type: Dict[str, int] = {}
        for n in self.nodes.values():
            by_type[n.memory_type.value] = by_type.get(n.memory_type.value, 0) + 1
        return {"nodes": len(self.nodes), "edges": len(self.edges), "by_type": by_type}


# ---------------------------------------------------------------------------
# Memory update operations (spec §19): never blind overwrite; keep history.
# The SQLite node_history table (written by sqlite_store.upsert_node) is the
# history mechanism; these helpers set the right fields per action.
# ---------------------------------------------------------------------------
VALID_ACTIONS = {"create", "merge", "update", "split", "invalidate", "archive", "restore"}


def apply_update(existing: MemoryNode, action: str,
                 patch: Optional[Dict[str, Any]] = None) -> MemoryNode:
    """Return a node copy with the action applied. History is recorded by the store."""
    patch = patch or {}
    n = MemoryNode.from_row({**existing.to_row(), "embedding": existing.embedding})
    n.updated_at = iso_now()
    if action == "update":
        for k, v in patch.items():
            if k in MemoryNode.__dataclass_fields__ and k not in ("id", "created_at"):
                setattr(n, k, v)
    elif action == "merge":
        # merge content: append raw text + sources, keep max importance/conf
        n.raw_text = (existing.raw_text + "\n" + patch.get("raw_text", "")).strip()
        n.summary = patch.get("summary", existing.summary)
        n.source_ids = sorted(set(existing.source_ids) | set(patch.get("source_ids", [])))
        n.importance = max(existing.importance, patch.get("importance", 0.0))
        n.confidence = max(existing.confidence, patch.get("confidence", 0.0))
        n.metadata = {**existing.metadata, **patch.get("metadata", {})}
    elif action == "invalidate":
        n.valid_until = iso_now()
        n.confidence = min(n.confidence, 0.2)
        n.metadata = {**n.metadata, "invalidated": patch.get("reason", "")}
    elif action == "archive":
        n.metadata = {**n.metadata, "archived": True}
    elif action == "restore":
        n.metadata = {k: v for k, v in n.metadata.items() if k != "archived"}
        n.valid_until = None
    elif action == "split":
        # caller creates the new node(s); this marks the original as split
        n.metadata = {**n.metadata, "split_from": n.id, "split_reason": patch.get("reason", "")}
    return n
