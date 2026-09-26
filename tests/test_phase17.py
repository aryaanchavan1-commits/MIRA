"""Phase 17 tests: real mandala topology — §9 ring semantics, §12 radial
distance, re-placement of existing memories."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.memory import MemoryEdge, MemoryFrame, MemoryNode, MemoryType
from core.placement import place
from core.workspace import Workspace
from models.embeddings import init_embeddings
from storage.graph_store import GraphStore


def _frame() -> MemoryFrame:
    f = MemoryFrame()
    doc = MemoryNode(id="doc1", concept="wikipedia article",
                     memory_type=MemoryType.DOCUMENT, summary="a document node")
    core = MemoryNode(id="core", concept="spreading activation",
                      memory_type=MemoryType.CONCEPT, summary="core mechanism",
                      parent_id="doc1")
    hub = MemoryNode(id="hub", concept="semantic network",
                     memory_type=MemoryType.CONCEPT, summary="well connected",
                     parent_id="doc1")
    fact = MemoryNode(id="fact1", concept="1975 paper",
                      memory_type=MemoryType.FACT, summary="published in 1975",
                      parent_id="core")
    leaf = MemoryNode(id="leaf1", concept="minor mention",
                      memory_type=MemoryType.ENTITY, summary="barely connected",
                      parent_id="hub")
    for n in (doc, core, hub, fact, leaf):
        f.add_node(n)
    f.add_edge(MemoryEdge(source_id="core", target_id="hub", relation_type="related"))
    f.add_edge(MemoryEdge(source_id="core", target_id="fact1", relation_type="cites"))
    f.add_edge(MemoryEdge(source_id="hub", target_id="leaf1", relation_type="mentions"))
    f.add_edge(MemoryEdge(source_id="core", target_id="leaf1", relation_type="related"))
    return f, doc, core, hub, fact, leaf


def test_ring_semantics_not_collapsed():
    f, doc, core, hub, fact, leaf = _frame()
    emb = init_embeddings("stub")
    for n in f.nodes.values():
        n.embedding = emb.encode([n.concept + " " + (n.summary or "")])[0]
    place(f, "hybrid_mira", config={})
    rings = {n.id: n.ring for n in f.nodes.values()}
    assert rings["doc1"] == 4, "documents belong on the outer ring (§9)"
    assert rings["fact1"] == 3, "facts/evidence on ring 3"
    assert rings["core"] <= 1, "the graph hub belongs at the core"
    assert rings["hub"] <= 2, "well-connected concepts stay inner"
    assert len(set(rings.values())) >= 3, "rings must span the topology"


def test_radial_distance_computed_for_all():
    f, *_ = _frame()
    emb = init_embeddings("stub")
    for n in f.nodes.values():
        n.embedding = emb.encode([n.concept])[0]
    place(f, "hybrid_mira", config={"radial": {"alpha_semantic": 0.4}})
    rd = [n.radial_distance for n in f.nodes.values()]
    assert all(v is not None for v in rd), "§12 radial distance must be computed"
    assert all(0.0 <= v <= 1.0 for v in rd)
    assert len(set(rd)) > 1, "radial distances must differentiate nodes"


def test_replace_all_reassigns():
    f, doc, core, *_ = _frame()
    emb = init_embeddings("stub")
    for n in f.nodes.values():
        n.embedding = emb.encode([n.concept])[0]
    place(f, "temporal", config={})
    r_before = {n.id: (n.ring, n.sector) for n in f.nodes.values()}
    place(f, "hybrid_mira", config={})
    r_after = {n.id: (n.ring, n.sector) for n in f.nodes.values()}
    assert r_before != r_after, "different strategies must produce different mandalas"
    assert all(n.radial_distance is not None for n in f.nodes.values())


def test_workspace_hydrates_edges_before_graph_build():
    source = MemoryNode(id="a", concept="source", memory_type=MemoryType.CONCEPT)
    target = MemoryNode(id="b", concept="target", memory_type=MemoryType.FACT,
                        parent_id="a")
    edge = MemoryEdge(source_id="a", target_id="b", relation_type="supports")

    class Store:
        def all_nodes(self):
            return [source.to_row(), target.to_row()]

        def all_edges(self):
            return [edge.to_row()]

    workspace = Workspace.__new__(Workspace)
    workspace.store = Store()
    workspace.frame = MemoryFrame()
    workspace.gs = GraphStore()
    workspace._load_frame()
    assert len(workspace.frame.edges) == 1
    assert workspace.gs.g.number_of_edges() == 1
    assert workspace.gs.neighborhood("a") == {"a", "b"}


def test_workspace_strategy_apply_updates_live_config():
    class Store:
        def __init__(self):
            self.rows = []

        def upsert_node(self, row):
            self.rows.append(row)

    workspace = Workspace.__new__(Workspace)
    workspace._lock = threading.RLock()
    workspace.config = {"topology": {"placement_strategy": "hybrid_mira"}}
    workspace.frame = MemoryFrame()
    node = MemoryNode(id="n", concept="node", memory_type=MemoryType.FACT,
                      summary="node")
    workspace.frame.add_node(node)
    workspace.store = Store()
    info = workspace.replace_all(strategy="temporal")
    assert info["strategy"] == "temporal"
    assert workspace.config["topology"]["placement_strategy"] == "temporal"
    assert workspace.store.rows and workspace.store.rows[0]["id"] == "n"


def test_workspace_strategy_failure_restores_live_placement():
    class FailingStore:
        def __init__(self):
            self.calls = 0

        def upsert_node(self, row):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("placement write failed")

    workspace = Workspace.__new__(Workspace)
    workspace._lock = threading.RLock()
    workspace.config = {"topology": {"placement_strategy": "hybrid_mira"}}
    workspace.frame = MemoryFrame()
    for node_id in ("a", "b"):
        workspace.frame.add_node(MemoryNode(
            id=node_id, concept=node_id, memory_type=MemoryType.FACT,
            summary=node_id))
    workspace.store = FailingStore()
    before = {node_id: (node.ring, node.sector, node.depth)
              for node_id, node in workspace.frame.nodes.items()}
    try:
        workspace.replace_all(strategy="temporal")
    except RuntimeError:
        pass
    else:
        raise AssertionError("placement persistence failure must propagate")
    assert workspace.config["topology"]["placement_strategy"] == "hybrid_mira"
    assert {node_id: (node.ring, node.sector, node.depth)
            for node_id, node in workspace.frame.nodes.items()} == before


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
