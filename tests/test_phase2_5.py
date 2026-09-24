"""Phase 2-5 tests: memory, stores, mandala topology, placement.

Run: .venv/Scripts/python.exe -m tests.test_phase2_5
Assert-based; no frameworks. Uses the hashing embedding fallback so it runs
without network/model downloads.
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.memory import MemoryFrame, MemoryNode, MemoryEdge, MemoryType, apply_update
from core.mandala import MandalaBuilder
from core.placement import place, STRATEGIES
from core.types import new_id
from models.embeddings import EmbeddingBackend
from storage.sqlite_store import SQLiteStore
from storage.vector_store import VectorStore
from storage.graph_store import GraphStore


def make_frame(n: int = 24) -> MemoryFrame:
    rng = np.random.RandomState(42)
    frame = MemoryFrame()
    emb = EmbeddingBackend("stub")  # hashing backend
    texts = [f"concept {i} about topic {i % 4} with detail {i}" for i in range(n)]
    vecs = emb.encode(texts)
    parent = None
    for i, text in enumerate(texts):
        node = MemoryNode(
            concept=text.split(" with ")[0],
            memory_type=MemoryType.CONCEPT if i < 4 else (MemoryType.FACT if i % 2 else MemoryType.EVENT),
            summary=f"summary {i}",
            raw_text=text,
            ring=None,
        )
        node.embedding = vecs[i]
        node.parent_id = parent
        node.importance = rng.rand()
        node.confidence = 0.5 + rng.rand() * 0.5
        frame.add_node(node)
        if parent is None:
            parent = node.id
        if i > 0 and i % 3 == 0:
            frame.add_edge(MemoryEdge(source_id=frame.nodes[list(frame.nodes)[i - 3]].id,
                                      target_id=node.id, relation_type="related",
                                      weight=0.8))
    return frame


def test_memory_model():
    frame = make_frame(12)
    stats = frame.stats()
    assert stats["nodes"] == 12, stats
    assert stats["edges"] >= 3, stats
    # children derivation
    root = list(frame.nodes.values())[0]
    assert frame.children_of(root.id), "root should have children"
    # update ops
    n = list(frame.nodes.values())[1]
    merged = apply_update(n, "merge", {"raw_text": "extra", "importance": 0.9})
    assert "extra" in merged.raw_text and merged.importance == 0.9
    inv = apply_update(n, "invalidate", {"reason": "contradicted"})
    assert inv.valid_until is not None and inv.confidence <= 0.2
    print("  memory model OK")


def test_sqlite_roundtrip():
    tmp = tempfile.mkdtemp(prefix="mira_test_")
    try:
        store = SQLiteStore(os.path.join(tmp, "t.db"))
        doc_id = store.add_document("Test Doc", "test.txt", "txt", "sha123", 100)
        chunk_id = store.add_chunk(doc_id, 0, "hello world", page=1)
        assert store.get_document(doc_id)["title"] == "Test Doc"
        assert store.get_chunk(chunk_id)["text"] == "hello world"

        node = MemoryNode(concept="python version", memory_type=MemoryType.FACT,
                          raw_text="python is version 3.11", source_ids=[chunk_id])
        store.upsert_node(node.to_row())
        loaded = MemoryNode.from_row(store.get_node(node.id))
        assert loaded.concept == "python version"
        assert loaded.source_ids == [chunk_id]
        # update keeps history
        updated = apply_update(loaded, "update", {"summary": "v3.11 confirmed"})
        store.upsert_node({**updated.to_row(), "_action": "update"})
        assert store.get_node(node.id)["summary"] == "v3.11 confirmed"
        assert store.count_nodes() == 1
        edges = store.all_edges()
        store.add_edge(node.id, "mem_other", "related", 1.0, 0.9)
        assert len(store.all_edges()) == 1

        exp_id = store.save_experiment("demo", {"a": 1}, {"recall": 0.5})
        assert store.list_experiments()[0]["id"] == exp_id
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  sqlite roundtrip OK")


def test_vector_store():
    tmp = tempfile.mkdtemp(prefix="mira_test_")
    try:
        emb = EmbeddingBackend("stub")
        vecs = emb.encode(["alpha beta", "gamma delta", "alpha beta again",
                           "completely different words here"])
        vs = VectorStore(dim=vecs.shape[1], persist_path=os.path.join(tmp, "v.faiss"))
        metas = [{"node_id": f"n{i}", "chunk_id": f"c{i}"} for i in range(4)]
        ids = vs.add(vecs, metas)
        assert vs.size() == 4
        q = emb.encode(["alpha beta"])
        hits = vs.search(q[0], k=2)
        assert hits and hits[0][1].get("node_id") == "n0", hits
        # persistence round trip
        vs.save()
        vs2 = VectorStore.load(os.path.join(tmp, "v.faiss"))
        hits2 = vs2.search(q[0], k=2)
        assert hits2[0][1] == hits[0][1]
        # delete
        vs.delete(ids[:2])
        assert vs.size() == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  vector store OK")


def test_graph_store():
    gs = GraphStore()
    gs.add_node("a"); gs.add_node("b"); gs.add_node("c"); gs.add_node("d")
    gs.add_edge("a", "b", "related", 2.0)
    gs.add_edge("b", "c", "related", 1.0)
    gs.add_edge("a", "c", "related", 1.0)
    gs.add_edge("d", "a", "related", 1.0)
    assert gs.shortest_path("d", "c") == ["d", "a", "b", "c"] or gs.shortest_path("d", "c") == ["d", "a", "c"]
    assert gs.degree("a") == 3
    cent = gs.centrality()
    assert abs(max(cent.values()) - cent["a"]) < 1e-9, cent
    assert len(gs.connected_components()) == 1
    sub = gs.subgraph({"a", "b", "c"})
    assert sub.degree("a") == 2
    nb = gs.neighborhood("a", radius=1)
    assert nb == {"a", "b", "c", "d"}
    print("  graph store OK")


def test_mandala_placement():
    for strat in STRATEGIES:
        frame = make_frame(24)
        result = place(frame, strat)
        assert result["strategy"] == strat
        rings = {n.ring for n in frame.nodes.values()}
        sectors = {n.sector for n in frame.nodes.values()}
        assert all(r is not None and 0 <= r <= 4 for r in rings), (strat, rings)
        assert all(s for s in sectors), (strat, sectors)
    # mandala builder assigns radial distance on the default path
    from models.embeddings import init_embeddings
    emb = init_embeddings("stub")
    frame = make_frame(20)
    builder = MandalaBuilder({"topology": {"max_rings": 5}}, emb)
    info = builder.build(frame)
    assert info["n_nodes"] == 20
    rads = [n.radial_distance for n in frame.nodes.values()]
    assert all(r is not None and 0.0 <= r <= 1.0 for r in rads), rads
    print("  mandala + 5 placement strategies OK")


def test_embedding_fallback_determinism():
    emb = EmbeddingBackend("stub")
    v1 = emb.encode(["deterministic check"])
    v2 = emb.encode(["deterministic check"])
    assert np.allclose(v1, v2), "hashing embeddings must be deterministic"
    assert not emb.is_real_model
    print("  embedding fallback OK")


if __name__ == "__main__":
    print("Phase 2-5 tests:")
    test_memory_model()
    test_sqlite_roundtrip()
    test_vector_store()
    test_graph_store()
    test_mandala_placement()
    test_embedding_fallback_determinism()
    print("ALL PASS")
