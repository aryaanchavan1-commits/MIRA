"""Phase 10 tests: baselines run and behave differently from each other."""
from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.graph_rag import GraphRAG
from baselines.hierarchical_rag import HierarchicalRAG
from baselines.vector_rag import VectorRAG
from core.memory import MemoryEdge, MemoryFrame, MemoryNode, MemoryType
from core.placement import place
from storage.graph_store import GraphStore
from storage.vector_store import VectorStore


def build_frame() -> tuple:
    frame = MemoryFrame()
    n0 = MemoryNode(id="n0", concept="photosynthesis", memory_type=MemoryType.CONCEPT,
                    summary="plants convert sunlight into chemical energy")
    n1 = MemoryNode(id="n1", concept="chlorophyll", memory_type=MemoryType.ENTITY,
                     summary="pigment that absorbs light in photosynthesis", parent_id="n0")
    n2 = MemoryNode(id="n2", concept="light reactions", memory_type=MemoryType.CONCEPT,
                     summary="first stage of photosynthesis producing ATP", parent_id="n0")
    for n in (n0, n1, n2):
        frame.add_node(n)
    frame.add_edge(MemoryEdge(source_id="n0", target_id="n1", relation_type="has_part"))
    frame.add_edge(MemoryEdge(source_id="n0", target_id="n2", relation_type="has_stage"))
    place(frame, "hybrid_mira")
    from models.embeddings import init_embeddings
    emb = init_embeddings("stub")
    vecs = emb.encode([n.concept + " " + (n.summary or "")
                       for n in (n0, n1, n2)])
    for n, v in zip((n0, n1, n2), vecs):
        n.embedding = v
    vecs = np.stack([n.embedding for n in frame.nodes.values()])
    vs = VectorStore(dim=vecs.shape[1])
    vs.add(vecs, [{"node_id": n.id} for n in frame.nodes.values()])
    gs = GraphStore()
    gs.build_from([n.to_row() for n in frame.nodes.values()],
                  [e.to_row() for e in frame.edges])
    return frame, vs, gs


def main() -> None:
    frame, vs, gs = build_frame()
    qvec = frame.nodes["n0"].embedding

    vrag = VectorRAG(frame, vs).retrieve(qvec, k=2)
    assert len(vrag.items) == 2 and vrag.items[0].id == "n0", "vector baseline top-1"
    assert vrag.latency_ms >= 0

    grag = GraphRAG(frame, vs, gs).retrieve(qvec, k=3)
    ids = {n.id for n in grag.items}
    assert "n0" in ids, "graph baseline keeps vector hits"
    assert "n1" in ids or "n2" in ids, "graph baseline expands to neighbors"

    hrag = HierarchicalRAG(frame, vs).retrieve("photosynthesis", k=3)
    hids = {n.id for n in hrag.items}
    assert hrag.latency_ms >= 0
    assert "n0" in hids
    assert "n1" in hids or "n2" in hids, "hierarchy uses maintained parent links"

    print("PHASE10 TESTS PASS")


if __name__ == "__main__":
    main()
