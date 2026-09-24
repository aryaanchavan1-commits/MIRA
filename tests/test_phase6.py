"""Phase 6 tests: retrieval, ablation, compression, conflicts, updater.

Run: .venv/Scripts/python.exe -m tests.test_phase6
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.compression import compress, est_tokens
from core.conflicts import detect_conflict, find_conflicts, resolve
from core.memory import MemoryEdge, MemoryFrame, MemoryNode, MemoryType
from core.placement import place
from core.retrieval import MIRARetriever, ablation_configs
from core.updater import MemoryUpdater
from models.embeddings import EmbeddingBackend
from storage.graph_store import GraphStore
from storage.sqlite_store import SQLiteStore
from storage.vector_store import VectorStore


def build_world(emb: EmbeddingBackend) -> tuple:
    frame = MemoryFrame()
    docs = {
        "py311": "python version is 3.11",
        "py312": "python version is 3.12",
        "gpt": "gpt models are large language models",
        "faiss": "faiss is a vector search library",
        "cuda": "cuda enables gpu acceleration",
    }
    vecs = emb.encode(list(docs.values()))
    nodes = []
    for (key, text), vec in zip(docs.items(), vecs):
        n = MemoryNode(concept=key, memory_type=MemoryType.FACT,
                       raw_text=text, summary=text, confidence=0.8)
        n.embedding = vec
        frame.add_node(n)
        nodes.append(n)
    # small hierarchy: one core concept
    core = MemoryNode(concept="ai infrastructure", memory_type=MemoryType.CONCEPT)
    core.embedding = emb.encode(["ai infrastructure"])[0]
    frame.add_node(core)
    for n in nodes:
        frame.add_edge(MemoryEdge(source_id=core.id, target_id=n.id,
                                  relation_type="related", weight=1.0, confidence=0.7))
    vs = VectorStore(dim=vecs.shape[1])
    vs.add(np.stack([n.embedding for n in frame.nodes.values()]),
           [{"node_id": n.id} for n in frame.nodes.values()])
    gs = GraphStore()
    gs.build_from([n.to_row() for n in frame.nodes.values()],
                  [e.to_row() for e in frame.edges])
    config = {
        "topology": {"max_rings": 5},
        "retrieval": {"candidate_k": 12, "final_k": 4, "max_hops": 3},
        "retrieval_score": {},
    }
    place(frame, "hybrid_mira")
    return frame, vs, gs, config, nodes, core


def test_retrieval_and_ablations():
    emb = EmbeddingBackend("stub")
    frame, vs, gs, config, nodes, core = build_world(emb)
    ret = MIRARetriever(frame, vs, gs, config)
    q = "python version"
    qvec = emb.encode([q])[0]
    result = ret.retrieve(q, qvec)

    assert result.items, "no items retrieved"
    top_ids = {it.node.id for it in result.items[:2]}
    py_ids = {n.id for n in nodes if "python" in n.raw_text}
    assert top_ids & py_ids, f"python facts not retrieved: {top_ids}"

    # every ablation config must run and produce a ranking
    for name, active in ablation_configs().items():
        r = ret.retrieve(q, qvec, active_components=active)
        assert r.items, f"ablation {name} produced nothing"
        for it in r.items:
            assert set(it.components) == set(
                ["semantic", "structural", "radial", "graph", "importance",
                 "confidence", "recency", "path", "activation"])
    # vector_only must rank the python facts top-2
    r = ret.retrieve(q, qvec, active_components=("semantic",))
    assert {it.node.id for it in r.items[:2]} == py_ids
    print("  retrieval + 10 ablation configs OK")


def test_compression():
    # realistic-size evidence: long summaries where truncation actually engages
    emb = EmbeddingBackend("stub")
    frame = MemoryFrame()
    long_docs = [
        ("python history", "python was created by guido van rossum and first released in 1991. "
         "python emphasizes code readability with its notable use of significant indentation. "
         "the language supports multiple programming paradigms including structured and functional styles. "
         "python 3.11 introduced faster execution and improved error messages."),
        ("python versions", "python version 3.11 was released in october 2022 with major speed improvements. "
         "python version 3.12 followed in october 2023 adding better f-string support. "
         "each release includes security fixes and standard library updates."),
        ("faiss", "faiss is a library for efficient similarity search and clustering of dense vectors. "
         "it contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in ram. "
         "faiss is written in c plus plus with wrappers for python."),
        ("cuda", "cuda is a parallel computing platform created by nvidia. "
         "it allows software developers to use a cuda-enabled graphics processing unit for general purpose processing. "
         "cuda provides a software layer giving direct access to the gpu virtual instruction set."),
        ("gguf format", "gguf is a file format for storing models for inference with ggml based executors. "
         "it contains all the information needed to load a model including weights and metadata. "
         "quantized gguf files reduce memory usage at a small cost in quality."),
        ("embedding models", "sentence embedding models map text into fixed size vectors. "
         "mini lm models are small and fast while keeping good semantic similarity quality. "
         "they are widely used for retrieval augmented generation pipelines."),
    ]
    vecs = emb.encode([d[1] for d in long_docs])
    for (key, text), vec in zip(long_docs, vecs):
        n = MemoryNode(concept=key, memory_type=MemoryType.FACT, raw_text=text,
                       summary=text, confidence=0.8)
        n.embedding = vec
        frame.add_node(n)
    vs = VectorStore(dim=vecs.shape[1])
    vs.add(vecs, [{"node_id": n.id} for n in frame.nodes.values()])
    gs = GraphStore()
    gs.build_from([n.to_row() for n in frame.nodes.values()], [])
    config = {"topology": {"max_rings": 5}, "retrieval": {"candidate_k": 12, "final_k": 6}}
    ret = MIRARetriever(frame, vs, gs, config)
    qvec = emb.encode(["python version release history"])[0]
    result = ret.retrieve("python version release history", qvec, final_k=6)
    raw_total = sum(est_tokens(i.node.raw_text) for i in result.items)
    ctx = compress(result, max_tokens=120)
    assert ctx.n_tokens <= 120, ctx.n_tokens
    assert ctx.units, "no provenance units"
    assert raw_total > 120, "test data too small for real compression"
    assert ctx.n_tokens < raw_total, (ctx.n_tokens, raw_total)  # genuinely compressed
    for u in ctx.units:
        assert "node_id" in u and "chunk_ids" in u
    print("  compression OK")


def test_conflicts():
    emb = EmbeddingBackend("stub")
    a = MemoryNode(concept="py version", memory_type=MemoryType.FACT,
                   raw_text="python version is 3.11", confidence=0.9)
    b = MemoryNode(concept="py version", memory_type=MemoryType.FACT,
                   raw_text="python version is 3.12", confidence=0.6)
    c = MemoryNode(concept="unrelated", memory_type=MemoryType.FACT,
                   raw_text="faiss is a vector library", confidence=0.9)
    assert detect_conflict(a, b)
    assert not detect_conflict(a, c)
    pairs = find_conflicts([a, b, c])
    assert (a.id, b.id) in pairs or (b.id, a.id) in pairs
    order = resolve([a, b])
    assert len(order) == 1 and len(list(order.values())[0]) == 2
    # higher confidence first
    first_id = list(order.values())[0][0]
    assert first_id == a.id
    print("  conflicts OK")


def test_updater_history_and_merge_block():
    tmp = tempfile.mkdtemp(prefix="mira_t6_")
    try:
        store = SQLiteStore(os.path.join(tmp, "u.db"))
        upd = MemoryUpdater(store)
        a = MemoryNode(concept="py", memory_type=MemoryType.FACT,
                       raw_text="python version is 3.11", confidence=0.9)
        b = MemoryNode(
            concept="py", memory_type=MemoryType.FACT,
            raw_text="python version is 3.12", confidence=0.6)
        upd.create(a)
        # merge contradiction → blocked, new node + contradicts edge
        result_id = upd.merge(a.id, b)
        assert result_id == b.id
        assert store.count_nodes() == 2
        assert any(e["relation_type"] == "contradicts" for e in store.all_edges())
        # non-conflicting merge → merges into a
        c = MemoryNode(concept="py", memory_type=MemoryType.FACT,
                       raw_text="python 3.11 released oct 2022", confidence=0.7)
        result_id = upd.merge(a.id, c)
        assert result_id == a.id
        loaded = MemoryNode.from_row(store.get_node(a.id))
        assert "oct 2022" in loaded.raw_text
        # history recorded
        hist = upd.history(a.id)
        actions = [h["action"] for h in hist]
        assert "create" in actions and "merge" in actions
        # invalidate/restore
        upd.invalidate(a.id, "superseded")
        assert store.get_node(a.id)["valid_until"] is not None
        upd.restore(a.id)
        assert store.get_node(a.id)["valid_until"] is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  updater history + merge-block OK")


if __name__ == "__main__":
    print("Phase 6 tests:")
    test_retrieval_and_ablations()
    test_compression()
    test_conflicts()
    test_updater_history_and_merge_block()
    print("ALL PASS")
