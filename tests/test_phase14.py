"""Phase 14 tests: bio-inspired activation dynamics + learned neural scorer.

Also regression-guards the label-only concept-node suppression that keeps
structural nodes out of answer contexts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.activation import SpreadingActivation
from core.compression import compress
from core.memory import MemoryEdge, MemoryFrame, MemoryNode, MemoryType
from core.retrieval import MIRARetriever, ablation_configs
from core.retrieval import RetrievalResult
from core.neural import DeltaRuleScorer, FEATURES
from models.embeddings import init_embeddings
from storage.graph_store import GraphStore
from storage.vector_store import VectorStore


def build_frame() -> tuple:
    frame = MemoryFrame()
    n0 = MemoryNode(id="n0", concept="photosynthesis", memory_type=MemoryType.CONCEPT,
                    summary="plants convert sunlight into chemical energy",
                    source_ids=["c0"])
    n1 = MemoryNode(id="n1", concept="chlorophyll", memory_type=MemoryType.ENTITY,
                    summary="pigment that absorbs light in photosynthesis",
                    source_ids=["c1"])
    n2 = MemoryNode(id="n2", concept="light reactions", memory_type=MemoryType.CONCEPT,
                    summary="first stage of photosynthesis producing ATP",
                    source_ids=["c2"])
    n3 = MemoryNode(id="n3", concept="glucose synthesis", memory_type=MemoryType.FACT,
                    summary="sugar is produced in photosynthesis", source_ids=["c3"])
    for n in (n0, n1, n2, n3):
        frame.add_node(n)
    frame.add_edge(MemoryEdge(source_id="n0", target_id="n1", relation_type="has_part"))
    frame.add_edge(MemoryEdge(source_id="n0", target_id="n2", relation_type="has_stage"))
    frame.add_edge(MemoryEdge(source_id="n2", target_id="n3", relation_type="produces"))
    return frame, n0, n1, n2, n3


def _retriever(frame, vs, gs, config=None) -> MIRARetriever:
    cfg = config or {"retrieval": {"candidate_k": 8, "final_k": 4, "max_hops": 3}}
    return MIRARetriever(frame, vs, gs, cfg)


def test_activation_spreads_through_graph():
    frame, n0, n1, n2, n3 = build_frame()
    act = SpreadingActivation(frame, {"activation": {"hops": 3, "decay": 0.5}})
    out = dict(act.activate({"n0": 1.0}))
    # 2-hop associate n3 must receive energy; seed keeps the most
    assert "n3" in out, f"activation did not reach 2-hop node: {out.keys()}"
    assert out["n0"] >= out.get("n1", 0), "seed should retain most activation"
    assert all(v <= 1.0 for v in out.values()), "activation must be clipped to [0,1]"


def test_activation_empty_and_unknown():
    frame, *_ = build_frame()
    act = SpreadingActivation(frame)
    assert act.activate({}) == []
    assert act.activate({"missing": 1.0}) == []


def test_activation_component_in_retrieval():
    frame, n0, n1, n2, n3 = build_frame()
    emb = init_embeddings("stub")
    vecs = emb.encode([n.concept + " " + (n.summary or "") for n in frame.nodes.values()])
    for n, v in zip(frame.nodes.values(), vecs):
        n.embedding = v
    order = list(frame.nodes.values())
    mat = np.stack([n.embedding for n in order])
    vs = VectorStore(dim=mat.shape[1])
    vs.add(mat, [{"node_id": n.id} for n in order])
    gs = GraphStore()
    gs.build_from([n.to_row() for n in frame.nodes.values()],
                  [e.to_row() for e in frame.edges])
    r = _retriever(frame, vs, gs)
    q = emb.encode(["how do plants make energy from sunlight?"])[0]
    res = r.retrieve("how do plants make energy from sunlight?", q,
                     active_components=None, final_k=4)
    assert res.items, "retrieval returned nothing"
    if "activation" in res.items[0].components:
        vals = [it.components["activation"] for it in res.items]
        assert all(0.0 <= v <= 1.0 for v in vals), "activation out of [0,1]"
    # ablation: activation_only must be a legal set and runnable
    res2 = r.retrieve("how do plants make energy from sunlight?", q,
                      active_components=("activation",), final_k=4)
    assert res2.n_candidates >= 0
    assert "activation_only" in ablation_configs()
    assert "vector_activation" in ablation_configs()


def test_delta_rule_scorer_learns():
    rng = np.random.default_rng(0)
    # rule: semantic+activation relevant, others noise
    n = 200
    X = rng.random((n, len(FEATURES)))
    y = (X[:, 0] * 0.8 + X[:, 8] * 0.8 > 0.9).astype(float)
    if y.sum() < 5 or (1 - y).sum() < 5:
        y = (X[:, 0] > np.median(X[:, 0])).astype(float)
    s = DeltaRuleScorer(epochs=80)
    hist = s.fit(X, y)
    assert hist["final_loss"] < hist["first_loss"], "loss did not decrease"
    w = s.weights()
    assert abs(sum(w.values()) - 1.0) < 1e-6, "weights must normalize to 1"
    assert w["semantic"] > w["graph"], "semantic signal should outweigh noise feature"


def test_delta_rule_save_load_roundtrip(tmp="data/.test_neural.json"):
    s = DeltaRuleScorer(epochs=3)
    X = np.random.default_rng(1).random((20, len(FEATURES)))
    y = (X[:, 0] > 0.5).astype(float)
    s.fit(X, y)
    s.save(tmp)
    s2 = DeltaRuleScorer.load(tmp)
    assert s2 is not None and s2.trained
    assert np.allclose(s.w, s2.w)
    import os
    os.remove(tmp)


def test_label_only_nodes_excluded_from_context():
    """Regression: structural nodes with no real text must not appear as
    evidence units in the compressed context."""
    frame, n0, n1, n2, n3 = build_frame()
    # a label-only structural node: no summary, no raw_text, no sources
    label_only = MemoryNode(id="nX", concept="appearing 3x",
                            memory_type=MemoryType.CONCEPT)
    frame.add_node(label_only)
    from core.retrieval import RetrievedItem, RetrievalResult
    res = RetrievalResult(items=[
        RetrievedItem(node=n0, score=1.0),
        RetrievedItem(node=label_only, score=0.9),
    ])
    ctx = compress(res, max_tokens=256, target_ratio=0.6)
    assert "appearing 3x" not in ctx.text, \
        f"label-only node leaked into context: {ctx.text[:200]}"


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
