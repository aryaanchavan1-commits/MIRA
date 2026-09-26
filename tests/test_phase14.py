"""Phase 14 tests: Bio-NN-inspired activation dynamics + learned neural scorer.

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
from core.ranking import apply_weights
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


def _lif_config(**overrides):
    activation = {
        "mode": "lif_like",
        "ticks": 3,
        "leak": 0.5,
        "threshold": 0.8,
        "reset_voltage": 0.0,
        "refractory_ticks": 1,
        "top_k_neighbors": 0,
        "max_trace_events": 128,
    }
    activation.update(overrides)
    return {"activation": activation}


def test_lif_below_threshold_has_no_spike():
    frame, *_ = build_frame()
    act = SpreadingActivation(frame, _lif_config())
    assert act.activate({"n0": 0.5}) == []
    assert act.last_trace
    assert all(event["spike"] is False for event in act.last_trace)
    assert all(0.0 <= event["post_voltage"] <= 1.0
               for event in act.last_trace)


def test_lif_spike_reset_and_refractory_are_exact():
    frame, *_ = build_frame()
    act = SpreadingActivation(frame, _lif_config(
        ticks=4, threshold=0.8, reset_voltage=0.2, refractory_ticks=1,
    ))
    out = act.activate({"n0": 1.0})
    events = act.last_trace
    assert out == [("n0", 1.0)]
    assert [event["spike"] for event in events] == [True, False, False, False]
    assert events[0]["pre_voltage"] == 0.0
    assert events[0]["post_voltage"] == 0.2
    assert events[1]["pre_voltage"] == 0.2
    assert events[1]["post_voltage"] == 0.2  # refractory hold
    assert abs(events[2]["post_voltage"] - 0.1) < 1e-9
    assert abs(events[3]["post_voltage"] - 0.05) < 1e-9


def test_lif_is_deterministic_across_queries():
    frame, *_ = build_frame()
    cfg = _lif_config(ticks=3, top_k_neighbors=2)
    first = SpreadingActivation(frame, cfg)
    second = SpreadingActivation(frame, cfg)
    first_out = first.activate({"n0": 1.0, "n1": 0.7})
    second_out = second.activate({"n0": 1.0, "n1": 0.7})
    assert first_out == second_out
    assert first.last_trace == second.last_trace


def test_lif_propagates_only_bounded_top_neighbors_without_dense_matrix():
    frame = MemoryFrame()
    for index in range(7):
        frame.add_node(MemoryNode(id=f"n{index}", concept=f"n{index}"))
    for index in range(1, 7):
        frame.add_edge(MemoryEdge(
            source_id="n0", target_id=f"n{index}",
            weight=1.0 if index <= 2 else 0.1,
        ))
    act = SpreadingActivation(frame, _lif_config(
        ticks=2, leak=0.0, threshold=0.5, refractory_ticks=0,
        top_k_neighbors=2,
    ))
    out = act.activate({"n0": 1.0})
    assert {node for node, _ in out} == {"n0", "n1", "n2"}
    assert act._mat is None, "LIF mode must not build a dense adjacency matrix"
    assert all(len(row) <= 2 for row in act._neighbors)
    assert {event["node"] for event in act.last_trace} <= {"n0", "n1", "n2"}
    assert all(0.0 <= event["post_voltage"] <= 1.0
               for event in act.last_trace)


def test_continuous_mode_remains_compatible_and_trace_free():
    frame, n0, *_ = build_frame()
    n0.importance = 0.8
    act = SpreadingActivation(frame, {"activation": {
        "mode": "continuous", "hops": 0,
        "seed_semantic_weight": 0.2, "seed_importance_weight": 0.5,
    }})
    out = dict(act.activate({"n0": 1.0}))
    assert abs(out["n0"] - 0.6) < 1e-6
    assert act.last_trace == []


def test_lif_trace_attaches_to_retrieval_result():
    frame, n0, *_ = build_frame()
    emb = init_embeddings("stub")
    for node in frame.nodes.values():
        node.embedding = emb.encode([node.concept])[0]
    order = list(frame.nodes.values())
    vectors = np.stack([node.embedding for node in order])
    vs = VectorStore(dim=vectors.shape[1])
    vs.add(vectors, [{"node_id": node.id} for node in order])
    gs = GraphStore()
    gs.build_from([node.to_row() for node in order],
                  [edge.to_row() for edge in frame.edges])
    config = {
        "retrieval": {"candidate_k": 8, "final_k": 4, "max_hops": 3},
        **_lif_config(top_k_neighbors=2),
    }
    result = _retriever(frame, vs, gs, config).retrieve(
        "photosynthesis", n0.embedding.copy(), active_components=("activation",),
    )
    assert result.activation_trace
    required = {"tick", "node", "pre_voltage", "post_voltage", "threshold", "spike"}
    assert all(required <= set(event) for event in result.activation_trace)


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


def test_retrieval_component_iteration_is_order_independent():
    components = {"semantic": 0.7, "graph": 0.4, "activation": 0.2}
    weights = {"semantic": 0.5, "graph": 0.3, "activation": 0.2}
    assert apply_weights(components, weights, ["activation", "semantic", "graph"]) == \
        apply_weights(components, weights, ["graph", "semantic", "activation"])


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
