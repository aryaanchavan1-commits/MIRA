"""Phase 11 tests: metrics, benchmark harness, ablation runner, persistence."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core.memory import MemoryEdge, MemoryFrame, MemoryNode, MemoryType
from core.placement import place
from evaluation import report
from evaluation.benchmark import load_dataset, run_system
from evaluation.metrics import mrr, recall_at_k, token_f1
from models.embeddings import init_embeddings
from storage.graph_store import GraphStore
from storage.vector_store import VectorStore


def test_metrics():
    assert abs(token_f1("the cat sat", "the cat sat") - 1.0) < 1e-9
    assert token_f1("dogs bark", "the cat sat") == 0.0
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 3) == 1.0
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 1) == 0.5
    assert mrr(["x", "a"], {"a"}) == 0.5
    print("  metrics OK")


def build_tiny_world():
    frame = MemoryFrame()
    docs = {
        "d1": ("Eiffel Tower", "The Eiffel Tower is located in Paris, France. "
               "It was completed in 1889."),
        "d2": ("Louvre Museum", "The Louvre Museum is in Paris. It houses the "
               "Mona Lisa painting."),
        "d3": ("Berlin", "Berlin is the capital of Germany."),
    }
    emb = init_embeddings("stub")
    for did, (concept, text) in docs.items():
        n = MemoryNode(id=did, concept=concept, memory_type=MemoryType.FACT,
                       summary=text, raw_text=text)
        frame.add_node(n)
    place(frame, "hybrid_mira")
    vecs = emb.encode([n.summary for n in frame.nodes.values()])
    for n, v in zip(frame.nodes.values(), vecs):
        n.embedding = v
    vs = VectorStore(dim=vecs.shape[1])
    vs.add(vecs, [{"node_id": n.id} for n in frame.nodes.values()])
    gs = GraphStore()
    gs.build_from([n.to_row() for n in frame.nodes.values()], [])
    return frame, vs, gs, emb


def test_benchmark_and_persistence():
    frame, vs, gs, emb = build_tiny_world()

    # fake retriever fn with deterministic behavior
    def fn(question, qvec):
        hits = vs.search(qvec, k=2)
        return {"node_ids": [m["node_id"] for _, m in hits],
                "latency_ms": 1.0}

    records = [
        {"question": "Where is the Eiffel Tower?",
         "answer": "The Eiffel Tower is in Paris, France.",
         "supporting_ids": ["d1"]},
        {"question": "Where is the Mona Lisa?",
         "answer": "The Mona Lisa is in the Louvre in Paris.",
         "supporting_ids": ["d2"]},
    ]
    res = run_system("tiny", fn, emb, records, k=2)
    assert res["aggregate"]["n_questions"] == 2
    assert res["aggregate"]["retrieval_recall"] == 1.0, \
        "stub embeddings must retrieve the right doc for these queries"
    assert res["aggregate"]["mrr"] >= 0.5

    # persistence — real files on disk
    old = report.EXPERIMENTS_DIR
    try:
        with tempfile.TemporaryDirectory() as td:
            report.EXPERIMENTS_DIR = Path(td)
            exp_id = report.save_experiment(
                "unit-test", {"k": 2}, {"tiny": res}, store=None, seed=42)
            d = Path(td) / exp_id
            assert (d / "results.json").exists()
            assert (d / "results.csv").exists()
            assert (d / "summary.md").exists()
            assert (d / "config.json").exists()
            cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
            assert cfg["seed"] == 42 and "hardware" in cfg
    finally:
        report.EXPERIMENTS_DIR = old
    print("  benchmark + persistence OK")


def main():
    test_metrics()
    test_benchmark_and_persistence()
    print("PHASE11 TESTS PASS")


if __name__ == "__main__":
    main()
