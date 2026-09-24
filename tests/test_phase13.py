"""Phase 13 tests: dataset adapters + LLM-as-judge + answered-benchmark fn."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.datasets import load_any
from evaluation.judge import LLMJudge, judge_records


def test_adapters():
    with tempfile.TemporaryDirectory() as td:
        # custom
        p = Path(td) / "custom.json"
        p.write_text(json.dumps([
            {"question": "Q1?", "answer": "A1", "supporting_ids": ["n1"]},
            {"question": "Q2?", "answer": "A2"},
        ]), encoding="utf-8")
        recs = load_any(str(p), "custom")
        assert len(recs) == 2 and recs[0]["supporting_ids"] == ["n1"]
        # hotpotqa train format
        p2 = Path(td) / "hotpot.json"
        p2.write_text(json.dumps([
            {"question": "Who wrote X?", "answer": "Y",
             "supporting_facts": {"title": ["DocA", "DocB"], "sent": [0, 1]}},
        ]), encoding="utf-8")
        recs2 = load_any(str(p2), "hotpotqa")   # ws=None -> ids unresolved, still usable
        assert len(recs2) == 1 and recs2[0]["supporting_ids"] == []
        assert recs2[0]["answer"] == "Y"
    print("  dataset adapters OK")


class FakeLLM:
    """Scripted judge: returns the last integer in the prompt's expected slot."""

    def chat(self, messages, max_tokens=8, temperature=0.0):
        content = messages[0]["content"]
        if "grounded" in content.lower():       # faithfulness prompt
            return "4"
        return "5"                              # correctness prompt


class BrokenLLM:
    def chat(self, *a, **k):
        return "the answer is not a number"


def test_judge():
    j = LLMJudge(FakeLLM())
    assert j.correctness("q", "a", "ref") == 5
    assert j.faithfulness("a", "ev") == 4
    rows = [{"question": "q", "answer_text": "a", "reference_answer": "r"}]
    agg = judge_records(j, rows, answers={0: "ev"})
    assert agg["judge_correctness"] == 5 and agg["judge_faithfulness"] == 4
    assert agg["n_judged"] == 1 and rows[0]["judge_correctness"] == 5
    # broken judge -> errors counted, no fabricated scores
    agg2 = judge_records(LLMJudge(BrokenLLM()), [{"question": "q",
        "answer_text": "a", "reference_answer": "r"}], answers={})
    assert agg2["n_judged"] == 0 and agg2["judge_errors"] == 1
    assert "judge_correctness" not in agg2
    print("  judge scoring + failure handling OK")


def test_answer_retrieve_fn_shape():
    from models.embeddings import init_embeddings
    from core.memory import MemoryFrame, MemoryNode, MemoryType
    from core.placement import place
    from storage.graph_store import GraphStore
    from storage.vector_store import VectorStore
    import numpy as np

    emb = init_embeddings("stub")
    frame = MemoryFrame()
    n = MemoryNode(id="d1", concept="Eiffel Tower", memory_type=MemoryType.FACT,
                   summary="The Eiffel Tower is in Paris and was designed by Gustave Eiffel.",
                   raw_text="The Eiffel Tower is in Paris and was designed by Gustave Eiffel.")
    frame.add_node(n)
    place(frame, "hybrid_mira")
    n.embedding = emb.encode([n.summary])[0]
    vs = VectorStore(dim=emb.dim)
    vs.add(n.embedding.reshape(1, -1), [{"node_id": "d1"}])
    gs = GraphStore()
    gs.build_from([n.to_row()], [])

    class FakeWS:
        pass
    w = FakeWS()
    w.frame, w.vs, w.gs, w.embeddings = frame, vs, gs, emb
    w.llm = None
    w.config = {}
    w._doc_titles = lambda: {}

    from evaluation.benchmark import answer_retrieve_fn
    fn = answer_retrieve_fn(w, k=3)
    out = fn("Who designed the Eiffel Tower?", emb.encode(["Who designed the Eiffel Tower?"])[0])
    assert out["node_ids"] == ["d1"], "answered fn must expose retrieval ids"
    assert "Eiffel" in out["context_text"], "context_text present for faithfulness judging"
    assert out["answer"] is not None
    print("  answered-benchmark fn OK")


def main():
    test_adapters()
    test_judge()
    test_answer_retrieve_fn_shape()
    print("PHASE13 TESTS PASS")


if __name__ == "__main__":
    main()
