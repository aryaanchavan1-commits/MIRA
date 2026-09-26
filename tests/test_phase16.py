"""Phase 16 tests: agent routing (memory → web → parametric).

No network. The web path is exercised with a monkeypatched search.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.answer import Answer
from core.agent import AgentPipeline
from core.memory import MemoryFrame, MemoryNode, MemoryType
from models.embeddings import init_embeddings
from storage.graph_store import GraphStore
from storage.vector_store import VectorStore


def _memory_pipe():
    """A real AnswerPipeline over a tiny corpus with the stub embedder."""
    from core.answer import AnswerPipeline
    frame = MemoryFrame()
    n0 = MemoryNode(id="n0", concept="eiffel tower", memory_type=MemoryType.FACT,
                    summary="The Eiffel Tower is a lattice tower in Paris, France. "
                            "It was designed by Gustave Eiffel and completed in 1889.",
                    source_ids=["c0"])
    frame.add_node(n0)
    emb = init_embeddings("stub")
    vec = emb.encode([n0.concept + " " + n0.summary])[0]
    n0.embedding = vec
    vs = VectorStore(dim=vec.shape[0])
    vs.add(vec[None, :], [{"node_id": "n0"}])
    gs = GraphStore()
    pipe = AnswerPipeline(frame, vs, gs, emb, llm=None,
                          config={"retrieval": {"candidate_k": 4, "final_k": 2}})
    return pipe, emb, n0


def test_identity_route_is_case_insensitive_and_narrow():
    class Pipe:
        doc_titles = {}
        llm = None

        def __init__(self):
            self.calls = []

        def ask(self, question, active_components=None):
            self.calls.append(question)
            return Answer(text="ordinary answer", mode="no_evidence")

    pipe = Pipe()
    agent = AgentPipeline(pipe, {})
    expected = (
        "MIRA was made by Aryan Chavan; it is a local-first Bio-NN-inspired, "
        "mandala-based symbolic memory and retrieval research system, not a "
        "biological brain simulation."
    )
    for question in (
        "Who made MIRA?",
        "Who made you?",
        "who CrEaTeD mira?",
        "WHO BUILT THE MIRA SYSTEM?",
        "MIRA was made by whom?",
    ):
        answer = agent.ask(question, allow_web=True)
        assert answer.text == expected
        assert answer.agent_mode == "identity"
        assert answer.mode == "deterministic"
        assert not answer.memories
        assert not answer.sources
        assert "high confidence" in answer.confidence_note
        assert "no sources" in answer.confidence_note
    assert not pipe.calls

    for ordinary_question in (
        "Who built the Eiffel Tower?",
        "What is MIRA?",
        "Who built MIRA's retrieval graph?",
    ):
        ordinary = agent.ask(ordinary_question)
        assert ordinary.agent_mode != "identity"
    assert pipe.calls == [
        "Who built the Eiffel Tower?",
        "What is MIRA?",
        "Who built MIRA's retrieval graph?",
    ]


def test_strong_memory_routes_to_memory():
    pipe, emb, n0 = _memory_pipe()
    agent = AgentPipeline(pipe, {})
    ans = agent.ask("Who designed the Eiffel Tower?")
    assert ans.agent_mode == "memory", f"routed to {ans.agent_mode}"
    assert "Eiffel" in ans.text


def test_off_corpus_query_routes_parametric_with_note():
    pipe, emb, n0 = _memory_pipe()
    agent = AgentPipeline(pipe, {})
    ans = agent.ask("What is quantum chromodynamics?")
    # no web consent → parametric, labeled ungrounded
    assert ans.agent_mode == "parametric"
    assert "ungrounded" in (ans.confidence_note or "")
    # no memories attached: never fabricated citations
    assert not ans.memories


def test_parametric_without_llm_has_honest_message():
    pipe, emb, n0 = _memory_pipe()
    agent = AgentPipeline(pipe, {})
    ans = agent.ask("What is quantum chromodynamics?")
    assert "No local LLM" in ans.text or "Ingest documents" in ans.text


def test_web_escalation_when_consent_given(monkeypatch=None):
    pipe, emb, n0 = _memory_pipe()
    agent = AgentPipeline(pipe, {"offline": False,
                                   "web_search": {"enabled": False}})
    # web off and not consented → parametric, no escalation attempted
    ans = agent.ask("What is quantum chromodynamics?")
    assert ans.agent_mode == "parametric"

    # consent on: monkeypatch search+fetch to return a page about the query
    import tools.websearch as ws_mod
    orig_search = ws_mod.search
    orig_fetch = ws_mod.fetch_page_text
    calls = {"search": 0, "ingest": 0}
    pipe.workspace_ingest = lambda text, title="t", source_path="(inline)": calls.__setitem__(
        "ingest", calls["ingest"] + 1)
    ws_mod.search = lambda q, cfg, allow_web=False, backend="auto": {
        "backend": "stub", "results": [
            {"title": "QCD", "url": "https://example.com/qcd",
             "snippet": "quantum chromodynamics"}], "note": ""}
    ws_mod.fetch_page_text = lambda url, max_chars=12000, config=None, allow_web=None: (
        "Quantum chromodynamics is the theory of the strong interaction "
        "between quarks and gluons. " * 8)
    try:
        ans2 = agent.ask("What is quantum chromodynamics?", allow_web=True)
    finally:
        ws_mod.search = orig_search
        ws_mod.fetch_page_text = orig_fetch
    assert calls["ingest"] >= 1, "web path must ingest fetched pages"
    assert ans2.agent_mode == "web"
    # content match is asserted only via routing: the stub embedder is a
    # hashing fallback whose similarity is semantically weak by design —
    # with real MiniLM the fresh page outranks the corpus trivially
    assert ans2.mode in ("llm", "extractive") and ans2.memories


def test_web_reretrieval_preserves_active_components():
    from core.answer import _answer_metrics

    class Pipe:
        doc_titles = {}
        llm = None
        def workspace_ingest(self, text, title="web", source_path="(inline)"):
            return None


        def __init__(self):
            self.calls = []

        def ask(self, question, active_components=None):
            self.calls.append(active_components)
            return Answer(
                text="Fresh grounded evidence answers the question.",
                mode="extractive",
                memories=[{"id": "web", "concept": "web",
                           "components": {"semantic": 1.0}}],
                selected_evidence_ids=["web"],
                context_text="[1] Fresh grounded evidence answers the question.",
                metrics=_answer_metrics(n_candidates=1, n_retrieved=1,
                                        n_memories=1, context_tokens=8),
            )

    pipe = Pipe()
    agent = AgentPipeline(pipe, {"offline": False})
    weak = Answer(text="no evidence", mode="no_evidence",
                  selected_evidence_ids=[])
    with patch("tools.websearch.search", return_value={
            "results": [{"url": "https://example.test/page", "title": "page"}],
    }), patch("tools.websearch.fetch_page_text",
              return_value="Fresh grounded evidence " * 30):
        answer = agent._web_answer("question", weak,
                                   active_components=("semantic", "graph"))
    assert pipe.calls == [("semantic", "graph")]
    assert answer.selected_evidence_ids == ["web"]


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
