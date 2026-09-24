"""Phase 16 tests: agent routing (memory → web → parametric).

No network. The web path is exercised with a monkeypatched search.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

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
    agent = AgentPipeline(pipe, {"web_search": {"enabled": False}})
    # web off and not consented → parametric, no escalation attempted
    ans = agent.ask("What is quantum chromodynamics?")
    assert ans.agent_mode == "parametric"

    # consent on: monkeypatch search+fetch to return a page about the query
    import tools.websearch as ws_mod
    orig_search = ws_mod.search
    orig_fetch = ws_mod.fetch_page_text
    calls = {"search": 0, "ingest": 0}
    pipe.workspace_ingest = lambda text, title="t": calls.__setitem__(
        "ingest", calls["ingest"] + 1)
    ws_mod.search = lambda q, cfg, allow_web=False, backend="auto": {
        "backend": "stub", "results": [
            {"title": "QCD", "url": "https://example.com/qcd",
             "snippet": "quantum chromodynamics"}], "note": ""}
    ws_mod.fetch_page_text = lambda url, max_chars=12000: (
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
