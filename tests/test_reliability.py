"""Focused regression tests for the reliability pass."""
from __future__ import annotations

import re
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.activation import SpreadingActivation
from core.answer import Answer, AnswerPipeline, _citation_error, _degenerate
from core.compression import compress, est_tokens
from core.memory import MemoryEdge, MemoryFrame, MemoryNode, MemoryType
from core.retrieval import MIRARetriever, RetrievedItem, RetrievalResult
from core.workspace import Workspace
from evaluation.metrics import context_tokens, token_f1
from ingestion.pipeline import IngestionPipeline
from models import llm as llm_module
from models.embeddings import EmbeddingBackend
from storage.graph_store import GraphStore
from storage.sqlite_store import SQLiteStore


@contextmanager
def _assert_raises(exception, match=None):
    try:
        yield
    except exception as exc:
        if match is not None:
            assert re.search(match, str(exc)), (
                f"exception message {str(exc)!r} does not match {match!r}"
            )
    else:
        raise AssertionError(f"expected {exception.__name__}")


class _Embeddings:
    dim = 2

    def encode(self, texts, batch_size=32, show_progress=False):
        return np.ones((len(texts), self.dim), dtype="float32")


class _VectorStore:
    def __init__(self, node_ids=()):
        self.node_ids = list(node_ids)
        self.dim = 2

    def search(self, query, k=8):
        return [(1.0, {"node_id": node_id})
                for node_id in self.node_ids[:k]]


class _RecordingLLM:
    available = True

    def __init__(self, n_ctx=256):
        self.n_ctx = n_ctx
        self.calls = []

    def chat(self, messages, max_tokens=512, temperature=0.2):
        self.calls.append((messages, max_tokens, temperature))
        return "This is a sufficiently grounded answer [1]."


def _answer_pipe(llm=None, text="A grounded sentence answers the question clearly."):
    frame = MemoryFrame()
    node = MemoryNode(
        id="n1", concept="grounded fact", memory_type=MemoryType.FACT,
        summary=text, raw_text=text,
    )
    node.embedding = np.ones(_Embeddings.dim, dtype="float32")
    frame.add_node(node)
    graph = GraphStore()
    graph.build_from([node.to_row()], [])
    return AnswerPipeline(
        frame, _VectorStore(["n1"]), graph, _Embeddings(), llm=llm,
        config={"context": {"max_tokens": 4096}},
    )


def test_direct_hash_backend_never_loads_a_model():
    attempted = []
    with patch.object(
        EmbeddingBackend, "_attempt_load",
        lambda self, devices: attempted.append(devices),
    ):
        for name in ("stub", "hashing"):
            backend = EmbeddingBackend(name)
            backend.info()
            backend.encode(["deterministic"])
    assert not attempted


def test_real_backend_encode_failure_is_not_hashed():
    class BrokenModel:
        def encode(self, *args, **kwargs):
            raise RuntimeError("broken encoder")

    backend = EmbeddingBackend("real-model")
    backend._load_attempted = True
    backend.backend_kind = "st"
    backend.dim = 2
    backend.model = BrokenModel()
    hash_calls = []
    with patch.object(backend, "_hash_encode", lambda texts: hash_calls.append(texts)):
        with _assert_raises(RuntimeError, "sentence-transformer encode failed"):
            backend.encode(["do not hash me"])
    assert not hash_calls
    assert "broken encoder" in backend.info()["error"]


def test_activation_applies_configured_seed_weights():
    frame = MemoryFrame()
    frame.add_node(MemoryNode(id="n", concept="seed", importance=0.8))
    activation = SpreadingActivation(frame, {"activation": {
        "hops": 0,
        "seed_semantic_weight": 0.2,
        "seed_importance_weight": 0.5,
    }})
    out = dict(activation.activate({"n": 1.0}))
    assert abs(out["n"] - 0.6) < 1e-6


def test_graph_parallel_edges_keep_stronger_metadata():
    graph = GraphStore()
    graph.add_edge("a", "b", "weak", weight=0.2, confidence=0.2,
                   provenance=["weak"])
    graph.add_edge("a", "b", "strong", weight=0.9, confidence=0.8,
                   provenance=["strong"])
    graph.add_edge("a", "b", "later", weight=0.1, confidence=0.99,
                   provenance=["later"])
    data = graph.edge_data("a", "b")
    assert data["weight"] == 0.9
    assert data["relation_type"] == "strong"
    assert data["confidence"] == 0.8
    assert data["provenance"] == ["strong"]
    assert graph.edges_of("a")[0]["relation_type"] == "strong"


def test_workspace_reload_rebinds_cached_agent():
    class Store:
        def __init__(self):
            self.nodes = []

        def all_nodes(self):
            return [dict(n.to_row(), embedding=n.embedding) for n in self.nodes]

        def all_edges(self):
            return []

        def list_documents(self):
            return []

    ws = Workspace.__new__(Workspace)
    ws.config = {}
    ws.embeddings = _Embeddings()
    ws.llm = None
    ws.store = Store()
    ws.frame = MemoryFrame()
    ws.gs = GraphStore()
    ws.vs = _VectorStore()
    ws._ensure_lineage = lambda: None
    ws._load_or_build_index = lambda: setattr(
        ws, "vs", _VectorStore(list(ws.frame.nodes)))
    ws._refresh_answer_pipeline()
    old_agent = ws.agent

    new_node = MemoryNode(
        id="n2", concept="new fact", memory_type=MemoryType.FACT,
        summary="newly ingested evidence", raw_text="newly ingested evidence",
    )
    new_node.embedding = np.ones(_Embeddings.dim, dtype="float32")
    ws.store.nodes = [new_node]
    ws.reload()

    assert old_agent._pipe.frame is ws.frame
    assert old_agent._pipe.retriever._node_index["n2"].id == new_node.id


def test_workspace_preserves_real_index_when_backend_degrades():
    class DegradedEmbeddings:
        dim = 2

        def info(self):
            return {"backend": "hashing", "model": "real-model"}

        def encode(self, texts, batch_size=32, show_progress=False):
            raise AssertionError("degraded backend must not re-embed")

    import models.embeddings_lineage as lineage

    ws = Workspace.__new__(Workspace)
    ws.config = {}
    ws.embeddings = DegradedEmbeddings()
    node = MemoryNode(id="real-node", concept="real")
    node.embedding = np.ones(2, dtype="float32")
    ws.frame = MemoryFrame()
    ws.frame.add_node(node)
    ws.gs = GraphStore()
    ws.vs = object()
    with patch.object(lineage, "recorded_model", lambda: "st:real-model"):
        ws._ensure_lineage()
        ws._load_or_build_index()

    assert ws.embedding_degraded is True
    assert "hashing" in ws.embedding_degraded_reason
    assert ws.vs is None
    ws.store = type("Store", (), {"list_documents": lambda self: []})()
    assert ws.stats()["embedding_degraded"] is True
    with _assert_raises(RuntimeError, "ingestion paused"):
        ws.ingest_text("must not enter the real index")
    retriever = MIRARetriever(ws.frame, None, ws.gs, {})
    assert retriever.retrieve("real", np.ones(2, dtype="float32")).items == []


def test_web_ingest_reload_is_visible_to_same_agent():
    class Store:
        def __init__(self):
            self.nodes = []

        def all_nodes(self):
            return [dict(n.to_row(), embedding=n.embedding) for n in self.nodes]

        def all_edges(self):
            return []

        def list_documents(self):
            return []

    ws = Workspace.__new__(Workspace)
    ws.config = {"offline": False}
    ws.embeddings = _Embeddings()
    ws.llm = None
    ws.store = Store()
    ws.frame = MemoryFrame()
    ws.gs = GraphStore()
    ws.vs = _VectorStore()
    ws._ensure_lineage = lambda: None
    ws._load_or_build_index = lambda: setattr(
        ws, "vs", _VectorStore(list(ws.frame.nodes)))

    def ingest(text, title="pasted text", source_path="(inline)"):
        node = MemoryNode(
            id="web", concept="web fact", memory_type=MemoryType.FACT,
            summary=text, raw_text=text,
        )
        node.embedding = np.ones(_Embeddings.dim, dtype="float32")
        ws.store.nodes = [node]
        ws.reload()
        return {"document_id": "web"}

    ws.ingest_text = ingest
    ws._refresh_answer_pipeline()
    import tools.websearch as websearch

    with patch.object(websearch, "search", lambda *a, **k: {
        "results": [{"title": "web", "url": "https://example.test/page"}],
    }), patch.object(
        websearch, "fetch_page_text",
        lambda *a, **k: "newly ingested web evidence " * 20,
    ):
        answer = ws.agent.ask("What is the web fact?", allow_web=True)
    assert answer.agent_mode == "web"
    assert "newly ingested web evidence" in answer.context_text


def test_agent_defers_hebbian_and_labels_no_llm_fallback():
    from core.agent import AgentPipeline

    class Store:
        def __init__(self):
            self.updates = []

        def update_edge_weight(self, source_id, target_id, weight):
            self.updates.append((source_id, target_id, weight))

    def make_pipe(store):
        frame = MemoryFrame()
        n1 = MemoryNode(id="n1", concept="first", memory_type=MemoryType.FACT,
                        summary="The first grounded fact.", importance=0.8)
        n2 = MemoryNode(id="n2", concept="second", memory_type=MemoryType.FACT,
                        summary="The second grounded fact.", importance=0.8)
        for node in (n1, n2):
            node.embedding = np.ones(2, dtype="float32")
            frame.add_node(node)
        frame.add_edge(MemoryEdge("n1", "n2", relation_type="supports"))
        graph = GraphStore()
        graph.build_from([n.to_row() for n in (n1, n2)],
                         [frame.edges[0].to_row()])
        return AnswerPipeline(
            frame, _VectorStore(["n1"]), graph, _Embeddings(), llm=None,
            config={"retrieval": {"candidate_k": 4, "final_k": 2},
                    "context": {"max_tokens": 256}, "memory": {"hebbian": True}},
            store=store,
        )

    weak_store = Store()
    weak = AgentPipeline(
        make_pipe(weak_store), {"agent": {"memory_gate_semantic": 1.1}}
    ).ask("grounded question")
    assert weak.agent_mode == "parametric"
    assert weak.mode == "no_evidence"
    assert not weak_store.updates

    accepted_store = Store()
    accepted = AgentPipeline(make_pipe(accepted_store), {}).ask("grounded question")
    assert accepted.agent_mode == "memory"
    assert accepted_store.updates

    identity_store = Store()
    identity = AgentPipeline(make_pipe(identity_store), {}).ask("Who made MIRA?")
    assert identity.agent_mode == "identity"
    assert not identity_store.updates


def test_empty_compressed_context_is_no_evidence():
    llm = _RecordingLLM()
    answer = _answer_pipe(llm=llm, text="").ask("question")
    assert answer.mode == "no_evidence"
    assert answer.context_text == ""
    assert not llm.calls


def test_degeneracy_detection_is_unicode_aware():
    assert not _degenerate("これは十分に長い回答です。")
    assert _degenerate("[1]")


def test_citation_error_signal_rejects_invalid_markers():
    assert _citation_error(" grounded [1]", 1) == 0
    assert _citation_error("grounded [2]", 1) == 1
    assert _citation_error("grounded [1][9]", 1) == 1


def test_answer_budgets_follow_actual_llm_window():
    llm = _RecordingLLM(n_ctx=256)
    answer = _answer_pipe(llm=llm).ask("What is the grounded fact?")
    assert answer.mode == "llm"
    assert answer.metrics["context_budget"] < 4096
    for messages, max_tokens, _temperature in llm.calls:
        assert max_tokens <= llm.n_ctx
        assert sum(est_tokens(m["content"]) for m in messages) + max_tokens <= llm.n_ctx


def test_parametric_completion_respects_window():
    from core.agent import AgentPipeline
    llm = _RecordingLLM(n_ctx=64)
    agent = AgentPipeline(_answer_pipe(llm=llm, text=""), {})
    agent._parametric("What is the answer?")
    assert all(max_tokens <= llm.n_ctx for _, max_tokens, _ in llm.calls)


def test_compression_uses_relevance_and_formatted_budget():
    node = MemoryNode(
        id="n1", concept="fact", memory_type=MemoryType.FACT,
        summary="irrelevant words here. quantum answer",
        raw_text="irrelevant words here. quantum answer",
    )
    result = RetrievalResult(query="quantum", items=[RetrievedItem(node, 1.0)])
    context = compress(result, max_tokens=6)
    assert context.n_tokens <= 6
    assert "quantum" in context.text


def test_sources_follow_compressed_context_units():
    frame = MemoryFrame()
    good = MemoryNode(
        id="good", concept="quantum", memory_type=MemoryType.FACT,
        summary="quantum answer", raw_text="quantum answer",
        source_ids=["chunk-good"], metadata={"document_id": "doc-good"},
    )
    unused = MemoryNode(
        id="unused", concept="irrelevant", memory_type=MemoryType.FACT,
        summary="irrelevant words here", raw_text="irrelevant words here",
        source_ids=["chunk-unused"], metadata={"document_id": "doc-unused"},
    )
    for node in (good, unused):
        node.embedding = np.ones(2, dtype="float32")
        frame.add_node(node)
    graph = GraphStore()
    graph.build_from([n.to_row() for n in (good, unused)], [])
    pipe = AnswerPipeline(
        frame, _VectorStore(["good", "unused"]), graph, _Embeddings(), llm=None,
        config={"context": {"max_tokens": 6}},
        doc_titles={"doc-good": "Good document", "doc-unused": "Unused document"},
    )
    result = RetrievalResult(
        query="quantum",
        items=[RetrievedItem(good, 1.0), RetrievedItem(unused, 0.9)],
        n_candidates=2,
    )
    pipe.retriever.retrieve = lambda *args, **kwargs: result
    answer = pipe.ask("quantum question")
    assert answer.sources == ["Good document (n.p.)"]


def test_invalid_llm_file_stops_retry_ladder():
    with tempfile.TemporaryDirectory() as directory:
        model_path = Path(directory) / "broken.gguf"
        model_path.write_text("not a gguf", encoding="utf-8")
        calls = []

        class BrokenLlama:
            def __init__(self, **kwargs):
                calls.append(kwargs)
                raise RuntimeError("Failed to load model from file")

        with patch.object(llm_module, "Llama", BrokenLlama), patch.object(
            llm_module, "_LLAMA_IMPORT_OK", True
        ):
            backend = llm_module.LLMBackend(str(model_path), n_ctx=256)
        assert not backend.available
        assert len(calls) == 1


def test_llm_backend_passes_configured_n_batch():
    with tempfile.TemporaryDirectory() as directory:
        model_path = Path(directory) / "fake.gguf"
        model_path.write_text("fake", encoding="utf-8")
        captured = {}

        class FakeLlama:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with patch.object(llm_module, "Llama", FakeLlama), patch.object(
            llm_module, "_LLAMA_IMPORT_OK", True
        ):
            llm_module.LLMBackend(str(model_path), n_ctx=256, n_batch=37)
        assert captured["n_batch"] == 37


def test_metrics_use_multisets_and_empty_context_is_zero():
    assert abs(token_f1("a a b", "a a a") - 2 / 3) < 1e-9
    assert context_tokens("") == 0


def test_accepted_trace_update_refreshes_activation_and_centrality_caches():
    class Store:
        def __init__(self):
            self.updates = []

        def update_edge_weight(self, source, target, weight):
            self.updates.append((source, target, weight))

    frame = MemoryFrame()
    for node_id in ("a", "b"):
        node = MemoryNode(id=node_id, concept=node_id,
                          memory_type=MemoryType.FACT,
                          summary=f"grounded fact {node_id}")
        node.embedding = np.ones(_Embeddings.dim, dtype="float32")
        frame.add_node(node)
    frame.add_edge(MemoryEdge("a", "b", relation_type="supports"))
    graph = GraphStore()
    graph.build_from([n.to_row() for n in frame.nodes.values()],
                     [e.to_row() for e in frame.edges])
    store = Store()
    pipe = AnswerPipeline(
        frame, _VectorStore(["a", "b"]), graph, _Embeddings(), llm=None,
        config={"memory": {"hebbian": True, "stdp": {
            "enabled": True, "decay": 0.0,
        }}, "activation": {"mode": "lif_like"}},
        store=store,
    )
    workspace = Workspace.__new__(Workspace)
    workspace.frame = frame
    workspace.gs = graph
    workspace.answer_pipeline = pipe
    pipe.workspace_cache_refresh = workspace.refresh_plasticity_caches
    pipe.retriever.activation._neighbors = [[] for _ in frame.nodes]
    pipe.retriever.activation._mat = np.zeros((2, 2), dtype="float32")
    old_centrality = pipe.retriever.centrality
    pipe._last_activation_trace = [
        {"tick": 0, "node": "a", "spike": True},
        {"tick": 1, "node": "b", "spike": True},
    ]
    pipe._last_result_paths = [["a", "b"]]
    pipe.reinforce_last_result()

    assert store.updates
    assert pipe.retriever.activation._neighbors is None
    assert pipe.retriever.activation._mat is None
    assert pipe.retriever.centrality is not old_centrality
    assert pipe.retriever.gs.edge_data("a", "b")["weight"] == frame.get_edge("a", "b").weight


def test_stdp_can_be_disabled_for_trace_updates():
    class Store:
        def __init__(self):
            self.updates = []

        def update_edge_weight(self, source, target, weight):
            self.updates.append((source, target, weight))

    frame = MemoryFrame()
    for node_id in ("a", "b"):
        node = MemoryNode(id=node_id, concept=node_id,
                          memory_type=MemoryType.FACT,
                          summary=f"grounded fact {node_id}")
        node.embedding = np.ones(_Embeddings.dim, dtype="float32")
        frame.add_node(node)
    frame.add_edge(MemoryEdge("a", "b", relation_type="supports"))
    graph = GraphStore()
    graph.build_from([n.to_row() for n in frame.nodes.values()],
                     [e.to_row() for e in frame.edges])
    store = Store()
    pipe = AnswerPipeline(
        frame, _VectorStore(["a", "b"]), graph, _Embeddings(), llm=None,
        config={"memory": {"hebbian": True, "stdp": False}},
        store=store,
    )
    pipe._last_activation_trace = [
        {"tick": 0, "node": "a", "spike": True},
        {"tick": 1, "node": "b", "spike": True},
    ]
    pipe._last_result_paths = [["a", "b"]]
    pipe.reinforce_last_result()
    assert not store.updates
    assert frame.get_edge("a", "b").weight == 1.0


def test_workspace_serializes_concurrent_chat_and_ingest():
    """Shared frame/index/SQLite state must not be entered concurrently."""
    active = 0
    maximum = 0
    state_lock = threading.Lock()

    class FakeAgent:
        def ask(self, question, active_components=None, allow_web=None):
            nonlocal active, maximum
            with state_lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.04)
            with state_lock:
                active -= 1
            return Answer(text="grounded", mode="extractive")

    class FakeIngestion:
        def __init__(self, store, embeddings, llm=None, config=None):
            pass

        def ingest_text(self, text, title="pasted text", source_path="(inline)"):
            nonlocal active, maximum
            with state_lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.04)
            with state_lock:
                active -= 1
            return {"document_id": "fake", "n_chunks": 1}

    ws = Workspace.__new__(Workspace)
    ws._lock = threading.RLock()
    ws.config = {}
    ws.embeddings = _Embeddings()
    ws.llm = None
    ws.store = object()
    ws.frame = MemoryFrame()
    ws.gs = GraphStore()
    ws.vs = None
    ws.answer_pipeline = None
    ws.embedding_degraded = False
    ws.agent = FakeAgent()
    ws.reload = lambda: None

    with patch("ingestion.pipeline.IngestionPipeline", FakeIngestion):
        with ThreadPoolExecutor(max_workers=2) as pool:
            chat = pool.submit(ws.ask, "question")
            ingest = pool.submit(ws.ingest_text, "document text")
            assert chat.result().text == "grounded"
            assert ingest.result()["document_id"] == "fake"
    assert maximum == 1


def test_failed_encode_rolls_back_document_chunks_and_nodes():
    class BrokenEmbeddings:
        dim = 2

        def encode(self, texts, batch_size=32, show_progress=False):
            raise RuntimeError("simulated encode failure")

    with tempfile.TemporaryDirectory() as directory:
        store = SQLiteStore(str(Path(directory) / "mira.db"))
        pipeline = IngestionPipeline(store, BrokenEmbeddings(), config={})
        try:
            pipeline.ingest_text(
                "A document with enough text to create chunks and memories. "
                "The encoder fails before persistence can complete.",
                title="rollback",
            )
        except RuntimeError as exc:
            assert "encode failure" in str(exc)
        else:
            raise AssertionError("broken encoder must fail ingestion")
        assert store.list_documents() == []
        assert store.all_nodes(include_inactive=True) == []
        assert store.all_edges() == []
        store.close()


def test_answer_grounding_uses_only_compressed_evidence():
    frame = MemoryFrame()
    good = MemoryNode(
        id="good", concept="quantum", memory_type=MemoryType.FACT,
        summary="Quantum evidence answers the question clearly.",
        raw_text="Quantum evidence answers the question clearly.",
    )
    excluded = MemoryNode(
        id="excluded", concept="unrelated", memory_type=MemoryType.FACT,
        summary="Unrelated text " * 20, raw_text="Unrelated text " * 20,
    )
    for node in (good, excluded):
        node.embedding = np.ones(_Embeddings.dim, dtype="float32")
        frame.add_node(node)
    graph = GraphStore()
    graph.build_from([node.to_row() for node in (good, excluded)], [])
    pipe = AnswerPipeline(
        frame, _VectorStore(["good", "excluded"]), graph, _Embeddings(),
        llm=None, config={"context": {"max_tokens": 32}},
    )
    result = RetrievalResult(
        query="quantum",
        items=[
            RetrievedItem(good, 1.0, {"semantic": 1.0}, ["good"]),
            RetrievedItem(excluded, 0.99, {"semantic": 0.99}, ["excluded"]),
        ],
        n_candidates=2,
        activation_trace=[
            {"node": "excluded", "spike": True},
            {"node": "good", "spike": True},
        ],
    )
    pipe.retriever.retrieve = lambda *args, **kwargs: result
    answer = pipe.ask("quantum question")
    assert answer.selected_evidence_ids == ["good"]
    assert [memory["id"] for memory in answer.memories] == ["good"]
    assert answer.metrics["n_memories"] == 1
    assert answer.metrics["n_excluded_candidates"] == 1
    assert pipe._last_activation_trace == [
        {"node": "good", "spike": True}
    ]


def test_explicit_no_evidence_model_reply_is_not_grounded():
    class NoEvidenceLLM(_RecordingLLM):
        def chat(self, messages, max_tokens=512, temperature=0.2):
            return "The memories do not contain enough information."

    answer = _answer_pipe(llm=NoEvidenceLLM(), text="A grounded sentence answers the question clearly.").ask(
        "question")
    assert answer.mode == "no_evidence"
    assert answer.selected_evidence_ids == []
    assert answer.memories == []
    assert answer.metrics["n_memories"] == 0


def test_explicit_final_k_keeps_selected_evidence_and_display_in_sync():
    frame = MemoryFrame()
    nodes = []
    for index in range(3):
        node = MemoryNode(
            id=f"n{index}", concept=f"fact {index}", memory_type=MemoryType.FACT,
            summary=f"Fact {index} contains enough grounded evidence for the question.",
            raw_text=f"Fact {index} contains enough grounded evidence for the question.",
        )
        node.embedding = np.ones(_Embeddings.dim, dtype="float32")
        frame.add_node(node)
        nodes.append(node)
    graph = GraphStore()
    graph.build_from([node.to_row() for node in nodes], [])
    pipe = AnswerPipeline(
        frame, _VectorStore(["n0", "n1", "n2"]), graph, _Embeddings(), llm=None,
        config={"retrieval": {"final_k": 3}, "context": {"max_tokens": 256}},
    )
    result = RetrievalResult(
        query="fact",
        items=[RetrievedItem(node, 1.0 - index * 0.01,
                              {"semantic": 1.0}, [node.id])
               for index, node in enumerate(nodes)],
        n_candidates=3,
    )
    pipe.retriever.retrieve = lambda *args, **kwargs: result
    answer = pipe.ask("fact", final_k=1)
    assert len(answer.selected_evidence_ids) == 1
    assert len(answer.memories) == 1
    assert answer.metrics["n_memories"] == 1
    assert answer.metrics["n_excluded_candidates"] == 2


def test_explicit_empty_selection_hides_candidates_and_metrics():
    answer = Answer(
        memories=[{"id": "candidate", "concept": "not retained"}],
        selected_evidence_ids=[],
        metrics={"n_candidates": 2, "n_retrieved": 2},
    )
    assert answer.memories == []
    assert answer.selected_evidence_ids == []
    assert answer.metrics["n_memories"] == 0
    assert answer.metrics["n_excluded_candidates"] == 2


def test_answer_benchmark_scores_retained_ids_only():
    from evaluation.benchmark import run_system

    class Embeddings:
        dim = 2

        def encode(self, texts, batch_size=32, show_progress=False):
            return np.ones((len(texts), self.dim), dtype="float32")

    def retrieve(question, qvec):
        return {
            "node_ids": ["excluded", "retained"],
            "selected_evidence_ids": ["retained"],
            "n_candidates": 2,
            "n_retrieved": 2,
            "context_text": "retained evidence",
            "answer": "retained answer",
            "latency_ms": 1.0,
        }

    result = run_system(
        "answered", retrieve, Embeddings(),
        [{"question": "q", "answer": "retained answer",
          "supporting_ids": ["excluded"]}], k=8,
    )
    row = result["rows"][0]
    assert row["selected_evidence_ids"] == ["retained"]
    assert row["candidate_ids"] == ["excluded", "retained"]
    assert row["retrieval_recall"] == 0.0
    assert row["mrr"] == 0.0


def test_plasticity_reward_uses_retained_path_only():
    class Store:
        def __init__(self):
            self.updates = []

        def update_edge_weight(self, source, target, weight):
            self.updates.append((source, target, weight))

    frame = MemoryFrame()
    seed = MemoryNode(id="seed", concept="seed", memory_type=MemoryType.CONCEPT,
                      summary="seed")
    retained = MemoryNode(
        id="retained", concept="retained", memory_type=MemoryType.FACT,
        summary="Retained evidence answers the question clearly.",
    )
    excluded = MemoryNode(
        id="excluded", concept="excluded", memory_type=MemoryType.FACT,
        summary="Unrelated filler words " * 20,
    )
    for node in (seed, retained, excluded):
        node.embedding = np.ones(_Embeddings.dim, dtype="float32")
        frame.add_node(node)
    frame.add_edge(MemoryEdge("seed", "retained", relation_type="supports"))
    frame.add_edge(MemoryEdge("seed", "excluded", relation_type="supports"))
    graph = GraphStore()
    graph.build_from([node.to_row() for node in (seed, retained, excluded)],
                     [edge.to_row() for edge in frame.edges])
    store = Store()
    pipe = AnswerPipeline(
        frame, _VectorStore(["retained", "excluded"]), graph, _Embeddings(),
        llm=None, config={"context": {"max_tokens": 12},
                          "memory": {"hebbian": True}}, store=store,
    )
    result = RetrievalResult(
        query="retained question",
        items=[
            RetrievedItem(retained, 1.0, {"semantic": 1.0}, ["seed", "retained"]),
            RetrievedItem(excluded, 0.99, {"semantic": 0.99}, ["seed", "excluded"]),
        ],
        n_candidates=2,
    )
    pipe.retriever.retrieve = lambda *args, **kwargs: result
    answer = pipe.ask("retained question")
    pipe.reinforce_last_result()
    assert answer.selected_evidence_ids == ["retained"]
    assert all("excluded" not in path for path in answer.path_labels)
    assert frame.get_edge("seed", "retained").weight > 1.0
    assert frame.get_edge("seed", "excluded").weight <= 1.0


def test_failed_index_publish_rolls_back_document_state():
    class FailingIndexPipeline(IngestionPipeline):
        def _append_vectors(self, frame):
            raise RuntimeError("simulated index failure")

    with tempfile.TemporaryDirectory() as directory:
        store = SQLiteStore(str(Path(directory) / "mira.db"))
        try:
            try:
                FailingIndexPipeline(store, _Embeddings(), config={}).ingest_text(
                    "A document with enough text to create durable chunks and nodes. "
                    "The index publisher fails before the document can commit.",
                    title="index-failure",
                )
            except RuntimeError as exc:
                assert "index failure" in str(exc)
            else:
                raise AssertionError("index failure must fail ingestion")
            assert store.list_documents() == []
            assert store.document_chunks("missing") == []
            assert store.all_nodes(include_inactive=True) == []
            assert store.all_edges() == []
            assert store._conn.execute("SELECT COUNT(*) FROM node_history").fetchone()[0] == 0
        finally:
            store.close()


def main() -> None:
    tests = (
        test_direct_hash_backend_never_loads_a_model,
        test_real_backend_encode_failure_is_not_hashed,
        test_activation_applies_configured_seed_weights,
        test_graph_parallel_edges_keep_stronger_metadata,
        test_workspace_reload_rebinds_cached_agent,
        test_workspace_preserves_real_index_when_backend_degrades,
        test_web_ingest_reload_is_visible_to_same_agent,
        test_agent_defers_hebbian_and_labels_no_llm_fallback,
        test_empty_compressed_context_is_no_evidence,
        test_degeneracy_detection_is_unicode_aware,
        test_citation_error_signal_rejects_invalid_markers,
        test_answer_budgets_follow_actual_llm_window,
        test_parametric_completion_respects_window,
        test_compression_uses_relevance_and_formatted_budget,
        test_sources_follow_compressed_context_units,
        test_invalid_llm_file_stops_retry_ladder,
        test_llm_backend_passes_configured_n_batch,
        test_metrics_use_multisets_and_empty_context_is_zero,
        test_accepted_trace_update_refreshes_activation_and_centrality_caches,
        test_stdp_can_be_disabled_for_trace_updates,
        test_workspace_serializes_concurrent_chat_and_ingest,
        test_failed_encode_rolls_back_document_chunks_and_nodes,
        test_answer_grounding_uses_only_compressed_evidence,
        test_explicit_no_evidence_model_reply_is_not_grounded,
        test_explicit_final_k_keeps_selected_evidence_and_display_in_sync,
        test_explicit_empty_selection_hides_candidates_and_metrics,
        test_answer_benchmark_scores_retained_ids_only,
        test_plasticity_reward_uses_retained_path_only,
        test_failed_index_publish_rolls_back_document_state,
    )
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {test.__name__}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)
    print("RELIABILITY TESTS PASS")


if __name__ == "__main__":
    main()
