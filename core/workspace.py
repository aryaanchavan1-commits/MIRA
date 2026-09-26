"""Workspace facade.

One object that owns the SQLite store, the live memory frame, the FAISS
index and the graph — what the UI and benchmarks talk to. Rebuilds the
frame from SQLite on startup; ingestion appends incrementally.
"""
from __future__ import annotations

import logging
import os
from threading import RLock
from typing import Any, Dict, Optional

import numpy as np

from config.auto_config import DATA_DIR, ensure_dirs
from core.affect import AffectiveState
from core.answer import AnswerPipeline
from core.memory import MemoryFrame, MemoryNode
from core.placement import place
from models.embeddings import EmbeddingBackend
from storage.graph_store import GraphStore
from storage.sqlite_store import SQLiteStore
from storage.vector_store import VectorStore

logger = logging.getLogger("mira.workspace")

INDEX_PATH = os.path.join(DATA_DIR, "indexes", "main.faiss")


def index_path() -> str:
    return os.path.join(DATA_DIR, "indexes", "main.faiss")


class Workspace:
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._lock = RLock()
        return instance

    def __init__(self, embeddings: EmbeddingBackend, llm=None,
                 config: Optional[Dict] = None):
        ensure_dirs()
        self._lock = RLock()
        self.config = config if isinstance(config, dict) else {}
        self.embeddings = embeddings
        self.llm = llm
        # Affect is transient, workspace-scoped bookkeeping.  Mandala memory
        # remains the sole authoritative knowledge/retrieval store.
        self.affect = AffectiveState()
        self.affect_state = self.affect
        self.embedding_degraded = False
        self.embedding_degraded_reason = ""
        self.embedding_backend = ""
        self.store = SQLiteStore(os.path.join(DATA_DIR, "mira.db"))
        self.frame = MemoryFrame()
        self.vs: Optional[VectorStore] = None
        self.gs = GraphStore()
        self._load_frame()
        # Check lineage before filling missing vectors: a transient hashing
        # fallback must not overwrite a real backend's persisted index.
        self._ensure_lineage()
        if (not getattr(self, "embedding_degraded", False)
                and any(n.embedding is None for n in self.frame.nodes.values())):
            self.reembed_missing()
        self._load_or_build_index()
        self._refresh_answer_pipeline()

    def _get_lock(self) -> RLock:
        lock = getattr(self, "_lock", None)
        if lock is None:
            lock = RLock()
            self._lock = lock
        return lock

    @property
    def lock(self) -> RLock:
        """The workspace operation lock shared by server and core callers."""
        return self._get_lock()

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    def _load_frame(self) -> None:
        for row in self.store.all_nodes():
            node = MemoryNode.from_row(row)
            self.frame.add_node(node)
        edges = self.store.all_edges()
        from core.memory import MemoryEdge
        for e in edges:
            self.frame.add_edge(MemoryEdge(
                source_id=e["source_id"], target_id=e["target_id"],
                relation_type=e["relation_type"], weight=e["weight"],
                confidence=e["confidence"], provenance=e["provenance"]))
        self.gs.build_from(
            [self.frame.nodes[node_id].to_row()
             for node_id in sorted(self.frame.nodes)],
            [edge.to_row() for edge in sorted(
                self.frame.edges,
                key=lambda item: (item.source_id, item.target_id,
                                  item.relation_type))],
        )
        logger.info("workspace frame loaded",
                    extra={"nodes": len(self.frame.nodes), "edges": len(edges)})

    def _load_or_build_index(self) -> None:
        if getattr(self, "embedding_degraded", False):
            # Never query a real index with hash query vectors. Keep the real
            # files untouched until the configured backend is available again.
            self.vs = None
            logger.warning("embedding retrieval disabled while backend is degraded")
            return
        nodes = sorted(self.frame.nodes.values(), key=lambda node: node.id)
        expected_ids = {n.id for n in nodes if n.embedding is not None}
        if os.path.exists(index_path() + ".meta.json"):
            try:
                self.vs = VectorStore.load(index_path())
                # stale when the indexed id set differs from live nodes (counts
                # can coincide across delete+reingest cycles)
                indexed_ids = {m.get("node_id") for m in self.vs.id_meta.values()}
                if indexed_ids != expected_ids:
                    self._rebuild_index()
            except Exception as exc:
                logger.warning("index load failed (%s) — rebuilding", exc)
                self._rebuild_index()
        else:
            self._rebuild_index()

    def _get_affect_state(self) -> AffectiveState:
        state = getattr(self, "affect", None)
        if not isinstance(state, AffectiveState):
            state = AffectiveState()
            self.affect = state
        self.affect_state = state
        return state

    def _refresh_answer_pipeline(self) -> None:
        """Rebind answer state after frame/index replacement.

        The agent keeps a pipeline reference while answering.  Updating that
        reference here makes a web ingest visible to the *same* request,
        instead of only to later requests that happen to construct a new one.
        """
        from core.agent import AgentPipeline
        from core.retrieval import MIRARetriever

        old_pipeline = getattr(self, "answer_pipeline", None)
        affect = self._get_affect_state()
        doc_titles = self._doc_titles()
        doc_sources = self._doc_sources()
        if isinstance(old_pipeline, AnswerPipeline):
            old_pipeline._lock = self._get_lock()
            old_pipeline.retriever = MIRARetriever(
                self.frame, self.vs, self.gs, self.config)
            old_pipeline.frame = self.frame
            old_pipeline.embeddings = self.embeddings
            old_pipeline.llm = self.llm
            old_pipeline.config = self.config
            old_pipeline.doc_titles = doc_titles
            old_pipeline.doc_sources = doc_sources
            old_pipeline.store = self.store
            old_pipeline.workspace_ingest = self.ingest_text
            # A reload starts a new consistency epoch.  Do not let plasticity
            # from a query answered before the reload leak into the next one.
            old_pipeline._last_result_paths = []
            old_pipeline._last_activation_trace = []
            pipeline = old_pipeline
        else:
            pipeline = AnswerPipeline(
                self.frame, self.vs, self.gs, self.embeddings, llm=self.llm,
                config=self.config, doc_titles=doc_titles,
                doc_sources=doc_sources, store=self.store,
                operation_lock=self._get_lock(),
            )
            pipeline.workspace_ingest = self.ingest_text
        pipeline.workspace_cache_refresh = self.refresh_plasticity_caches
        self.answer_pipeline = pipeline

        old_agent = getattr(self, "agent", None)
        if old_agent is not None and hasattr(old_agent, "_pipe"):
            # Keep the object currently serving a request alive, but point it
            # at the freshly loaded frame/index.
            old_agent._pipe = pipeline
            old_agent._lock = pipeline._lock
            old_agent.cfg = self.config
            old_agent.store = self.store
            old_agent.doc_titles = doc_titles
            old_agent.affect_state = affect
            old_agent.affect = affect
            self.agent = old_agent
        else:
            self.agent = AgentPipeline(pipeline, self.config, store=self.store,
                                       doc_titles=doc_titles,
                                       affect_state=affect)

    def refresh_plasticity_caches(self) -> None:
        with self._get_lock():
            self._refresh_plasticity_caches()

    def _refresh_plasticity_caches(self) -> None:
        """Rebuild graph and activation caches after an accepted update.

        Edge weights are mutated in the live frame and persisted by the
        plasticity pass.  The retriever keeps a centrality snapshot and a
        LIF neighbor frontier, so both must be rebuilt before the next query.
        """
        self.gs.build_from(
            [self.frame.nodes[node_id].to_row()
             for node_id in sorted(self.frame.nodes)],
            [edge.to_row() for edge in sorted(
                self.frame.edges,
                key=lambda item: (item.source_id, item.target_id,
                                  item.relation_type))],
        )
        pipeline = getattr(self, "answer_pipeline", None)
        retriever = getattr(pipeline, "retriever", None)
        if retriever is None:
            return
        retriever.frame = self.frame
        retriever.gs = self.gs
        if hasattr(retriever, "_node_index"):
            retriever._node_index = {
                node.id: node for node in sorted(
                    self.frame.nodes.values(), key=lambda item: item.id
                )
            }
        activation = getattr(retriever, "activation", None)
        if activation is not None:
            activation.frame = self.frame
            if hasattr(activation, "invalidate"):
                activation.invalidate()
        retriever.centrality = self.gs.centrality()

    def _rebuild_index(self) -> None:
        nodes = sorted(
            (n for n in self.frame.nodes.values() if n.embedding is not None),
            key=lambda node: node.id,
        )
        if not nodes:
            dim = self.embeddings.dim
            self.vs = VectorStore(dim=dim, persist_path=index_path())
            save_atomic = getattr(self.vs, "save_atomic", self.vs.save)
            save_atomic()
            return
        vecs = np.stack([n.embedding for n in nodes])
        self.vs = VectorStore(dim=vecs.shape[1], persist_path=index_path())
        self.vs.add(vecs, [{"node_id": n.id} for n in nodes])
        save_atomic = getattr(self.vs, "save_atomic", self.vs.save)
        save_atomic()

    def reload(self) -> None:
        with self._get_lock():
            self._reload()

    def _reload(self) -> None:
        """Full reload from sqlite + embeddings for nodes missing vectors."""
        self.frame = MemoryFrame()
        self.gs = GraphStore()
        self._load_frame()
        self._ensure_lineage()
        if (not getattr(self, "embedding_degraded", False)
                and any(n.embedding is None for n in self.frame.nodes.values())):
            self.reembed_missing()
        self._load_or_build_index()
        self._refresh_answer_pipeline()

    def _ensure_lineage(self) -> None:
        """Keep the persisted index aligned with the active embedder."""
        from models.embeddings_lineage import ensure_consistent, recorded_model

        info = self.embeddings.info() if self.embeddings else {}
        backend = info.get("backend", "?")
        model = info.get("model", "?")
        active_name = f"{backend}:{model}"
        previous = recorded_model()
        self.embedding_backend = str(backend)
        self.embedding_degraded = bool(
            backend == "hashing" and previous
            and not previous.startswith("hashing:")
        )
        self.embedding_degraded_reason = (
            f"active embedding backend degraded to hashing; preserving "
            f"previous real lineage {previous!r}"
            if self.embedding_degraded else ""
        )
        if self.embedding_degraded:
            logger.warning(self.embedding_degraded_reason)
        n = ensure_consistent(self, active_name)
        if n:
            logger.info("re-embedded %d nodes for new embedding backend", n)

    # ------------------------------------------------------------------
    # ingestion
    # ------------------------------------------------------------------
    def _pipe_config(self) -> Dict[str, Any]:
        ing = self.config.get("ingestion", {}) or {}
        if not isinstance(ing, dict):
            ing = {}
        topology = self.config.get("topology", {}) or {}
        if not isinstance(topology, dict):
            topology = {}
        return {
            "chunk_size": ing.get("chunk_size", 800),
            "chunk_overlap": ing.get("chunk_overlap", 120),
            "placement_strategy": topology.get("placement_strategy", "hybrid_mira"),
            "radial": (self.config.get("radial")
                       or self.config.get("radial_distance") or {}),
        }

    def _restore_live_references(self, frame: MemoryFrame, gs: GraphStore,
                                 vs: Optional[VectorStore],
                                 pipeline=None, agent=None) -> None:
        """Put the pre-ingest live objects back if a rollback reload fails."""
        from core.retrieval import MIRARetriever

        self.frame = frame
        self.gs = gs
        self.vs = vs
        self.answer_pipeline = pipeline
        self.agent = agent
        if isinstance(pipeline, AnswerPipeline):
            pipeline._lock = self._get_lock()
            pipeline.retriever = MIRARetriever(
                frame, vs, gs, getattr(self, "config", {}))
            pipeline.frame = frame
            pipeline.store = self.store
            pipeline.embeddings = self.embeddings
            pipeline.llm = self.llm
            pipeline.config = getattr(self, "config", {})
            pipeline.doc_titles = getattr(pipeline, "doc_titles", {})
            pipeline.doc_sources = self._doc_sources()
            pipeline.workspace_ingest = self.ingest_text
            pipeline.workspace_cache_refresh = self.refresh_plasticity_caches
            pipeline._last_result_paths = []
            pipeline._last_activation_trace = []
            self.answer_pipeline = pipeline
        if agent is not None and pipeline is not None:
            agent._pipe = pipeline
            agent._lock = getattr(pipeline, "_lock", self._get_lock())
            agent.cfg = getattr(self, "config", {})
            agent.store = self.store
            agent.doc_titles = getattr(pipeline, "doc_titles", {})
            self.agent = agent

    def replace_all(self, strategy: Optional[str] = None) -> Dict[str, Any]:
        """Re-run mandala placement over ALL memories and persist (§11/§12)."""
        with self._get_lock():
            config = getattr(self, "config", None)
            if not isinstance(config, dict):
                config = {}
                self.config = config
            topology = config.get("topology")
            old_topology = dict(topology) if isinstance(topology, dict) else None
            old_frame = self.frame
            old_graph = getattr(self, "gs", None)
            snapshot = {
                node.id: (node.ring, node.sector, node.depth,
                          node.radial_distance)
                for node in self.frame.nodes.values()
            }
            try:
                strat = strategy or (config.get("topology", {}) or {})
                if isinstance(strat, dict):
                    strat = strat.get("placement_strategy", "hybrid_mira")
                info = place(self.frame, strat, config=config)
                applied_strategy = info.get("strategy", strat)
                placements = [
                    {"id": node.id, "ring": node.ring, "sector": node.sector,
                     "depth": node.depth, "radial_distance": node.radial_distance}
                    for node in sorted(self.frame.nodes.values(),
                                       key=lambda item: item.id)
                ]
                update_placements = getattr(self.store, "update_placements", None)
                if callable(update_placements):
                    update_placements(placements)
                else:
                    for placement in placements:
                        node = self.frame.nodes[placement["id"]]
                        self.store.upsert_node({**node.to_row(), "_action": "update"})
                if not isinstance(topology, dict):
                    topology = {}
                    config["topology"] = topology
                topology["placement_strategy"] = applied_strategy
                logger.info("re-placed mandala", extra={"strategy": applied_strategy})
                return info
            except Exception:
                self.frame = old_frame
                self.gs = old_graph
                for node_id, values in snapshot.items():
                    node = self.frame.nodes.get(node_id)
                    if node is not None:
                        (node.ring, node.sector, node.depth,
                         node.radial_distance) = values
                if old_topology is None:
                    config.pop("topology", None)
                else:
                    config["topology"] = old_topology
                try:
                    self.reload()
                except Exception:
                    logger.exception("workspace reload after placement failure failed")
                    self.frame = old_frame
                    self.gs = old_graph
                raise

    def _reload_after_ingest(self, stats: Dict[str, Any], old_state) -> None:
        try:
            self.reload()
        except Exception:
            document_id = stats.get("document_id") if isinstance(stats, dict) else None
            try:
                cleanup = getattr(self.store, "delete_document_cascade", None)
                if document_id and callable(cleanup):
                    cleanup(document_id)
                self.reload()
            except Exception:
                logger.exception("workspace reload compensation after ingestion failed")
                self._restore_live_references(*old_state)
            raise

    def ingest_file(self, path: str, title: Optional[str] = None) -> Dict[str, Any]:
        with self._get_lock():
            if getattr(self, "embedding_degraded", False):
                raise RuntimeError("ingestion paused while embedding backend is degraded")
            from ingestion.pipeline import IngestionPipeline
            old_state = (self.frame, self.gs, self.vs,
                         getattr(self, "answer_pipeline", None),
                         getattr(self, "agent", None))
            pipe = IngestionPipeline(self.store, self.embeddings, llm=self.llm,
                                     config=self._pipe_config())
            try:
                stats = pipe.ingest_file(path, title)
            except Exception:
                # The pipeline removes its staged SQLite rows.  Reloading also
                # discards any in-memory/index residue from a failed append.
                try:
                    self.reload()
                except Exception:
                    logger.exception("workspace reload after failed file ingest failed")
                    self._restore_live_references(*old_state)
                raise
            self._reload_after_ingest(stats, old_state)
            return stats

    def ingest_text(self, text: str, title: str = "pasted text",
                    source_path: str = "(inline)") -> Dict[str, Any]:
        with self._get_lock():
            if getattr(self, "embedding_degraded", False):
                raise RuntimeError("ingestion paused while embedding backend is degraded")
            from ingestion.pipeline import IngestionPipeline
            old_state = (self.frame, self.gs, self.vs,
                         getattr(self, "answer_pipeline", None),
                         getattr(self, "agent", None))
            pipe = IngestionPipeline(self.store, self.embeddings, llm=self.llm,
                                     config=self._pipe_config())
            try:
                stats = pipe.ingest_text(text, title, source_path=source_path)
            except Exception:
                try:
                    self.reload()
                except Exception:
                    logger.exception("workspace reload after failed text ingest failed")
                    self._restore_live_references(*old_state)
                raise
            self._reload_after_ingest(stats, old_state)
            return stats

    def delete_document(self, document_id: str) -> int:
        """Delete a document and every node/edge created from its chunks."""
        with self._get_lock():
            cleanup = getattr(self.store, "delete_document_cascade", None)
            if callable(cleanup):
                removed = cleanup(document_id)
            else:
                # Compatibility for lightweight stores used by embedders/tests.
                chunks = self.store.document_chunks(document_id)
                chunk_ids = {c["id"] for c in chunks}
                removed = sum(
                    1 for node in self.frame.nodes.values()
                    if chunk_ids.intersection(node.source_ids)
                )
                with self.store.tx() as conn:
                    conn.execute("DELETE FROM chunks WHERE document_id=?",
                                 (document_id,))
                    conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
            self.reload()
            return removed

    # ------------------------------------------------------------------
    # asking
    # ------------------------------------------------------------------
    def ask(self, question: str, active_components=None,
            system_name: str = "mira", allow_web: Optional[bool] = None) -> Any:
        with self._get_lock():
            if getattr(self, "agent", None) is None:
                self._refresh_answer_pipeline()
            return self.agent.ask(question, active_components=active_components,
                                  allow_web=allow_web)

    def _doc_titles(self) -> Dict[str, str]:
        with self._get_lock():
            return {d["id"]: d["title"] for d in self.store.list_documents()}

    def _doc_sources(self) -> Dict[str, str]:
        with self._get_lock():
            return {d["id"]: d.get("source_path") or ""
                    for d in self.store.list_documents()}

    # ------------------------------------------------------------------
    # maintenance / info
    # ------------------------------------------------------------------
    def reembed_missing(self) -> int:
        with self._get_lock():
            return self._reembed_missing()

    def _reembed_missing(self) -> int:
        """Embed any nodes that lack vectors (e.g. loaded from old db)."""
        if getattr(self, "embedding_degraded", False):
            return 0
        missing = [n for n in self.frame.nodes.values() if n.embedding is None]
        if not missing:
            return 0
        texts = [n.concept + " " + (n.summary or n.raw_text[:400]) for n in missing]
        vecs = self.embeddings.encode(texts)
        for n, v in zip(missing, vecs):
            n.embedding = v
        self._rebuild_index()
        return len(missing)

    def stats(self) -> Dict[str, Any]:
        with self._get_lock():
            return self._stats()

    def _stats(self) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}
        for n in self.frame.nodes.values():
            by_type[n.memory_type.value] = by_type.get(n.memory_type.value, 0) + 1
        return {
            "nodes": len(self.frame.nodes),
            "edges": len(self.frame.edges),
            "documents": len(self.store.list_documents()),
            "vectors": self.vs.size() if self.vs else 0,
            "embedding_backend": getattr(self, "embedding_backend", ""),
            "embedding_degraded": bool(getattr(self, "embedding_degraded", False)),
            "embedding_degraded_reason": getattr(self, "embedding_degraded_reason", ""),
            "by_type": by_type,
            "graph_connected_components": len(self.gs.connected_components()),
        }

    def close(self) -> None:
        with self._get_lock():
            try:
                if self.vs:
                    save_atomic = getattr(self.vs, "save_atomic", self.vs.save)
                    save_atomic()
                self.store.close()
            except Exception:
                pass
