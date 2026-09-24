"""Workspace facade.

One object that owns the SQLite store, the live memory frame, the FAISS
index and the graph — what the UI and benchmarks talk to. Rebuilds the
frame from SQLite on startup; ingestion appends incrementally.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.auto_config import DATA_DIR, MODELS_DIR, ensure_dirs
from core.answer import AnswerPipeline
from core.memory import MemoryFrame, MemoryNode
from core.placement import place
from core.types import new_id
from models.embeddings import EmbeddingBackend
from storage.graph_store import GraphStore
from storage.sqlite_store import SQLiteStore
from storage.vector_store import VectorStore

logger = logging.getLogger("mira.workspace")

INDEX_PATH = os.path.join(DATA_DIR, "indexes", "main.faiss")


def index_path() -> str:
    return os.path.join(DATA_DIR, "indexes", "main.faiss")


class Workspace:
    def __init__(self, embeddings: EmbeddingBackend, llm=None,
                 config: Optional[Dict] = None):
        ensure_dirs()
        self.config = config or {}
        self.embeddings = embeddings
        self.llm = llm
        self.store = SQLiteStore(os.path.join(DATA_DIR, "mira.db"))
        self.frame = MemoryFrame()
        self.vs: Optional[VectorStore] = None
        self.gs = GraphStore()
        self._load_frame()
        # vectors live in FAISS, not SQLite — restore them after a restart
        if any(n.embedding is None for n in self.frame.nodes.values()):
            self.reembed_missing()
        self._ensure_lineage()
        self._load_or_build_index()

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    def _load_frame(self) -> None:
        for row in self.store.all_nodes():
            node = MemoryNode.from_row(row)
            self.frame.add_node(node)
        edges = self.store.all_edges()
        self.gs.build_from([n.to_row() for n in self.frame.nodes.values()],
                           [e.to_row() for e in self.frame.edges])
        # edges into the frame too (for placement/retrieval)
        from core.memory import MemoryEdge
        for e in edges:
            self.frame.add_edge(MemoryEdge(
                source_id=e["source_id"], target_id=e["target_id"],
                relation_type=e["relation_type"], weight=e["weight"],
                confidence=e["confidence"], provenance=e["provenance"]))
        logger.info("workspace frame loaded",
                    extra={"nodes": len(self.frame.nodes), "edges": len(edges)})

    def _load_or_build_index(self) -> None:
        nodes = [n for n in self.frame.nodes.values()]
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

    def _rebuild_index(self) -> None:
        nodes = [n for n in self.frame.nodes.values() if n.embedding is not None]
        if not nodes:
            dim = self.embeddings.dim
            self.vs = VectorStore(dim=dim, persist_path=index_path())
            self.vs.save()
            return
        vecs = np.stack([n.embedding for n in nodes])
        self.vs = VectorStore(dim=vecs.shape[1], persist_path=index_path())
        self.vs.add(vecs, [{"node_id": n.id} for n in nodes])
        self.vs.save()

    def reload(self) -> None:
        """Full reload from sqlite + embeddings for nodes missing vectors."""
        self.frame = MemoryFrame()
        self.gs = GraphStore()
        self._load_frame()
        if any(n.embedding is None for n in self.frame.nodes.values()):
            self.reembed_missing()
        self._ensure_lineage()
        self._load_or_build_index()

    def _ensure_lineage(self) -> None:
        """Re-embed everything if the embedding backend changed (e.g. hashing
        fallback -> MiniLM after a model download). Prevents mixing vectors
        from incompatible semantics in one index."""
        from models.embeddings_lineage import ensure_consistent
        info = self.embeddings.info() if self.embeddings else {}
        n = ensure_consistent(self, f"{info.get('backend', '?')}:{info.get('model', '?')}")
        if n:
            logger.info("re-embedded %d nodes for new embedding backend", n)

    # ------------------------------------------------------------------
    # ingestion
    # ------------------------------------------------------------------
    def _pipe_config(self) -> Dict[str, Any]:
        ing = self.config.get("ingestion", {}) or {}
        return {
            "chunk_size": ing.get("chunk_size", 800),
            "chunk_overlap": ing.get("chunk_overlap", 120),
            "placement_strategy": (self.config.get("topology", {}) or {})
                .get("placement_strategy", "hybrid_mira"),
            "radial": self.config.get("radial", {}),
        }

    def replace_all(self, strategy: Optional[str] = None) -> Dict[str, Any]:
        """Re-run mandala placement over ALL memories and persist (§11/§12).
        Used after strategy changes or to upgrade memories placed by an
        older placement algorithm."""
        from core.placement import place
        strat = strategy or (self.config.get("topology", {}) or {})
        if isinstance(strat, dict):
            strat = strat.get("placement_strategy", "hybrid_mira")
        info = place(self.frame, strat, config=self.config)
        for n in self.frame.nodes.values():
            self.store.upsert_node({**n.to_row(), "_action": "create"})
        logger.info("re-placed mandala", extra={"strategy": strat})
        return info

    def ingest_file(self, path: str, title: Optional[str] = None) -> Dict[str, Any]:
        from ingestion.pipeline import IngestionPipeline
        pipe = IngestionPipeline(self.store, self.embeddings, llm=self.llm,
                                 config=self._pipe_config())
        stats = pipe.ingest_file(path, title)
        self.reload()
        return stats

    def ingest_text(self, text: str, title: str = "pasted text") -> Dict[str, Any]:
        from ingestion.pipeline import IngestionPipeline
        pipe = IngestionPipeline(self.store, self.embeddings, llm=self.llm,
                                 config=self._pipe_config())
        stats = pipe.ingest_text(text, title)
        self.reload()
        return stats

    def delete_document(self, document_id: str) -> int:
        """Deletes document, its chunks, and nodes sourced to those chunks."""
        chunks = self.store.document_chunks(document_id)
        chunk_ids = {c["id"] for c in chunks}
        removed = 0
        for node in self.frame.nodes.values():
            if chunk_ids & set(node.source_ids):
                self.store.delete_node(node.id)
                removed += 1
        with self.store.tx() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
            conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
        self.reload()
        return removed

    # ------------------------------------------------------------------
    # asking
    # ------------------------------------------------------------------
    def ask(self, question: str, active_components=None,
            system_name: str = "mira", allow_web: Optional[bool] = None) -> Any:
        from core.agent import AgentPipeline
        pipeline = AnswerPipeline(
            self.frame, self.vs, self.gs, self.embeddings, llm=self.llm,
            config=self.config, doc_titles=self._doc_titles(),
            store=self.store,
        )
        pipeline.workspace_ingest = self.ingest_text  # agent can grow memory
        agent = AgentPipeline(pipeline, self.config, store=self.store,
                              doc_titles=self._doc_titles())
        return agent.ask(question, active_components=active_components,
                         allow_web=allow_web)

    def _doc_titles(self) -> Dict[str, str]:
        return {d["id"]: d["title"] for d in self.store.list_documents()}

    # ------------------------------------------------------------------
    # maintenance / info
    # ------------------------------------------------------------------
    def reembed_missing(self) -> int:
        """Embed any nodes that lack vectors (e.g. loaded from old db)."""
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
        by_type: Dict[str, int] = {}
        for n in self.frame.nodes.values():
            by_type[n.memory_type.value] = by_type.get(n.memory_type.value, 0) + 1
        return {
            "nodes": len(self.frame.nodes),
            "edges": len(self.frame.edges),
            "documents": len(self.store.list_documents()),
            "vectors": self.vs.size() if self.vs else 0,
            "by_type": by_type,
            "graph_connected_components": len(self.gs.connected_components()),
        }

    def close(self) -> None:
        try:
            if self.vs:
                self.vs.save()
            self.store.close()
        except Exception:
            pass
