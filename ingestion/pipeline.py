"""Full ingestion orchestrator (spec §16).

DOCUMENT → TEXT EXTRACTION → CLEANING → CHUNKING → ENTITY/CONCEPT/RELATION
EXTRACTION → SUMMARY → EMBEDDING → HIERARCHY → GRAPH → MANDALA PLACEMENT →
PERSISTENCE.

LLM extraction when available, deterministic fallbacks otherwise. Every
memory node carries provenance: source_ids → chunks → document → file.
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.memory import MemoryEdge, MemoryFrame, MemoryNode, MemoryType
from core.types import stable_hash
from ingestion.chunker import chunk_pages
from ingestion.extractors import (
    extract_concepts_fallback,
    extract_entities_fallback,
    extract_relations_fallback,
    summarize_fallback,
)
from ingestion.loaders import load_any, sanitize
from models.embeddings import EmbeddingBackend
from storage.sqlite_store import SQLiteStore
from storage.vector_store import VectorStore

logger = logging.getLogger("mira.pipeline")

# loaders module keeps the actual format parsers; spec name retained here
from ingestion import loaders as _loaders  # noqa: F401


class IngestionPipeline:
    def __init__(self, store: SQLiteStore, embeddings: EmbeddingBackend,
                 llm=None, config: Optional[Dict] = None):
        self.store = store
        self.embeddings = embeddings
        self.llm = llm
        self.config = config or {}

    # ------------------------------------------------------------------
    def ingest_file(self, path: str, title: Optional[str] = None) -> Dict[str, Any]:
        """Returns stats: document id, counts, deduplicated flag."""
        with open(path, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        existing = self.store.find_document_by_sha(sha)
        if existing:
            return {"document_id": existing["id"], "deduplicated": True,
                    "n_chunks": existing.get("n_chunks", 0)}

        pages = load_any(path)
        if not pages:
            raise ValueError(f"no text extracted from {os.path.basename(path)}")
        title = title or os.path.splitext(os.path.basename(path))[0]
        doc_id = self.store.add_document(
            title=title, source_path=path,
            file_type=os.path.splitext(path)[1].lower(), sha256=sha,
            n_chars=sum(len(t) for _, t in pages),
        )
        return self._ingest_pages(pages, doc_id, title)

    def ingest_text(self, text: str, title: str = "pasted text") -> Dict[str, Any]:
        sha = hashlib.sha256(sanitize(text).encode("utf-8")).hexdigest()
        existing = self.store.find_document_by_sha(sha)
        if existing:
            return {"document_id": existing["id"], "deduplicated": True,
                    "n_chunks": existing.get("n_chunks", 0)}
        doc_id = self.store.add_document(title=title, source_path="(inline)",
                                         file_type="txt", sha256=sha,
                                         n_chars=len(text))
        return self._ingest_pages([(1, sanitize(text))], doc_id, title)

    # ------------------------------------------------------------------
    def _ingest_pages(self, pages: List[Tuple[int, str]], doc_id: str,
                      title: str) -> Dict[str, Any]:
        chunks = chunk_pages(pages, chunk_size=int(self.config.get("chunk_size", 800)),
                             overlap=int(self.config.get("chunk_overlap", 120)))
        if not chunks:
            raise ValueError("chunking produced no chunks")
        return self._finalize_ingest(chunks, doc_id, title)

    def _finalize_ingest(self, chunks, doc_id, title) -> Dict[str, Any]:
        from core.placement import place
        concepts_acc: List[str] = []
        entities_acc: List[str] = []
        relations_acc: List[Dict[str, str]] = []
        frame = MemoryFrame()

        fact_nodes: List[MemoryNode] = []
        for seq, (page, text) in enumerate(chunks):
            chunk_id = self.store.add_chunk(doc_id, seq, text, page=page)
            ext = None
            if self.llm is not None and getattr(self.llm, "available", False):
                from ingestion.extractors import extract_with_llm
                ext = extract_with_llm(text, self.llm)
            ext = ext or {
                "concepts": extract_concepts_fallback([text]),
                "entities": extract_entities_fallback(text),
                "relations": extract_relations_fallback(text),
                "summary": summarize_fallback(text),
            }
            concepts_acc += ext.get("concepts", [])
            entities_acc += ext.get("entities", [])
            relations_acc += ext.get("relations", [])

            doc_node = MemoryNode(
                concept=title, memory_type=MemoryType.DOCUMENT,
                summary=ext.get("summary", ""), raw_text=text,
                importance=0.5, confidence=1.0,
                source_ids=[chunk_id],
                metadata={"page": page, "document_id": doc_id},
            )
            frame.add_node(doc_node)

            fact = MemoryNode(
                concept=ext.get("summary", "")[:60] or text[:60],
                memory_type=MemoryType.FACT,
                summary=ext.get("summary", ""), raw_text=text,
                parent_id=doc_node.id,
                importance=0.5, confidence=0.6,
                source_ids=[chunk_id],
                metadata={"page": page, "document_id": doc_id},
            )
            frame.add_node(fact)
            frame.add_edge(MemoryEdge(source_id=doc_node.id, target_id=fact.id,
                                      relation_type="evidence", weight=1.0,
                                      confidence=1.0))
            fact_nodes.append(fact)

        # concept/entity nodes dedup by lowercased concept string
        concept_counts: Dict[str, int] = {}
        for c in concepts_acc + entities_acc:
            key = c.lower()
            concept_counts[key] = concept_counts.get(key, 0) + 1
        concept_nodes: Dict[str, MemoryNode] = {}
        for key, count in concept_counts.items():
            node = MemoryNode(
                concept=key, memory_type=MemoryType.CONCEPT,
                # structural node: no content summary — mention stats live in
                # metadata; an empty summary keeps it out of answer contexts
                summary="",
                importance=min(1.0, 0.4 + 0.1 * count),
                confidence=min(0.9, 0.5 + 0.05 * count),
                source_ids=[], metadata={"mentions": count, "document_id": doc_id},
            )
            concept_nodes[key] = node
            frame.add_node(node)

        # embed everything in one batch
        all_nodes = list(frame.nodes.values())
        texts = [n.concept + " " + (n.summary or n.raw_text[:400]) for n in all_nodes]
        vecs = self.embeddings.encode(texts, show_progress=False)
        for n, v in zip(all_nodes, vecs):
            n.embedding = v

        # CONCEPT → FACT mention edges; relation triples → typed edges
        for fact in fact_nodes:
            low = fact.raw_text.lower()
            for key, cnode in concept_nodes.items():
                if key in low:
                    frame.add_edge(MemoryEdge(source_id=cnode.id, target_id=fact.id,
                                              relation_type="mentions",
                                              weight=1.0, confidence=0.7))
        for rel in relations_acc[:20]:
            s = concept_nodes.get(rel["subject"].lower()) or \
                self._get_or_create_rel_node(frame, rel["subject"], doc_id)
            o = concept_nodes.get(rel["object"].lower()) or \
                self._get_or_create_rel_node(frame, rel["object"], doc_id)
            frame.add_edge(MemoryEdge(
                source_id=s.id, target_id=o.id,
                relation_type=rel["relation"], weight=0.9, confidence=0.6,
            ))

        # mandala placement on the completed frame (config carries §12 radial weights)
        place(frame, self.config.get("placement_strategy", "hybrid_mira"),
              config=self.config)

        # persist nodes, edges, vectors
        for n in frame.nodes.values():
            self.store.upsert_node({**n.to_row(), "_action": "create"})
        for e in frame.edges:
            self.store.add_edge(e.source_id, e.target_id, e.relation_type,
                                e.weight, e.confidence, e.provenance)
        self._append_vectors(frame)

        self.store.set_document_chunk_count(doc_id, len(chunks))
        logger.info("ingested", extra={
            "document": doc_id, "chunks": len(chunks),
            "nodes": len(frame.nodes), "edges": len(frame.edges),
        })
        return {
            "document_id": doc_id, "deduplicated": False,
            "n_chunks": len(chunks), "n_nodes": len(frame.nodes),
            "n_edges": len(frame.edges),
            "concepts": list(concept_nodes.keys())[:12],
        }

    # ------------------------------------------------------------------
    def _get_or_create_rel_node(self, frame: MemoryFrame, name: str,
                                doc_id: str) -> MemoryNode:
        key = name.lower()
        node = next((n for n in frame.nodes.values()
                     if n.concept.lower() == key), None)
        if node is None:
            node = MemoryNode(concept=key, memory_type=MemoryType.ENTITY,
                              confidence=0.5, source_ids=[],
                              metadata={"document_id": doc_id})
            node.embedding = self.embeddings.encode([key])[0]
            frame.add_node(node)
        return node

    def _append_vectors(self, frame: MemoryFrame) -> None:
        """Append new node vectors to the persisted FAISS index."""
        from config.auto_config import DATA_DIR
        idx_dir = os.path.join(DATA_DIR, "indexes")
        os.makedirs(idx_dir, exist_ok=True)
        path = os.path.join(idx_dir, "main.faiss")
        nodes = [n for n in frame.nodes.values() if n.embedding is not None]
        if not nodes:
            return
        dim = len(nodes[0].embedding)
        if os.path.exists(path + ".meta.json"):
            vs = VectorStore.load(path)
            if vs.dim != dim:
                logger.warning("faiss dim mismatch (%s vs %s) — rebuilding index", vs.dim, dim)
                vs = VectorStore(dim=dim, persist_path=path)
        else:
            vs = VectorStore(dim=dim, persist_path=path)
        vs.add(np.stack([n.embedding for n in nodes]),
               [{"node_id": n.id} for n in nodes])
        vs.save()
