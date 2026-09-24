"""Query → answer pipeline (spec §20/§27).

USER QUERY → QUERY EMBEDDING → RETRIEVAL → CANDIDATE MERGE → RERANK →
EVIDENCE SELECTION → CONTEXT COMPRESSION → SMALL LLM → ANSWER.

The answer object is auditable: retrieved memories with scores, the
retrieval path (node labels, not hidden chain-of-thought), and source
document/page/chunk references. No hidden CoT is ever exposed — only the
retrieval path and evidence (§27).
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.compression import CompressedContext, compress
from core.memory import MemoryFrame, MemoryNode
from core.retrieval import MIRARetriever, RetrievalResult
from models.embeddings import EmbeddingBackend
from storage.graph_store import GraphStore
from storage.vector_store import VectorStore

logger = logging.getLogger("mira.answer")


@dataclass
class SourceRef:
    document_title: str = ""
    page: Optional[int] = None
    chunk_id: str = ""

    def label(self) -> str:
        loc = f"p.{self.page}" if self.page else "n.p."
        return f"{self.document_title} ({loc})" if self.document_title else loc


@dataclass
class Answer:
    text: str = ""
    mode: str = "extractive"            # llm | extractive | no_evidence
    agent_mode: str = "memory"          # memory | web | parametric (routing origin)
    memories: List[Dict[str, Any]] = field(default_factory=list)   # ranked, with scores
    path_labels: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    confidence_note: str = ""
    context_text: str = ""              # compressed evidence fed to the LLM (for faithfulness judging)


class AnswerPipeline:
    def __init__(self, frame: MemoryFrame, vector_store: VectorStore,
                 graph_store: GraphStore, embeddings: EmbeddingBackend,
                 llm=None, config: Optional[Dict] = None,
                 doc_titles: Optional[Dict[str, str]] = None,
                 store=None):
        self.retriever = MIRARetriever(frame, vector_store, graph_store, config or {})
        self.frame = frame
        self.embeddings = embeddings
        self.llm = llm
        self.config = config or {}
        self.doc_titles = doc_titles or {}
        self.store = store  # optional persistence for Hebbian consolidation
        # injected by the Workspace so the agent layer can grow memory from
        # the web without a circular import
        self.workspace_ingest = None

    def _sources_for(self, node: MemoryNode) -> List[SourceRef]:
        out: List[SourceRef] = []
        for chunk_id in node.source_ids[:3]:
            title = self.doc_titles.get(node.metadata.get("document_id", ""), "")
            out.append(SourceRef(document_title=title,
                                 page=node.metadata.get("page"),
                                 chunk_id=chunk_id))
        return out

    def ask(self, question: str, active_components=None,
            max_context_tokens: Optional[int] = None,
            system_name: str = "mira") -> Answer:
        t0 = time.perf_counter()
        qvec = self.embeddings.encode([question])[0]
        result = self.retriever.retrieve(question, qvec,
                                         active_components=active_components)
        if not result.items:
            return Answer(text="The memories do not contain enough information.",
                          mode="no_evidence",
                          metrics={"latency_ms": round((time.perf_counter() - t0) * 1000, 1)})

        ctx_cfg = self.config.get("context", {})
        budget = max_context_tokens or parse_budget(self.config)
        ctx = compress(result, max_tokens=budget,
                       target_ratio=float(ctx_cfg.get("compression_target_ratio", 0.6)))

        # Hebbian consolidation (slow biological timescale): edges along
        # successful multi-hop retrieval paths strengthen; everything decays
        # slightly. Conservative, bounded, persisted.
        if (self.store is not None and result.paths
                and (self.config.get("memory", {}) or {}).get("hebbian", True)):
            from core.hebbian import reinforce
            try:
                reinforce(self.frame, result.paths, store=self.store)
            except Exception as exc:
                logger.warning("hebbian consolidation failed: %s", exc)

        # answer generation
        if self.llm is not None and self.llm.available and ctx.text.strip():
            from models.llm import answer_prompt
            prompt = answer_prompt(ctx.text, question)
            sysmsg = ("You are a careful research assistant. Answer only from "
                      "the provided excerpts and cite their numbers.")
            # greedy first: deterministic, grounded — best mode for small models
            text = self.llm.chat(
                [{"role": "system", "content": sysmsg},
                 {"role": "user", "content": prompt}],
                max_tokens=min(512, budget), temperature=0.0,
            )
            if _degenerate(text):
                # strategy 2: explicit sentence-first instruction. Small models
                # sometimes stop right after emitting a citation marker.
                retry = (f"Context:\n{ctx.text}\n\nQuestion: {question}\n\n"
                         "Write one complete sentence that answers the question "
                         "using only the context, then append the excerpt numbers "
                         "you used in brackets like [1]. If the context lacks the "
                         "answer, write exactly: The memories do not contain "
                         "enough information.")
                text = self.llm.chat(
                    [{"role": "user", "content": retry}],
                    max_tokens=min(512, budget), temperature=0.2,
                )
            if _degenerate(text):
                text = ""  # fall through to extractive
            mode = "llm" if text else "extractive"
            if not text:
                from models.llm import extractive_answer
                text = extractive_answer(question, ctx.text, ctx.units)["answer"]
        else:
            from models.llm import extractive_answer
            ex = extractive_answer(question, ctx.text, ctx.units)
            text, mode = ex["answer"], "extractive"

        # provenance: map units back to document/page via chunk table
        sources: List[str] = []
        seen: set = set()
        memories: List[Dict[str, Any]] = []
        for item in result.items:
            n = item.node
            memories.append({
                "id": n.id,
                "concept": n.concept, "type": n.memory_type.value,
                "ring": n.ring, "sector": n.sector, "score": round(item.score, 4),
                "components": {k: round(v, 3) for k, v in item.components.items()},
            })
            for ref in self._sources_for(n):
                label = ref.label()
                if label not in seen:
                    seen.add(label)
                    sources.append(label)

        total_ms = round((time.perf_counter() - t0) * 1000, 1)
        ans = Answer(
            text=text, mode=mode, memories=memories[: self.retriever.final_k],
            path_labels=result.path_labels()[:5],
            sources=sources[:6],
            context_text=ctx.text,
            metrics={
                "latency_ms": total_ms,
                "retrieval_ms": result.latency_ms,
                "n_candidates": result.n_candidates,
                "n_memories": len(result.items),
                "context_tokens": ctx.n_tokens,
                "raw_tokens": ctx.n_raw_tokens,
                "compression_ratio": round(ctx.n_tokens / max(1, ctx.n_raw_tokens), 3),
                "llm_mode": mode,
            },
        )
        if all(not n.source_ids for n in (i.node for i in result.items)):
            ans.confidence_note = "retrieved memories carry no chunk provenance"
        return ans


def _degenerate(text: str) -> bool:
    """True when an LLM output carries no answer substance (e.g. just "[1]")."""
    return len(re.sub(r"\[\d+\]|[^A-Za-z]", "", text or "").strip()) < 10


def parse_budget(config: Dict[str, Any]) -> int:
    ctx_cfg = config.get("context", {}) or {}
    val = ctx_cfg.get("max_tokens", "auto")
    if str(val).lower() != "auto":
        try:
            return int(val)
        except (TypeError, ValueError):
            pass
    return 2048  # safe default; auto refined by caller from model ctx
