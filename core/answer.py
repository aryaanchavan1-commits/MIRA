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
from threading import RLock
from typing import Any, Dict, List, Optional

import numpy as np

from core.affect import AffectSnapshot, coerce_snapshot, neutral_snapshot
from core.compression import compress, est_tokens
from core.memory import MemoryFrame, MemoryNode
from core.retrieval import MIRARetriever
from models.embeddings import EmbeddingBackend
from storage.graph_store import GraphStore
from storage.vector_store import VectorStore

logger = logging.getLogger("mira.answer")

_NO_EVIDENCE_TEXT = "The memories do not contain enough information."
_MAX_COMPLETION_TOKENS = 512
_ANSWER_SYSTEM = ("You are a careful research assistant. Answer only from "
                   "the provided excerpts and cite their numbers.")


def _synchronized(method):
    """Serialize operations on one pipeline's mutable query state."""
    def locked(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    locked.__name__ = getattr(method, "__name__", "locked")
    locked.__doc__ = method.__doc__
    locked.__wrapped__ = method
    return locked


def _answer_metrics(latency_ms: float = 0.0, retrieval_ms: float = 0.0,
                    n_candidates: int = 0, n_retrieved: int = 0,
                    n_memories: int = 0, context_tokens: int = 0,
                    raw_tokens: int = 0, llm_mode: str = "no_evidence",
                    context_budget: int = 0, completion_budget: int = 0,
                    citation_error: int = 0,
                    candidate_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """Return the stable metric shape consumed by the API and Streamlit."""
    out = {
        "latency_ms": round(float(latency_ms), 1),
        "retrieval_ms": round(float(retrieval_ms), 2),
        "n_candidates": max(0, int(n_candidates)),
        "n_retrieved": max(0, int(n_retrieved)),
        "n_memories": max(0, int(n_memories)),
        "n_selected_evidence": max(0, int(n_memories)),
        "n_excluded_candidates": max(0, int(n_retrieved) - int(n_memories)),
        "context_tokens": max(0, int(context_tokens)),
        "raw_tokens": max(0, int(raw_tokens)),
        "compression_ratio": round(
            float(context_tokens) / max(1, int(raw_tokens)), 3
        ) if context_tokens else 0.0,
        "citation_error": max(0, int(citation_error)),
        "context_budget": max(0, int(context_budget)),
        "completion_budget": max(0, int(completion_budget)),
        "llm_mode": llm_mode,
    }
    if candidate_ids is not None:
        out["candidate_ids"] = list(candidate_ids)
    return out


def _positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _llm_context_window(llm: Any, config: Dict[str, Any]) -> int:
    """Return the model's usable context window, not merely the config hint."""
    config = config or {}
    if llm is not None:
        for attr in ("n_ctx", "context_window", "ctx"):
            parsed = _positive_int(getattr(llm, attr, None))
            if parsed is not None:
                return parsed
        info = getattr(llm, "info", None)
        if callable(info):
            try:
                details = info() or {}
                for key in ("ctx", "n_ctx", "context_window"):
                    parsed = _positive_int(details.get(key))
                    if parsed is not None:
                        return parsed
            except Exception:
                pass
    llm_cfg = ((config.get("models", {}) or {}).get("llm", {}) or {})
    for key in ("max_context_tokens", "n_ctx", "context_window", "ctx"):
        parsed = _positive_int(llm_cfg.get(key))
        if parsed is not None:
            return parsed
    return 2048


def _completion_budget(llm: Any, messages: List[Dict[str, str]], cap: int,
                       config: Dict[str, Any]) -> int:
    if llm is None or not getattr(llm, "available", False):
        return 0
    try:
        cap = max(0, int(cap))
    except (TypeError, ValueError):
        cap = 0
    prompt_tokens = sum(est_tokens(m.get("content", ""))
                        for m in messages)
    return min(cap, max(0, _llm_context_window(llm, config) - prompt_tokens))


def _retry_prompt(context: str, question: str) -> str:
    return (f"Context:\n{context}\n\nQuestion: {question}\n\n"
            "Write one complete sentence that answers the question "
            "using only the context, then append the excerpt numbers "
            "you used in brackets like [1]. If the context lacks the "
            "answer, write exactly: " + _NO_EVIDENCE_TEXT)


def _citation_error(text: str, unit_count: int) -> int:
    if unit_count <= 0:
        return 0
    return sum(
        1 for marker in re.findall(r"\[(\d+)\]", text or "")
        if not 1 <= int(marker) <= unit_count
    )


@dataclass
class SourceRef:
    document_title: str = ""
    page: Optional[int] = None
    chunk_id: str = ""
    document_url: str = ""
    node_id: str = ""

    def label(self) -> str:
        loc = f"p.{self.page}" if self.page else "n.p."
        return f"{self.document_title} ({loc})" if self.document_title else loc

    def as_dict(self) -> Dict[str, Any]:
        return {
            "document": self.document_title,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "url": self.document_url,
            "node_id": self.node_id,
        }


@dataclass
class Answer:
    text: str = ""
    mode: str = "extractive"            # llm | extractive | no_evidence | deterministic
    agent_mode: str = "memory"          # memory | web | parametric | identity (routing origin)
    memories: List[Dict[str, Any]] = field(default_factory=list)   # ranked, with scores
    path_labels: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    confidence_note: str = ""
    context_text: str = ""              # compressed evidence fed to the LLM (for faithfulness judging)
    # Workspace/Agent replaces this with the post-route immutable snapshot.
    # The neutral default keeps standalone AnswerPipeline callers compatible.
    affect_snapshot: AffectSnapshot = field(default_factory=neutral_snapshot)
    # Node ids whose excerpts survived compression and were actually supplied
    # to the answer stage.  ``None`` means a legacy caller did not declare a
    # selection; an explicit empty list means that no evidence was retained.
    # Appended after the original fields to preserve positional construction
    # compatibility.
    selected_evidence_ids: Optional[List[str]] = None
    source_refs: List[Dict[str, Any]] = field(default_factory=list)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "affect_snapshot":
            value = coerce_snapshot(value)
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        selection_supplied = self.selected_evidence_ids is not None
        if not selection_supplied:
            self.selected_evidence_ids = []
        elif not isinstance(self.selected_evidence_ids, (list, tuple, set)):
            self.selected_evidence_ids = []
            selection_supplied = True
        else:
            seen = set()
            ids = []
            values = (sorted(self.selected_evidence_ids, key=lambda value: str(value))
                      if isinstance(self.selected_evidence_ids, set)
                      else self.selected_evidence_ids)
            for value in values:
                if not isinstance(value, str) or not value or value in seen:
                    continue
                seen.add(value)
                ids.append(value)
            self.selected_evidence_ids = ids
        if selection_supplied:
            # An explicit selection is authoritative, including an empty one.
            # Never expose records that were not supplied to the answer stage.
            allowed = set(self.selected_evidence_ids or [])
            self.memories = [
                memory for memory in (self.memories or [])
                if isinstance(memory, dict) and memory.get("id") in allowed
            ]
            if isinstance(self.metrics, dict):
                metrics = dict(self.metrics)
                retained = len(self.selected_evidence_ids or [])
                metrics["n_memories"] = retained
                metrics["n_selected_evidence"] = retained
                metrics.setdefault("n_candidates", retained)
                metrics.setdefault("n_retrieved", retained)
                try:
                    retrieved = int(metrics.get("n_retrieved", retained))
                except (TypeError, ValueError, OverflowError):
                    retrieved = retained
                metrics["n_excluded_candidates"] = max(0, retrieved - retained)
                self.metrics = metrics
            self.source_refs = [
                ref for ref in (self.source_refs or [])
                if isinstance(ref, dict) and ref.get("node_id") in allowed
            ]
            self.path_labels = [
                path for path in (self.path_labels or [])
                if all(token.strip() in allowed
                       for token in str(path).split(" -> "))
            ]
        object.__setattr__(self, "_selection_explicit", selection_supplied)
        self.source_refs = [dict(ref) for ref in (self.source_refs or [])
                            if isinstance(ref, dict)]
        self.affect_snapshot = coerce_snapshot(self.affect_snapshot)

    @property
    def evidence_ids(self) -> List[str]:
        """Alias for callers that use the shorter evidence-id spelling."""
        return list(self.selected_evidence_ids)

    @property
    def affect(self) -> AffectSnapshot:
        """Short alias used by API/UI adapters; the snapshot itself is read-only."""
        return self.affect_snapshot


class AnswerPipeline:
    def __init__(self, frame: MemoryFrame, vector_store: VectorStore,
                 graph_store: GraphStore, embeddings: EmbeddingBackend,
                 llm=None, config: Optional[Dict] = None,
                 doc_titles: Optional[Dict[str, str]] = None,
                 doc_sources: Optional[Dict[str, str]] = None,
                 store=None, operation_lock=None):
        self._lock = operation_lock or RLock()
        config = config if isinstance(config, dict) else {}
        self.retriever = MIRARetriever(frame, vector_store, graph_store, config)
        self.frame = frame
        self.embeddings = embeddings
        self.llm = llm
        self.config = config
        self.doc_titles = doc_titles or {}
        self.doc_sources = doc_sources or {}
        self.store = store  # optional persistence for Hebbian consolidation
        # injected by the Workspace so the agent layer can grow memory from
        # the web without a circular import
        self.workspace_ingest = None
        self._last_result_paths: List[List[str]] = []
        self._last_activation_trace: List[Dict[str, Any]] = []
        # Workspace installs this callback so cache refresh also keeps the
        # workspace-owned GraphStore and retriever in sync.
        self.workspace_cache_refresh = None

    def _sources_for(self, node: MemoryNode) -> List[SourceRef]:
        out: List[SourceRef] = []
        for chunk_id in node.source_ids[:3]:
            title = self.doc_titles.get(node.metadata.get("document_id", ""), "")
            out.append(SourceRef(document_title=title,
                                 page=node.metadata.get("page"),
                                 chunk_id=chunk_id,
                                 document_url=self.doc_sources.get(
                                     node.metadata.get("document_id", ""), ""),
                                 node_id=node.id))
        return out

    def _refresh_plasticity_caches(self) -> None:
        """Rebuild graph-dependent retriever caches after an edge update."""
        callback = getattr(self, "workspace_cache_refresh", None)
        if callable(callback):
            try:
                callback()
            except Exception as exc:
                logger.warning("plasticity cache refresh failed: %s", exc)
            return

        # Standalone AnswerPipeline callers do not have a Workspace callback.
        retriever = self.retriever
        retriever.frame = self.frame
        graph = getattr(retriever, "gs", None)
        if graph is not None and hasattr(graph, "build_from"):
            try:
                graph.build_from(
                    [self.frame.nodes[node_id].to_row()
                     for node_id in sorted(self.frame.nodes)],
                    [edge.to_row() for edge in sorted(
                        self.frame.edges,
                        key=lambda item: (item.source_id, item.target_id,
                                          item.relation_type))],
                )
            except Exception as exc:
                logger.warning("graph cache rebuild failed: %s", exc)
        activation = getattr(retriever, "activation", None)
        if activation is not None:
            activation.frame = self.frame
            if hasattr(activation, "invalidate"):
                activation.invalidate()
        if graph is not None and hasattr(graph, "centrality"):
            retriever.centrality = graph.centrality()
        else:
            retriever.centrality = {}

    @_synchronized
    def reinforce_last_result(self) -> None:
        """Consolidate the most recent *accepted grounded* retrieval.

        The agent calls this only after a strong memory answer or a fresh web
        answer.  A non-empty LIF trace takes the trace-driven path; continuous
        mode and callers without a trace retain the legacy path-based API.
        """
        memory_cfg = self.config.get("memory", {}) or {}
        if not isinstance(memory_cfg, dict):
            memory_cfg = {}
        if self.store is None:
            return
        hebbian = memory_cfg.get("hebbian", True)
        if isinstance(hebbian, str):
            hebbian = hebbian.strip().lower() not in {"0", "false", "no", "off", "none"}
        if not hebbian or (not self._last_result_paths and not self._last_activation_trace):
            return

        raw_stdp = memory_cfg.get(
            "stdp", self.config.get("stdp", memory_cfg.get("trace_plasticity", {})))
        if isinstance(raw_stdp, (bool, int, float)):
            stdp_enabled, stdp_cfg = bool(raw_stdp), {}
        elif isinstance(raw_stdp, dict):
            stdp_enabled = raw_stdp.get("enabled", True)
            if isinstance(stdp_enabled, str):
                stdp_enabled = stdp_enabled.strip().lower() not in {
                    "0", "false", "no", "off", "none"}
            stdp_cfg = raw_stdp
        else:
            stdp_enabled, stdp_cfg = True, {}

        try:
            has_lif_spikes = any(
                bool(event.get("spike")) for event in self._last_activation_trace
            )
            if has_lif_spikes:
                if not stdp_enabled:
                    return
                from core.hebbian import stdp_update
                changes = stdp_update(
                    self.frame,
                    self._last_activation_trace,
                    lr=stdp_cfg.get(
                        "potentiation", stdp_cfg.get("potentiate",
                                      stdp_cfg.get("ltp", stdp_cfg.get("lr", 0.05)))),
                    depression=stdp_cfg.get(
                        "depression", stdp_cfg.get("depression_rate",
                                      stdp_cfg.get("depress",
                                      stdp_cfg.get("ltd", 0.02)))),
                    window=stdp_cfg.get("window", stdp_cfg.get("window_ticks", 1)),
                    decay=stdp_cfg.get("decay",
                                      stdp_cfg.get("homeostatic_decay", 0.0)),
                    max_events=stdp_cfg.get("max_events",
                                            stdp_cfg.get("max_trace_events", 4096)),
                    spike_only=stdp_cfg.get("spike_only",
                                            stdp_cfg.get("use_spikes", True)),
                    store=self.store,
                )
            else:
                from core.hebbian import reinforce
                changes = reinforce(self.frame, self._last_result_paths, store=self.store)
            if changes:
                self._refresh_plasticity_caches()
        except Exception as exc:
            logger.warning("hebbian consolidation failed: %s", exc)

    @_synchronized
    def ask(self, question: str, active_components=None,
            max_context_tokens: Optional[int] = None,
            system_name: str = "mira",
            query_vec: Optional[np.ndarray] = None,
            final_k: Optional[int] = None) -> Answer:
        t0 = time.perf_counter()
        self._last_result_paths = []
        self._last_activation_trace = []
        qvec = query_vec if query_vec is not None else self.embeddings.encode([question])[0]
        try:
            effective_k = (max(1, int(final_k)) if final_k is not None
                           else max(1, int(self.retriever.final_k)))
        except (TypeError, ValueError):
            effective_k = max(1, int(self.retriever.final_k))
        retrieve_kwargs = {"active_components": active_components,
                           "final_k": effective_k}
        result = self.retriever.retrieve(question, qvec, **retrieve_kwargs)
        retrieved_ids = [item.node.id for item in result.items]
        retrieved_count = len(retrieved_ids)
        if len(result.items) > effective_k:
            result.items = result.items[:effective_k]
        if not getattr(result, "query", ""):
            result.query = question
        raw_trace = getattr(result, "activation_trace", None)
        if raw_trace is None:
            raw_trace = getattr(
                getattr(self.retriever, "activation", None), "last_trace", [])
        raw_trace = list(raw_trace or [])
        if not result.items:
            self._last_activation_trace = []
            return Answer(
                text=_NO_EVIDENCE_TEXT,
                mode="no_evidence",
                metrics=_answer_metrics(
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    retrieval_ms=result.latency_ms,
                    n_candidates=result.n_candidates,
                    n_retrieved=retrieved_count,
                    candidate_ids=[],
                ),
                selected_evidence_ids=[],
            )

        ctx_cfg = self.config.get("context", {}) or {}
        requested_budget = (max_context_tokens if max_context_tokens is not None
                            else parse_budget(self.config))
        try:
            requested_budget = max(0, int(requested_budget))
        except (TypeError, ValueError):
            requested_budget = parse_budget(self.config)

        llm_active = bool(getattr(self.llm, "available", False))
        completion_budget = 0
        completion_reserve = 0
        context_window: Optional[int] = None
        if llm_active:
            from models.llm import answer_prompt
            context_window = _llm_context_window(self.llm, self.config)
            # Reserve room for the fixed instructions and a useful completion
            # before packing excerpts.  The exact remaining budget is refined
            # below once the compressed text is known.
            prompt_overhead = max(
                est_tokens(answer_prompt("", question)),
                est_tokens(_retry_prompt("", question)),
            ) + est_tokens(_ANSWER_SYSTEM)
            completion_reserve = min(_MAX_COMPLETION_TOKENS,
                                     max(1, context_window // 4))
            requested_budget = min(
                requested_budget,
                max(0, context_window - prompt_overhead - completion_reserve),
            )
        else:
            # There is no generation call without an available model; retain
            # the configured context budget for the deterministic path.
            context_window = None

        try:
            target_ratio = float(ctx_cfg.get("compression_target_ratio", 0.6))
        except (TypeError, ValueError):
            target_ratio = 0.6
        ctx = compress(result, max_tokens=requested_budget,
                       target_ratio=target_ratio)

        # A retrieval hit that is only a structural/label node has no evidence
        # to answer from.  Do not call the model or mislabel the fallback as
        # extractive in that case.
        if not ctx.text.strip():
            self._last_activation_trace = []
            return Answer(
                text=_NO_EVIDENCE_TEXT,
                mode="no_evidence",
                metrics=_answer_metrics(
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    retrieval_ms=result.latency_ms,
                    n_candidates=result.n_candidates,
                    n_retrieved=retrieved_count,
                    raw_tokens=ctx.n_raw_tokens,
                    context_budget=requested_budget,
                candidate_ids=retrieved_ids,

                ),
                selected_evidence_ids=[],
            )

        # Answer generation; Hebbian consolidation is deferred until
        # AgentPipeline accepts this answer as grounded (memory or web).
        if llm_active:
            from models.llm import answer_prompt
            prompt = answer_prompt(ctx.text, question)
            retry = _retry_prompt(ctx.text, question)
            prompt_tokens = max(est_tokens(prompt), est_tokens(retry)) + est_tokens(_ANSWER_SYSTEM)
            completion_budget = min(
                completion_reserve,
                max(0, (context_window or 0) - prompt_tokens),
            )

        if llm_active and completion_budget > 0 and ctx.text.strip():
            # greedy first: deterministic, grounded — best mode for small models
            text = self.llm.chat(
                [{"role": "system", "content": _ANSWER_SYSTEM},
                 {"role": "user", "content": prompt}],
                max_tokens=completion_budget, temperature=0.0,
            ) or ""
            if _degenerate(text):
                # strategy 2: explicit sentence-first instruction. Small models
                # sometimes stop right after emitting a citation marker.
                text = self.llm.chat(
                    [{"role": "user", "content": retry}],
                    max_tokens=completion_budget, temperature=0.2,
                ) or ""
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

        explicit_no_evidence = _is_no_evidence_text(text)
        if explicit_no_evidence:
            text = _NO_EVIDENCE_TEXT
            mode = "no_evidence"

        # Provenance and learning signals are restricted to excerpts that
        # survived compression.  Candidates dropped by the context budget are
        # still visible in n_candidates/n_retrieved, but are never presented as
        # used memories, paths, or grounded evidence.
        selected_ids: List[str] = []
        selected_id_set: set = set()
        if not explicit_no_evidence:
            for unit in ctx.units:
                node_id = unit.get("node_id") if isinstance(unit, dict) else None
                if isinstance(node_id, str) and node_id and node_id not in selected_id_set:
                    selected_id_set.add(node_id)
                    selected_ids.append(node_id)
        selected_ids = selected_ids[:effective_k]
        selected_id_set = set(selected_ids)
        self._last_activation_trace = [
            event for event in raw_trace
            if isinstance(event, dict)
            and str(event.get("node")) in selected_id_set
        ]
        self._last_result_paths = [
            list(item.path) for item in result.items
            if item.node.id in selected_id_set and len(item.path) > 1
        ]

        selected_nodes: List[MemoryNode] = []
        sources: List[str] = []
        source_refs: List[Dict[str, Any]] = []
        seen: set = set()
        memories: List[Dict[str, Any]] = []
        selected_paths: List[List[str]] = []
        for item in result.items:
            n = item.node
            if n.id not in selected_id_set:
                continue
            memories.append({
                "id": n.id,
                "concept": n.concept, "type": n.memory_type.value,
                "ring": n.ring, "sector": n.sector, "score": round(item.score, 4),
                "components": {k: round(item.components[k], 3)
                               for k in sorted(item.components)},
            })
            selected_nodes.append(n)
            if len(item.path) > 1:
                selected_paths.append(list(item.path))
            for ref in self._sources_for(n):
                label = ref.label()
                ref_key = (label, ref.chunk_id, ref.document_url)
                if ref_key not in seen:
                    seen.add(ref_key)
                    sources.append(label)
                    source_refs.append(ref.as_dict())

        total_ms = (time.perf_counter() - t0) * 1000
        citation_error = _citation_error(text, len(ctx.units))
        ans = Answer(
            text=text, mode=mode, memories=memories[:effective_k],
            path_labels=[" -> ".join(path) for path in selected_paths[:5]],
            sources=sources[:6],
            context_text=ctx.text,
            selected_evidence_ids=selected_ids,
            source_refs=source_refs[:12],
            metrics=_answer_metrics(
                latency_ms=total_ms,
                retrieval_ms=result.latency_ms,
                n_candidates=result.n_candidates,
                n_retrieved=retrieved_count,
                n_memories=len(selected_ids),
                context_tokens=ctx.n_tokens,
                raw_tokens=ctx.n_raw_tokens,
                llm_mode=mode,
                context_budget=requested_budget,
                completion_budget=completion_budget,
                citation_error=citation_error,
                 candidate_ids=retrieved_ids,

            ),
        )
        if explicit_no_evidence:
            ans.confidence_note = "the model reported that the supplied evidence was insufficient"
        elif citation_error:
            ans.confidence_note = (
                f"{citation_error} generated citation marker(s) point outside "
                "the supplied evidence")
        elif selected_nodes and all(not n.source_ids for n in selected_nodes):
            ans.confidence_note = "retrieved memories carry no chunk provenance"
        return ans


def _is_no_evidence_text(text: str) -> bool:
    without_citations = re.sub(r"\[\d+\]", "", text or "")
    normalized = re.sub(r"[^a-z0-9]+", " ", without_citations.casefold()).strip()
    expected = re.sub(r"[^a-z0-9]+", " ", _NO_EVIDENCE_TEXT.casefold()).strip()
    return normalized == expected


def _degenerate(text: str) -> bool:
    """True when an LLM output carries no answer substance (e.g. just "[1]")."""
    without_citations = re.sub(r"\[\d+\]", "", text or "")
    return sum(char.isalpha() for char in without_citations) < 10


def parse_budget(config: Dict[str, Any]) -> int:
    ctx_cfg = config.get("context", {}) or {}
    val = ctx_cfg.get("max_tokens", "auto")
    if str(val).lower() != "auto":
        try:
            return int(val)
        except (TypeError, ValueError):
            pass
    return 2048  # safe default; auto refined by caller from model ctx
