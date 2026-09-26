"""Benchmark framework (spec §30-31).

Runs a dataset of {question, answer, supporting_ids?} records through any
retrieval system (baselines + MIRA + ablations), computes metrics from real
output, and returns an aggregate report. Dataset loading is conservative:
local files only, never auto-downloaded.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Set

import numpy as np

from evaluation.metrics import aggregate, context_tokens, mrr, recall_at_k, token_f1

logger = logging.getLogger("mira.benchmark")

Record = Dict[str, Any]  # {question, answer, supporting_ids?: [node ids]}


def _workspace_guard(ws):
    """Use the workspace consistency lock when the adapter has one."""
    lock = getattr(ws, "lock", None)
    return lock if hasattr(lock, "__enter__") else nullcontext()


def _active_names(active) -> List[str]:
    from core.retrieval import ALL_COMPONENTS
    if isinstance(active, str):
        names = [name.strip() for name in active.split(",") if name.strip()]
        active = None if not names or names == ["all"] else names
    return list(ALL_COMPONENTS if active is None else sorted(set(active)))


def load_dataset(path: str) -> List[Record]:
    """Local JSON/JSONL dataset: [{question, answer, supporting_ids?}]."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read().strip()
    if text.startswith("["):
        data = json.loads(text)
    else:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    out = []
    for d in data:
        if d.get("question") and d.get("answer"):
            out.append({"question": d["question"], "answer": d["answer"],
                        "supporting_ids": list(d.get("supporting_ids") or [])})
    return out


def sample_dataset(records: List[Record], n: int, seed: int = 42) -> List[Record]:
    if n <= 0 or n >= len(records):
        return records
    import random
    rng = random.Random(seed)
    return rng.sample(records, n)


def run_system(name: str, retrieve_fn: Callable[[str, np.ndarray], Dict[str, Any]],
               embeddings, records: List[Record], k: int = 8) -> Dict[str, Any]:
    """retrieve_fn(question, qvec) -> {node_ids, answer?, context_text?, latency_ms}."""
    rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for rec in records:
        qvec = embeddings.encode([rec["question"]])[0]
        out = retrieve_fn(rec["question"], qvec)
        # Answer-stage adapters explicitly report the ids that survived
        # compression.  Retrieval-only adapters retain their full candidate
        # list; never silently fall back to candidates for an answer run.
        if "selected_evidence_ids" in out:
            ids = list(out.get("selected_evidence_ids") or [])
        else:
            ids = list(out.get("node_ids") or [])
        relevant: Set[str] = set(rec.get("supporting_ids") or [])
        row: Dict[str, Any] = {
            "question": rec["question"],
            "reference_answer": rec["answer"],
            "answer_text": out.get("answer") or "",
            "evidence_text": out.get("context_text") or "",
            "latency_ms": out.get("latency_ms", 0.0),
            "retrieval_latency_ms": out.get("retrieval_latency_ms", out.get("latency_ms", 0.0)),
            "n_retrieved": int(out.get("n_retrieved", len(ids))),
            "n_candidates": int(out.get("n_candidates", out.get("n_retrieved", len(ids)))),
            "context_tokens": out.get("context_tokens", context_tokens(out.get("context_text", ""))),
            "active_components": out.get("active_components"),
            "candidate_policy": out.get("candidate_policy"),
            "answer_metrics": dict(out.get("metrics") or {}),
        }
        if "selected_evidence_ids" in out:
            row["selected_evidence_ids"] = list(ids)
        candidate_ids = out.get("candidate_ids", out.get("node_ids", []))
        if candidate_ids:
            row["candidate_ids"] = list(candidate_ids)
        if relevant:
            row["retrieval_recall"] = recall_at_k(ids, relevant, k)
            row["mrr"] = mrr(ids, relevant, k)
        if out.get("answer") is not None:
            row["answer_token_f1"] = token_f1(out["answer"], rec["answer"])
        rows.append(row)
    total_s = time.perf_counter() - t0
    agg = aggregate(rows)
    agg["wall_time_s"] = round(total_s, 2)
    agg["questions_per_s"] = round(len(records) / total_s, 2) if total_s > 0 else 0.0
    return {"system": name, "aggregate": agg, "rows": rows}


def mira_retrieve_fn(ws, active_components=None, k: int = 8,
                     with_answer: bool = False) -> Callable:
    """One AnswerPipeline per sweep-lambda, built lazily on first query and
    rebuilt only if the workspace swaps its frame or vector store (stale-frame
    guard). Config is fixed for the lifetime of this fn by construction — every
    caller creates it inside a single run/request — so per-query rebuilding
    bought nothing except O(nodes) allocation churn per query, which OOM'd at
    82k nodes."""
    state: Dict[str, Any] = {}

    def fn(question: str, qvec: np.ndarray) -> Dict[str, Any]:
        from core.answer import AnswerPipeline
        with _workspace_guard(ws):
            entry = state.get("pipe")
            if entry is None or entry[0] is not ws.frame or entry[1] is not ws.vs:
                pipe = AnswerPipeline(ws.frame, ws.vs, ws.gs, ws.embeddings,
                                      llm=ws.llm, config=ws.config,
                                      doc_titles=ws._doc_titles(),
                                      operation_lock=getattr(ws, "_lock", None))
                state["pipe"] = (ws.frame, ws.vs, pipe)
            else:
                pipe = entry[2]
            res = pipe.retriever.retrieve(question, qvec,
                                          active_components=active_components,
                                          final_k=k)
            out: Dict[str, Any] = {
                "node_ids": [it.node.id for it in res.items],
                "n_retrieved": len(res.items),
                "n_candidates": res.n_candidates,
                "latency_ms": res.latency_ms,
                "active_components": _active_names(active_components),
                "candidate_policy": "shared",
            }
            return out
    return fn


def answer_retrieve_fn(ws, active_components=None, k: int = 8) -> Callable:
    """Retrieval + the FULL shared answer stage (compression + LLM), so every
    MIRA ablation is scored with an identical downstream pipeline. Baselines
    keep their own retrieval-only fns and are compared at retrieval level.
    Same lazy once-per-fn pipeline reuse as mira_retrieve_fn (OOM at 82k nodes
    if rebuilt per query)."""
    state: Dict[str, Any] = {}

    def fn(question: str, qvec: np.ndarray) -> Dict[str, Any]:
        from core.answer import AnswerPipeline
        with _workspace_guard(ws):
            entry = state.get("pipe")
            if entry is None or entry[0] is not ws.frame or entry[1] is not ws.vs:
                pipe = AnswerPipeline(ws.frame, ws.vs, ws.gs, ws.embeddings,
                                      llm=ws.llm, config=ws.config,
                                      doc_titles=ws._doc_titles(),
                                      operation_lock=getattr(ws, "_lock", None))
                state["pipe"] = (ws.frame, ws.vs, pipe)
            else:
                pipe = entry[2]
            ans = pipe.ask(question, active_components=active_components,
                           system_name="bench", query_vec=qvec, final_k=k)
            if hasattr(ans, "selected_evidence_ids"):
                selected_ids = list(ans.selected_evidence_ids or [])
            else:
                # Compatibility for Answer-like test doubles that predate the
                # explicit field; real pipeline answers always populate it.
                selected_ids = [m.get("id") for m in ans.memories if m.get("id")]
            return {
                "node_ids": [m.get("id") for m in ans.memories if m.get("id")],
                "candidate_ids": list(ans.metrics.get("candidate_ids", [])),
                "selected_evidence_ids": selected_ids,
                "n_retrieved": ans.metrics.get("n_retrieved", len(selected_ids)),
                "n_candidates": ans.metrics.get("n_candidates", len(selected_ids)),
                "n_memories": ans.metrics.get("n_memories", len(selected_ids)),
                "answer": ans.text if ans.mode != "no_evidence" else None,
                "context_text": ans.context_text,
                "context_tokens": ans.metrics.get("context_tokens", 0),
                "latency_ms": ans.metrics.get("latency_ms", 0.0),
                "metrics": dict(ans.metrics),
                "active_components": _active_names(active_components),
                "candidate_policy": "shared",
            }
    return fn


def baseline_retrieve_fn(system, k: int = 8) -> Callable:
    """Build an adapter for a named baseline retrieval implementation."""
    system_name = getattr(system, "system", "")
    vector_based = system_name in {"vector_rag", "graph_rag"}

    def fn(question: str, qvec: np.ndarray) -> Dict[str, Any]:
        if vector_based:
            res = system.retrieve(qvec, k=k)
        else:
            res = system.retrieve(question, k=k)
        return {"node_ids": [n.id for n in res.items], "latency_ms": res.latency_ms}
    return fn
