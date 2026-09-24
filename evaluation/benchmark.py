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
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

from core.memory import MemoryFrame
from evaluation.metrics import aggregate, context_tokens, mrr, recall_at_k, token_f1

logger = logging.getLogger("mira.benchmark")

Record = Dict[str, Any]  # {question, answer, supporting_ids?: [node ids]}


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
        ids = out["node_ids"]
        relevant: Set[str] = set(rec.get("supporting_ids") or [])
        row: Dict[str, Any] = {
            "question": rec["question"],
            "reference_answer": rec["answer"],
            "answer_text": out.get("answer") or "",
            "evidence_text": out.get("context_text") or "",
            "latency_ms": out.get("latency_ms", 0.0),
            "retrieval_latency_ms": out.get("retrieval_latency_ms", out.get("latency_ms", 0.0)),
            "n_retrieved": len(ids),
            "context_tokens": context_tokens(out.get("context_text", "")),
        }
        if relevant:
            row["retrieval_recall"] = recall_at_k(ids, relevant, k)
            row["mrr"] = mrr(ids, relevant)
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
    """Builds a FRESH retriever per call: no cross-run config caching (a cached
    retriever silently reused stale weights after config flips) and no stale
    frame reference after ingestion."""
    def fn(question: str, qvec: np.ndarray) -> Dict[str, Any]:
        from core.answer import AnswerPipeline
        pipe = AnswerPipeline(ws.frame, ws.vs, ws.gs, ws.embeddings,
                              llm=ws.llm, config=ws.config,
                              doc_titles=ws._doc_titles())
        res = pipe.retriever.retrieve(question, qvec,
                                      active_components=active_components,
                                      final_k=k)
        out: Dict[str, Any] = {
            "node_ids": [it.node.id for it in res.items],
            "latency_ms": res.latency_ms,
        }
        return out
    return fn


def answer_retrieve_fn(ws, active_components=None, k: int = 8) -> Callable:
    """Retrieval + the FULL shared answer stage (compression + LLM), so every
    MIRA ablation is scored with an identical downstream pipeline. Baselines
    keep their own retrieval-only fns and are compared at retrieval level."""
    def fn(question: str, qvec: np.ndarray) -> Dict[str, Any]:
        from core.answer import AnswerPipeline
        pipe = AnswerPipeline(ws.frame, ws.vs, ws.gs, ws.embeddings,
                              llm=ws.llm, config=ws.config,
                              doc_titles=ws._doc_titles())
        ans = pipe.ask(question, active_components=active_components,
                       system_name="bench")
        return {
            "node_ids": [m.get("id") for m in ans.memories if m.get("id")],
            "answer": ans.text if ans.mode != "no_evidence" else None,
            "context_text": ans.context_text,
            "latency_ms": ans.metrics.get("latency_ms", 0.0),
        }
    return fn


def baseline_retrieve_fn(system, k: int = 8) -> Callable:
    """system: VectorRAG / GraphRAG / HierarchicalRAG instance."""
    def fn(question: str, qvec: np.ndarray) -> Dict[str, Any]:
        if hasattr(system, "retrieve") and "query_vec" in system.retrieve.__code__.co_varnames:
            res = system.retrieve(qvec, k=k)
        else:
            res = system.retrieve(question, k=k)
        return {"node_ids": [n.id for n in res.items], "latency_ms": res.latency_ms}
    return fn
