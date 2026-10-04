"""Continual-learning / catastrophic-forgetting metrics (spec §16).

Definitions used throughout the lab and the UI — deliberately explicit, so a
reader can disagree with the definition rather than guess it:

    performance_i(step)  = retrieval quality on task i at that step
    best_previous_i(step) = max performance on task i over all EARLIER steps
    Forgetting_i(step)    = best_previous_i(step) - performance_i(step)
    Average Forgetting    = mean over tasks of Forgetting_i at the final step
    Retention_i(step)     = performance_i(step) / best_previous_i(step)

Note on what "forgetting" means here: the LLM is frozen and no weights are
updated (spec §20), so this measures **memory interference** — previously
acquired knowledge becoming harder to retrieve as new knowledge is added —
not parametric catastrophic forgetting. Positive Forgetting = degradation.

Every metric is computed from plain dicts so the experiment scripts, the API
and the tests all agree on the arithmetic.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

METRICS = ("mrr", "recall", "hit_at_k", "precision", "answer_coverage",
           "answer_found", "latency_ms")


def _series(rows: Iterable[Dict[str, Any]], task: int,
            metric: str) -> List[Dict[str, Any]]:
    out = [r for r in rows if r["task"] == task and metric in r]
    return sorted(out, key=lambda r: r["step"])


def forgetting_metrics(rows: List[Dict[str, Any]],
                       metrics: Iterable[str] = ("mrr", "recall"),
                       ) -> Dict[str, Any]:
    """Per-task retention/forgetting curves plus lab-level aggregates.

    ``rows`` are ``{"step", "task", "mrr", "recall", ...}`` observations.
    """
    metrics = [m for m in metrics if m in METRICS]
    steps = sorted({r["step"] for r in rows})
    tasks = sorted({r["task"] for r in rows})
    out: Dict[str, Any] = {"steps": steps, "tasks": tasks, "metrics": {},
                           "per_task": {}, "average_forgetting": {},
                           "mean_retention": {}, "final_performance": {}}
    for metric in metrics:
        curves: Dict[str, List[Dict[str, Any]]] = {}
        for task in tasks:
            pts = _series(rows, task, metric)
            best_prev: Optional[float] = None
            curve: List[Dict[str, Any]] = []
            for p in pts:
                current = float(p[metric])
                forgetting = 0.0 if best_prev is None else max(0.0, best_prev - current)
                retention = 1.0 if not best_prev else current / best_prev if best_prev else 1.0
                curve.append({
                    "step": p["step"],
                    "value": round(current, 6),
                    "best_previous": None if best_prev is None else round(best_prev, 6),
                    "forgetting": round(forgetting, 6),
                    "retention": round(retention, 6),
                    "n": p.get("n"),
                })
                if best_prev is None or current > best_prev:
                    best_prev = current
            curves[str(task)] = curve
            if curve:
                out["final_performance"].setdefault(metric, {})[str(task)] = curve[-1]["value"]
        out["metrics"][metric] = curves
        out["per_task"].setdefault(metric, curves)
        # lab-level aggregates at the final step
        final_forgetting = [c[-1]["forgetting"] for c in curves.values() if c]
        final_retention = [c[-1]["retention"] for c in curves.values() if c]
        out["average_forgetting"][metric] = round(
            sum(final_forgetting) / len(final_forgetting), 6) if final_forgetting else 0.0
        out["mean_retention"][metric] = round(
            sum(final_retention) / len(final_retention), 6) if final_retention else 1.0
    return out


def summary_row(rows: List[Dict[str, Any]], step: int) -> Dict[str, Any]:
    """Mean performance over all tasks tested at ``step`` (the lab's headline)."""
    at_step = [r for r in rows if r["step"] == step]
    if not at_step:
        return {}
    out: Dict[str, Any] = {"step": step, "tasks": len(at_step)}
    for metric in METRICS:
        vals = [float(r[metric]) for r in at_step if metric in r]
        if vals:
            out[metric] = round(sum(vals) / len(vals), 6)
    return out
