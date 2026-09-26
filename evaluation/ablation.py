"""Ablation runner (spec §29).

Runs the benchmark through full MIRA and every component-subset
configuration, so results show WHICH components contribute.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, List

from core.retrieval import ablation_configs
from evaluation.benchmark import baseline_retrieve_fn, mira_retrieve_fn, run_system


def run_all_systems(ws, records: List[Dict[str, Any]], k: int = 8,
                    include_baselines: bool = True,
                    include_ablations: bool = True) -> Dict[str, Dict[str, Any]]:
    """Returns {system_name: {aggregate, rows}}. All systems share the same
    frame, stores, embedding backend, dataset, and hardware."""
    lock = getattr(ws, "lock", None)
    guard = lock if hasattr(lock, "__enter__") else nullcontext()
    with guard:
        systems: Dict[str, Any] = {}

        if include_baselines:
            from baselines.graph_rag import GraphRAG
            from baselines.hierarchical_rag import HierarchicalRAG
            from baselines.vector_rag import VectorRAG
            systems["vector_rag"] = baseline_retrieve_fn(VectorRAG(ws.frame, ws.vs), k=k)
            systems["graph_rag"] = baseline_retrieve_fn(GraphRAG(ws.frame, ws.vs, ws.gs), k=k)
            systems["hierarchical_rag"] = baseline_retrieve_fn(
                HierarchicalRAG(ws.frame, ws.vs), k=k)

        if include_ablations:
            for name, comps in ablation_configs().items():
                systems[name] = mira_retrieve_fn(ws, active_components=comps, k=k)

        out: Dict[str, Dict[str, Any]] = {}
        for name, fn in systems.items():
            out[name] = run_system(name, fn, ws.embeddings, records, k=k)
        return out
