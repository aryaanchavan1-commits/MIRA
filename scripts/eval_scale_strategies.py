"""Scale test: does the topology advantage change with corpus size?

The paper's core hypothesis predicts the gap between topology-aware
placement strategies (hybrid_mira, hierarchy, graph_centrality) and
structure-blind ones (embedding_clusters, temporal) WIDENS as the corpus
grows. This runner sweeps all placement strategies on the big MuSiQue
bench workspace, plus a size-ladder sub-sweep (5k → 20k → 50k nodes)
so the trend can be plotted, not just the endpoint.

Self-supervised corpus QA (corpus_dataset) at each scale — the SAME probe
at each size, so the TREND is internally comparable. Real-data MuSiQue
numbers come from eval_benchmark_real.py; this script answers the scaling
question specifically.

Usage: .venv/Scripts/python.exe scripts/eval_scale_strategies.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.workspace as cw  # noqa: E402
_BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_bench")
cw.DATA_DIR = _BENCH
import config.auto_config as ac  # noqa: E402
ac.DATA_DIR = _BENCH

from config.auto_config import build_context  # noqa: E402
from core.placement import STRATEGIES as PLACEMENT_STRATEGIES  # noqa: E402
from core.workspace import Workspace  # noqa: E402
from evaluation.benchmark import mira_retrieve_fn, run_system  # noqa: E402
from evaluation.datasets import corpus_dataset  # noqa: E402

OUT = os.path.join(_BENCH, "scale_sweep_results.json")


def sweep(ws, records, k=8):
    table = []
    current = (ws.config.get("topology", {}) or {}).get("placement_strategy",
                                                        "hybrid_mira")
    try:
        for name in PLACEMENT_STRATEGIES:
            ws.replace_all(strategy=name)
            res = run_system(name, mira_retrieve_fn(ws, None, k=k),
                             ws.embeddings, records, k=k)
            agg = res["aggregate"]
            table.append({"strategy": name, "mrr": agg.get("mrr", 0.0),
                          "recall": agg.get("retrieval_recall", 0.0),
                          "latency_ms": agg.get("latency_ms", 0.0)})
            print(f"  {name:20s} mrr={agg.get('mrr', 0):.4f} "
                  f"recall={agg.get('retrieval_recall', 0):.4f}", flush=True)
    finally:
        ws.replace_all(strategy=current)
    return table


def main() -> int:
    ctx = build_context()
    ws = Workspace(embeddings=ctx.embeddings, llm=None, config=ctx.cfg)
    total = len(ws.frame.nodes)
    print(f"scale workspace: {total} nodes, vs={ws.vs is not None}")

    # size ladder: same questions at increasing corpus sizes
    ladder = []
    for frac in (0.05, 0.15, 0.4, 1.0):
        n_nodes = int(total * frac)
        # subset documents until we reach the target scale
        docs = sorted({sid for n in ws.frame.nodes.values()
                       for sid in (n.source_ids or []) if sid.startswith("doc_")})
        keep = docs[:max(1, int(len(docs) * frac))]
        sub_nodes = {nid: n for nid, n in ws.frame.nodes.items()
                     if any(sid in keep for sid in (n.source_ids or []))}
        # probe questions: from the FULL corpus set, fixed count for fairness
        records = corpus_dataset(ws, limit=40)
        print(f"\n=== scale {frac:.0%}: {len(keep)} docs / {len(sub_nodes)} nodes "
              f"(probe: {len(records)} fixed questions) ===", flush=True)
        t0 = time.time()
        # run sweep against a reduced frame view
        import copy
        full_frame = ws.frame
        ws.frame = type(full_frame)()
        for nid in sorted(sub_nodes):
            ws.frame.add_node(sub_nodes[nid])
        for e in full_frame.edges:
            if e.source_id in sub_nodes and e.target_id in sub_nodes:
                ws.frame.add_edge(e)
        ws.gs.build_from(ws.frame)
        try:
            table = sweep(ws, records, k=8)
        finally:
            ws.frame = full_frame
            ws.gs.build_from(full_frame)
        ladder.append({"frac": frac, "n_docs": len(keep),
                       "n_nodes": len(sub_nodes), "n_questions": len(records),
                       "table": table, "seconds": round(time.time() - t0, 1)})

    # trend summary: topology-aware vs structure-blind gap per scale
    print("\n=== trend: topology-aware vs structure-blind ===")
    aware = {"hybrid_mira", "hierarchy", "graph_centrality"}
    for step in ladder:
        best_aware = max(r["mrr"] for r in step["table"] if r["strategy"] in aware)
        best_blind = max(r["mrr"] for r in step["table"] if r["strategy"] not in aware)
        gap = best_aware - best_blind
        step["aware_best_mrr"] = round(best_aware, 4)
        step["blind_best_mrr"] = round(best_blind, 4)
        step["gap"] = round(gap, 4)
        print(f"  {step['n_nodes']:6d} nodes: aware={best_aware:.4f} "
              f"blind={best_blind:.4f} gap={gap:+.4f}")

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"n_nodes_total": total, "ladder": ladder}, fh, indent=1)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
