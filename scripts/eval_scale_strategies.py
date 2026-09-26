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
import random
import sys
import time

if hasattr(sys.stdout, "reconfigure"):  # cp1252 consoles choke on fancy glyphs
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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

    # size ladder: same probe design at increasing corpus sizes.
    # Nodes carry CHUNK ids in source_ids; resolve chunk->doc once.
    chunk_to_doc = {}
    for d in ws.store.list_documents():
        for c in ws.store.document_chunks(d["id"]):
            chunk_to_doc[c["id"]] = d["id"]
    node_doc = {nid: {chunk_to_doc[c] for c in (n.source_ids or [])
                      if c in chunk_to_doc}
                for nid, n in ws.frame.nodes.items()}
    docs = sorted({d for ds in node_doc.values() for d in ds})

    # probe pool: hub nodes (>=5 neighbors); gold = their graph neighbors.
    # corpus_dataset self-probes can't discriminate placement — the query node
    # is a perfect semantic match under every strategy (MRR flatlines at ~1).
    adj: dict = {}
    for e in ws.frame.edges:
        adj.setdefault(e.source_id, set()).add(e.target_id)
        adj.setdefault(e.target_id, set()).add(e.source_id)
    pool = [(n.id, sorted(adj[n.id] - {n.id}))
            for n in ws.frame.nodes.values()
            if len(adj.get(n.id, ())) >= 5 and (n.concept or "").strip()]
    print(f"probe pool: {len(pool)} hub nodes (gold = graph neighborhood)")

    ladder = []
    for frac in (0.05, 0.15, 0.4, 1.0):
        keep = set(docs[:max(1, int(len(docs) * frac))])
        # fact nodes of kept docs, plus concept/entity hubs adjacent to >=2
        # kept facts (concepts have no source_ids; without the closure every
        # subset loses its hubs and the probe pool empties).
        fact_ids = {nid for nid, ds in node_doc.items() if ds & keep}
        sub_ids = set(fact_ids)
        hub_count: dict = {}
        for e in ws.frame.edges:
            a, b = e.source_id, e.target_id
            a_in, b_in = a in sub_ids, b in sub_ids
            if a_in != b_in:
                other = b if a_in else a
                hub_count[other] = hub_count.get(other, 0) + 1
        sub_ids |= {nid for nid, c in hub_count.items() if c >= 2}
        sub_nodes = {nid: ws.frame.nodes[nid] for nid in sub_ids}
        sub_adj: dict = {}
        for e in ws.frame.edges:
            if e.source_id in sub_ids and e.target_id in sub_ids:
                sub_adj.setdefault(e.source_id, set()).add(e.target_id)
                sub_adj.setdefault(e.target_id, set()).add(e.source_id)
        sub_pool = [(nid, sorted(sub_adj[nid] - {nid}))
                    for nid in sub_ids
                    if len(sub_adj.get(nid, ())) >= 5
                    and (ws.frame.nodes[nid].concept or "").strip()]
        rng = random.Random(1234)
        rng.shuffle(sub_pool)
        probes = sub_pool[:40]
        if len(probes) < 10:
            print(f"\n=== scale {frac:.0%}: skipped (only {len(probes)} "
                  f"hub probes in induced subgraph) ===", flush=True)
            continue
        records = [{"question": ws.frame.nodes[nid].concept,
                    "answer": (ws.frame.nodes[nid].summary
                               or ws.frame.nodes[nid].raw_text or "")[:300],
                    "supporting_ids": gold}
                   for nid, gold in probes]
        print(f"\n=== scale {frac:.0%}: {len(keep)} docs / {len(sub_nodes)} nodes "
              f"(probe: {len(records)} hub-neighborhood questions) ===", flush=True)
        t0 = time.time()
        # run sweep against a reduced frame view
        full_frame = ws.frame
        ws.frame = type(full_frame)()
        for nid in sorted(sub_nodes):
            ws.frame.add_node(sub_nodes[nid])
        for e in full_frame.edges:
            if e.source_id in sub_nodes and e.target_id in sub_nodes:
                ws.frame.add_edge(e)
        ws.gs.build_from(
            [ws.frame.nodes[nid].to_row() for nid in sorted(ws.frame.nodes)],
            [e.to_row() for e in sorted(ws.frame.edges,
                                        key=lambda x: (x.source_id, x.target_id,
                                                       x.relation_type))])
        try:
            table = sweep(ws, records, k=8)
        finally:
            ws.frame = full_frame
            ws.gs.build_from(
                [full_frame.nodes[nid].to_row() for nid in sorted(full_frame.nodes)],
                [e.to_row() for e in sorted(full_frame.edges,
                                            key=lambda x: (x.source_id, x.target_id,
                                                           x.relation_type))])
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
