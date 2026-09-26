"""Full component ablation on real MuSiQue data.

For each of the 9 retrieval components, run MIRA without it (plus the
vector_only floor and the full system), N questions, 3 seeds. Reports the
per-component MRR/recall drop vs full MIRA with paired-bootstrap
significance. Retrieval-level (no LLM) — deterministic given the seed.

Usage: .venv/Scripts/python.exe scripts/eval_ablation_real.py --n 300 --seeds 3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.workspace as cw  # noqa: E402
cw.DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data_bench")
BENCH_DIR = cw.DATA_DIR

from config.auto_config import build_context  # noqa: E402
from core.retrieval import ALL_COMPONENTS  # noqa: E402
from core.workspace import Workspace  # noqa: E402
from evaluation.benchmark import mira_retrieve_fn, run_system  # noqa: E402
from scripts.eval_significance import paired_bootstrap  # noqa: E402

BENCH_JSON = os.path.join(BENCH_DIR, "benchmarks", "musique_bench.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    with open(BENCH_JSON, "r", encoding="utf-8") as fh:
        bench = json.load(fh)
    records = [{"question": q["question"], "answer": q["answer"],
                "supporting_titles": q.get("supporting_titles") or []}
               for q in bench["questions"]]

    ctx = build_context()
    ws = Workspace(embeddings=ctx.embeddings, llm=ctx.llm, config=ctx.cfg)
    from evaluation.datasets import resolve_titles_to_ids  # noqa: E402
    for r in records:
        r["supporting_ids"] = resolve_titles_to_ids(ws, r["supporting_titles"])
    print(f"{len(records)} questions, {len(ws.frame.nodes)} nodes, "
          f"gold resolved: {sum(1 for r in records if r['supporting_ids'])}")

    # conditions: full + leave-one-out for every component + vector floor
    conditions = {"full_mira": None, "vector_only": ("semantic",)}
    for comp in ALL_COMPONENTS:
        conditions[f"minus_{comp}"] = tuple(c for c in ALL_COMPONENTS if c != comp)

    per_seed = []
    pool = {name: [] for name in conditions}
    for seed in range(args.seeds):
        rng = random.Random(2000 + seed)
        sub = records if args.n >= len(records) else rng.sample(records, args.n)
        print(f"\n=== seed {seed} ({len(sub)} questions) ===")
        for name, active in conditions.items():
            t0 = time.time()
            res = run_system(name, mira_retrieve_fn(ws, active, k=args.k),
                             ws.embeddings, sub, k=args.k)
            pool[name].extend(res["rows"])
            agg = res["aggregate"]
            per_seed.append({"seed": seed, "condition": name, **agg})
            print(f"  {name:20s} mrr={agg.get('mrr', 0):.4f} "
                  f"recall={agg.get('retrieval_recall', 0):.4f} ({time.time() - t0:.0f}s)")

    print("\n=== ablation table (mean over seeds; delta vs full with significance) ===")
    table = []
    full_rows = pool["full_mira"]
    for name, rows in pool.items():
        mrr = sum(r.get("mrr", 0) for r in rows) / max(1, len(rows))
        rec = sum(r.get("retrieval_recall", 0) for r in rows) / max(1, len(rows))
        entry = {"condition": name, "mrr": round(mrr, 4),
                 "recall@8": round(rec, 4), "n": len(rows)}
        if name != "full_mira":
            bt = paired_bootstrap(full_rows, rows, "mrr")
            entry["mrr_delta_vs_full"] = bt["mean_diff"]
            entry["mrr_ci95"] = [bt["ci_low"], bt["ci_high"]]
            entry["mrr_p"] = bt["p_value"]
        table.append(entry)
    table.sort(key=lambda e: -e["mrr"])
    for e in table:
        star = ""
        if "mrr_p" in e and e["mrr_p"] < 0.05:
            star = " *" if e.get("mrr_delta_vs_full", 0) > 0 else " (sig. worse)"
        print(f"  {e['condition']:20s} mrr={e['mrr']:.4f} recall={e['recall@8']:.4f}"
              + (f"  d={e.get('mrr_delta_vs_full')}" if "mrr_delta_vs_full" in e else "")
              + star)

    outp = os.path.join(BENCH_DIR, "ablation_real_results.json")
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump({"config": vars(args), "n_nodes": len(ws.frame.nodes),
                   "per_seed": per_seed, "table": table}, fh, indent=1)
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
