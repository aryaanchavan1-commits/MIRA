"""NCM pilot evaluation — FIRST measured data points for H2/H6 (exploratory).

Protocol status: EXPLORATORY PILOT, not a confirmatory H2/H6 test (docs/
EXPERIMENT_PROTOCOL.md). n is small and one seed; no claim is made from these
numbers alone. A confirmatory run needs the protocol's full n and seed count.

Conditions (paired, same questions per seed):
  full_mira    — classic MIRA, ncm disabled (the §1 fair baseline)
  ncm_linear   — ncm.enabled, linear overlap kernel, default weights
  ncm_born     — ncm.enabled, Born-rule (quantum-inspired) kernel, H6 pilot

Usage:
  .venv/Scripts/python.exe scripts/eval_ncm_pilot.py --n 150 --seeds 1
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time

if hasattr(sys.stdout, "reconfigure"):  # cp1252 consoles choke on fancy glyphs
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.workspace as cw  # noqa: E402
import config.auto_config as ac  # noqa: E402
cw.DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data_bench")
ac.DATA_DIR = cw.DATA_DIR
BENCH_DIR = cw.DATA_DIR

from config.auto_config import build_context  # noqa: E402
from core.workspace import Workspace  # noqa: E402
from evaluation.benchmark import mira_retrieve_fn, run_system  # noqa: E402
from scripts.eval_significance import paired_bootstrap  # noqa: E402

BENCH_JSON = os.path.join(BENCH_DIR, "benchmarks", "musique_bench.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    with open(BENCH_JSON, "r", encoding="utf-8") as fh:
        bench = json.load(fh)
    records = [{"question": q["question"], "answer": q["answer"],
                "supporting_titles": q.get("supporting_titles") or []}
               for q in bench["questions"]]

    ctx = build_context(load_llm=False)  # retrieval-only pilot: no LLM load
    ws = Workspace(embeddings=ctx.embeddings, llm=ctx.llm, config=ctx.cfg)
    from evaluation.datasets import resolve_titles_to_ids  # noqa: E402
    for r in records:
        r["supporting_ids"] = resolve_titles_to_ids(ws, r["supporting_titles"])
    print(f"pilot: {len(records)} questions available, {len(ws.frame.nodes)} nodes, "
          f"gold resolved: {sum(1 for r in records if r['supporting_ids'])}")

    base_cfg = copy.deepcopy(ws.config)
    conditions = [("full_mira", None), ("ncm_linear", "linear"), ("ncm_born", "born")]
    pool: dict = {}
    per_seed = []
    for seed in range(args.seeds):
        rng = random.Random(2000 + seed)
        sub = records if args.n >= len(records) else rng.sample(records, args.n)
        print(f"\n=== seed {seed} ({len(sub)} questions) ===")
        for name, backend in conditions:
            # fresh config per condition; a fresh retrieve fn rebuilds the
            # pipeline with it (mira_retrieve_fn pins config at first query)
            ws.config = copy.deepcopy(base_cfg)
            if backend is not None:
                ws.config["ncm"] = {"enabled": True, "backend": backend}
            t0 = time.time()
            res = run_system(name, mira_retrieve_fn(ws, None, k=args.k),
                             ws.embeddings, sub, k=args.k)
            pool.setdefault(name, []).extend(res["rows"])
            agg = res["aggregate"]
            per_seed.append({"seed": seed, "condition": name, **agg})
            print(f"  {name:12s} mrr={agg.get('mrr', 0):.4f} "
                  f"recall={agg.get('retrieval_recall', 0):.4f} "
                  f"({time.time() - t0:.0f}s)")
            with open(os.path.join(BENCH_DIR, "ncm_pilot_checkpoint.json"), "w",
                      encoding="utf-8") as fh:
                json.dump({"config": vars(args), "per_seed": per_seed,
                           "pool": pool}, fh)
    ws.config = base_cfg

    print("\n=== pilot table (delta vs full_mira, paired bootstrap) ===")
    table = []
    full_rows = pool["full_mira"]
    for name, _ in conditions:
        rows = pool[name]
        mrr = sum(r.get("mrr", 0) for r in rows) / max(1, len(rows))
        rec = sum(r.get("retrieval_recall", 0) for r in rows) / max(1, len(rows))
        entry = {"condition": name, "mrr": round(mrr, 4),
                 "recall@8": round(rec, 4), "n": len(rows)}
        if name != "full_mira":
            bt = paired_bootstrap(full_rows, rows, "mrr")
            entry.update({"mrr_delta_vs_full": bt["mean_diff"],
                          "mrr_ci95": [bt["ci_low"], bt["ci_high"]],
                          "mrr_p": bt["p_value"]})
        table.append(entry)
    for e in table:
        print(f"  {e['condition']:12s} mrr={e['mrr']:.4f} recall={e['recall@8']:.4f}"
              + (f"  d={e.get('mrr_delta_vs_full')} p={e.get('mrr_p')}" if e.get("mrr_p") is not None else ""))

    outp = os.path.join(BENCH_DIR, "ncm_pilot_results.json")
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump({"config": vars(args), "label": "exploratory pilot",
                   "n_nodes": len(ws.frame.nodes), "per_seed": per_seed,
                   "table": table}, fh, indent=1)
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
