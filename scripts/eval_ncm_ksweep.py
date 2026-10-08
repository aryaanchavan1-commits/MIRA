"""H2 K-sweep (S1 in paper/steps.md): does any top-K beat the degenerate K=1?

Runs ncm.enabled with top_k in {1,2,3}, n=300 MuSiQue questions, 3 seeds, and
compares against the full_mira baseline rows already stored in
ncm_pilot_checkpoint.json (same questions, same seeds — paired). Per the
protocol: K>1 must beat K=1 AND full MIRA, or H2 is rejected on this benchmark.

Usage: .venv/Scripts/python.exe scripts/eval_ncm_ksweep.py
"""
from __future__ import annotations

import copy
import json
import os
import random
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
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
CKPT = os.path.join(BENCH_DIR, "ncm_pilot_checkpoint.json")
OUT = os.path.join(BENCH_DIR, "ncm_ksweep_results.json")
N, SEEDS, K = 300, 3, 8


def main() -> int:
    with open(BENCH_JSON, "r", encoding="utf-8") as fh:
        bench = json.load(fh)
    records = [{"question": q["question"], "answer": q["answer"],
                "supporting_titles": q.get("supporting_titles") or []}
               for q in bench["questions"]]

    ctx = build_context(load_llm=False)
    ws = Workspace(embeddings=ctx.embeddings, llm=ctx.llm, config=ctx.cfg)
    from evaluation.datasets import resolve_titles_to_ids
    for r in records:
        r["supporting_ids"] = resolve_titles_to_ids(ws, r["supporting_titles"])

    # baseline rows: full_mira from the pilot checkpoint (same n/seeds/questions)
    with open(CKPT, "r", encoding="utf-8") as fh:
        base_pool = {k: v for k, v in json.load(fh)["pool"].items()
                     if k == "full_mira"}
    base_rows = base_pool.get("full_mira", [])
    if len(base_rows) < N * SEEDS:
        print(f"baseline incomplete: {len(base_rows)} rows — run the pilot first")
        return 1

    pool = {f"k{k}": [] for k in (1, 2, 3)}
    ckpt = os.path.join(BENCH_DIR, "ncm_ksweep_checkpoint.json")
    done = set()
    per_seed = []
    if os.path.exists(ckpt):
        with open(ckpt, "r", encoding="utf-8") as fh:
            ck = json.load(fh)
        per_seed = ck["per_seed"]
        pool = ck["pool"]
        done = {(p["seed"], p["condition"]) for p in per_seed}
        print(f"resuming: {sorted(done)}")
    for seed in range(SEEDS):
        rng = random.Random(2000 + seed)
        sub = records if N >= len(records) else rng.sample(records, N)
        for k in (1, 2, 3):
            if (seed, f"k{k}") in done:
                continue
            ws.config = copy.deepcopy(ctx.cfg)
            ws.config["ncm"] = {"enabled": True, "top_k": k}
            t0 = time.time()
            try:
                res = run_system(f"k{k}", mira_retrieve_fn(ws, None, k=K),
                                 ws.embeddings, sub, k=K)
            except Exception as exc:  # noqa: BLE001 — record and continue
                print(f"  k{k} seed{seed} FAILED: {exc}")
                continue
            pool[f"k{k}"].extend(res["rows"])
            agg = res["aggregate"]
            per_seed.append({"seed": seed, "condition": f"k{k}", **agg})
            print(f"  k{k} seed{seed}   mrr={agg.get('mrr', 0):.4f} "
                  f"recall={agg.get('retrieval_recall', 0):.4f} ({time.time() - t0:.0f}s)")
            with open(ckpt, "w", encoding="utf-8") as fh:
                json.dump({"per_seed": per_seed, "pool": pool}, fh)
    ws.config = ctx.cfg

    print("\n=== H2 K-sweep (delta vs full_mira; positive = worse) ===")
    table = []
    for k in (1, 2, 3):
        rows = pool.get(f"k{k}", [])
        if not rows:
            continue
        mrr = sum(r.get("mrr", 0) for r in rows) / max(1, len(rows))
        rec = sum(r.get("retrieval_recall", 0) for r in rows) / max(1, len(rows))
        bt = paired_bootstrap(base_rows, rows, "mrr")
        table.append({"condition": f"top_k={k}", "mrr": round(mrr, 4),
                      "recall@8": round(rec, 4), "n": len(rows),
                      "delta_vs_full": bt["mean_diff"], "p": bt["p_value"],
                      "ci95": [bt["ci_low"], bt["ci_high"]]})
        print(f"  top_k={k}  mrr={mrr:.4f} recall={rec:.4f} "
              f"d={bt['mean_diff']} p={bt['p_value']}")
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"n": N, "seeds": SEEDS, "label": "H2 K-sweep",
                   "table": table}, fh, indent=1)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
