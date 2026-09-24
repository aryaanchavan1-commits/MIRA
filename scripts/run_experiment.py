"""Headless experiment runner (reproducibility path).

Example:
    .venv/Scripts/python.exe scripts/run_experiment.py \
        --dataset data/datasets/custom.json --format custom \
        --k 8 --limit 50 --judge --name baseline-vs-mira

Everything runs on the real workspace; results are saved with config,
hardware, seed, versions, and git commit (see evaluation/report.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a MIRA benchmark experiment")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--format", default="custom",
                    choices=["custom", "hotpotqa", "2wiki", "musique"])
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--judge", action="store_true", help="score answers with the local LLM as judge")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--name", default="cli-run")
    args = ap.parse_args()

    from config.auto_config import build_context
    from core.workspace import Workspace
    from evaluation.ablation import run_all_systems
    from evaluation.benchmark import answer_retrieve_fn, run_system
    from evaluation.datasets import load_any
    from evaluation.report import comparison_table, save_experiment

    ctx = build_context()
    ws = Workspace(embeddings=ctx.embeddings, llm=ctx.llm, config=ctx.cfg)
    records = load_any(args.dataset, args.format, ws=ws, limit=args.limit)
    if not records:
        sys.exit(f"no usable records in {args.dataset}")
    print(f"[mira] {len(records)} questions · k={args.k} · "
          f"llm={'on' if ctx.has_llm else 'fallback'} · "
          f"embeddings={ctx.embeddings.info()['backend']}")

    results = run_all_systems(ws, records, k=args.k,
                              include_baselines=not args.no_baselines,
                              include_ablations=not args.no_ablations)
    judge_summary = None
    if args.judge:
        if not ctx.has_llm:
            print("[mira] judge requested but no LLM available — skipping")
        else:
            ans_res = run_system("full_mira_answered", answer_retrieve_fn(ws, k=args.k),
                                 ws.embeddings, records, k=args.k)
            results["full_mira_answered"] = ans_res
            from evaluation.judge import LLMJudge, judge_records
            judge_summary = judge_records(
                LLMJudge(ctx.llm), ans_res["rows"],
                answers={i: r.get("evidence_text", "") for i, r in enumerate(ans_res["rows"])})
            print(f"[mira] judge: {judge_summary}")

    exp_id = save_experiment(args.name,
                             {"dataset": args.dataset, "format": args.format,
                              "k": args.k, "limit": args.limit, "judge": bool(judge_summary)},
                             results, store=ws.store, hw=ctx.hw)
    print(f"[mira] experiment saved: experiments/{exp_id}/")
    print(f"{'system':>22} | recall |   mrr  |  ctx   |  ms")
    for row in comparison_table(results):
        print(f"{row['system']:>22} | {str(row['retrieval_recall']):>6} | "
              f"{str(row['mrr']):>6} | {str(row['context_tokens']):>6} | "
              f"{str(row['latency_ms']):>6}")
    ws.close()


if __name__ == "__main__":
    main()
