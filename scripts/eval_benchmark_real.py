"""Real-data benchmark: MIRA vs flat-vector baseline on MuSiQue.

3 seeds x N questions, retrieval + answer-level metrics, paired bootstrap
and Wilcoxon significance. Uses the ISOLATED bench workspace
(data_bench/, built by scripts/build_bench_workspace.py).

Retrieval runs are cheap (no LLM). The answer-level run needs the LLM and
is opt-in via --answers; it evaluates a 50-question subsample (same subset
ids for all systems) so CPU/VRAM stays viable.

Usage:
  .venv/Scripts/python.exe scripts/eval_benchmark_real.py --n 300 --seeds 3
  .venv/Scripts/python.exe scripts/eval_benchmark_real.py --answers --n-answers 50
"""
from __future__ import annotations

import argparse
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
ac.DATA_DIR = cw.DATA_DIR  # ingestion/pipeline reads auto_config.DATA_DIR at call time
BENCH_DIR = cw.DATA_DIR

from config.auto_config import build_context  # noqa: E402
from core.workspace import Workspace  # noqa: E402
from evaluation.benchmark import (baseline_retrieve_fn, mira_retrieve_fn,  # noqa: E402
                                  run_system)
from scripts.eval_significance import paired_bootstrap, wilcoxon  # noqa: E402

BENCH_JSON = os.path.join(BENCH_DIR, "benchmarks", "musique_bench.json")
OUT_DIR = BENCH_DIR


def load_records() -> list:
    with open(BENCH_JSON, "r", encoding="utf-8") as fh:
        bench = json.load(fh)
    titles = bench["paragraphs"]
    records = []
    for q in bench["questions"]:
        records.append({"question": q["question"], "answer": q["answer"],
                        "answer_aliases": q.get("answer_aliases") or [],
                        "hops": q.get("hops", 2),
                        "supporting_titles": q.get("supporting_titles") or [],
                        "all_titles": [p["title"] for p in q.get("paragraphs", [])]})
    return records, titles


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--answers", action="store_true",
                    help="also run the LLM answer-level comparison subsample")
    ap.add_argument("--n-answers", type=int, default=50)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    records, titles = load_records()
    print(f"{len(records)} questions available, corpus {len(titles)} paragraphs")

    ctx = build_context()
    ws = Workspace(embeddings=ctx.embeddings, llm=ctx.llm, config=ctx.cfg)
    print(f"bench workspace: {len(ws.frame.nodes)} nodes, vs={ws.vs is not None}")

    # resolve gold node ids once via title->doc->chunk->node
    from evaluation.datasets import resolve_titles_to_ids  # noqa: E402
    title_map = {t: txt for t, txt in titles.items()}
    for r in records:
        r["supporting_ids"] = resolve_titles_to_ids(ws, r["supporting_titles"])
    n_gold = sum(1 for r in records if r["supporting_ids"])
    print(f"gold node ids resolved for {n_gold}/{len(records)} questions")

    # --- systems ---
    systems = {}
    vec = __import__("baselines.vector_rag", fromlist=["VectorRAG"]).VectorRAG(
        ws.frame, ws.vs)
    hier = __import__("baselines.hierarchical_rag", fromlist=["HierarchicalRAG"]).HierarchicalRAG(
        ws.frame)
    systems["flat_vector"] = baseline_retrieve_fn(vec, k=args.k)
    systems["hierarchical_rag"] = baseline_retrieve_fn(hier, k=args.k)
    systems["mira_full"] = mira_retrieve_fn(ws, None, k=args.k)

    seed_summaries = []
    all_rows = {}
    for seed in range(args.seeds):
        rng = random.Random(1000 + seed)
        sub = records if args.n >= len(records) else rng.sample(records, args.n)
        print(f"\n=== seed {seed}: {len(sub)} questions ===")
        for name, fn in systems.items():
            t0 = time.time()
            res = run_system(name, fn, ws.embeddings, sub, k=args.k)
            all_rows[(seed, name)] = res["rows"]
            agg = res["aggregate"]
            print(f"  {name:18s} mrr={agg.get('mrr', 0):.4f} "
                  f"recall@8={agg.get('retrieval_recall', 0):.4f} "
                  f"({time.time() - t0:.0f}s)")
            seed_summaries.append({"seed": seed, "system": name, **agg})
            # incremental checkpoint after each system×seed (this box has form)
            with open(os.path.join(OUT_DIR, "bench_real_checkpoint.json"), "w",
                      encoding="utf-8") as fh:
                json.dump({"config": {"n": args.n, "seeds": args.seeds, "k": args.k},
                           "seed_summaries": seed_summaries,
                           "rows": {"|".join(map(str, key)): v
                                    for key, v in all_rows.items()}}, fh)

    # --- aggregate over seeds ---
    def mean_over_seeds(name, key):
        vals = [s[key] for s in seed_summaries if s["system"] == name and key in s]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    print("\n=== mean over seeds (retrieval) ===")
    summary = {}
    for name in systems:
        summary[name] = {k: mean_over_seeds(name, k)
                         for k in ("mrr", "retrieval_recall", "latency_ms",
                                   "n_candidates", "context_tokens")}
        print(f"  {name:18s} " + "  ".join(f"{k}={v}" for k, v in summary[name].items()))

    # --- significance: mira vs flat_vector on pooled rows ---
    mira_rows = [r for (seed, name), rows in all_rows.items() if name == "mira_full" for r in rows]
    flat_rows = [r for (seed, name), rows in all_rows.items() if name == "flat_vector" for r in rows]
    sig = {}
    for metric in ("mrr", "retrieval_recall"):
        bt = paired_bootstrap(mira_rows, flat_rows, metric)
        wx = wilcoxon(mira_rows, flat_rows, metric)
        sig[metric] = {"bootstrap": bt, "wilcoxon": wx}
        print(f"\nsignificance mira_full vs flat_vector [{metric}]:")
        print(f"  paired bootstrap: mean_diff={bt['mean_diff']} "
              f"95% CI [{bt['ci_low']}, {bt['ci_high']}] p~{bt['p_value']}")
        print(f"  wilcoxon: {wx}")

    out = {"config": {"n": args.n, "seeds": args.seeds, "k": args.k,
                      "corpus_paragraphs": len(titles), "n_nodes": len(ws.frame.nodes),
                      "n_gold_resolved": n_gold},
           "seed_summaries": seed_summaries,
           "summary": summary, "significance": sig}
    outp = os.path.join(OUT_DIR, "bench_real_results.json")
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {outp}")

    # --- optional answer-level run ---
    if args.answers:
        if ctx.llm is None or not ctx.llm.available:
            print("LLM unavailable -- skipping answer-level run")
            return 0
        rng = random.Random(77)
        sub = rng.sample(records, min(args.n_answers, len(records)))
        from evaluation.benchmark import answer_retrieve_fn  # noqa: E402
        ans_systems = {
            "flat_vector": _flat_answer_fn(ws, titles, k=args.k),
            "mira_full": answer_retrieve_fn(ws, None, k=args.k),
        }
        arows = {}
        for name, fn in ans_systems.items():
            print(f"\nanswer run: {name} ({len(sub)} q)")
            res = run_system(name, fn, ws.embeddings, sub, k=args.k)
            arows[name] = res["rows"]
            agg = res["aggregate"]
            print(f"  token_f1={agg.get('answer_token_f1', 0):.4f} "
                  f"mrr={agg.get('mrr', 0):.4f} recall={agg.get('retrieval_recall', 0):.4f}")
        bt = paired_bootstrap(arows["mira_full"], arows["flat_vector"], "answer_token_f1")
        wx = wilcoxon(arows["mira_full"], arows["flat_vector"], "answer_token_f1")
        print(f"\nsignificance (token_f1): bootstrap {bt}\n wilcoxon {wx}")
        with open(os.path.join(OUT_DIR, "bench_real_answers.json"), "w", encoding="utf-8") as fh:
            json.dump({"n": len(sub), "rows": {k: v for k, v in arows.items()},
                       "significance_token_f1": {"bootstrap": bt, "wilcoxon": wx}}, fh, indent=1)
        print("wrote bench_real_answers.json")
    return 0


def _flat_answer_fn(ws, titles, k: int = 8):
    """Flat-vector retrieval + the SAME compression/LLM answer stage as MIRA,
    so the answer comparison isolates retrieval quality only."""
    from evaluation.benchmark import _workspace_guard  # noqa: E402

    def fn(question, qvec):
        from core.compression import compress  # noqa: E402
        from core.answer import (AnswerPipeline, _ANSWER_SYSTEM, parse_budget,  # noqa: E402
                                 _is_no_evidence_text)
        from models.llm import answer_prompt  # noqa: E402
        with _workspace_guard(ws):
            pipe = AnswerPipeline(ws.frame, ws.vs, ws.gs, ws.embeddings,
                                  llm=ws.llm, config=ws.config,
                                  doc_titles=ws._doc_titles(),
                                  operation_lock=getattr(ws, "_lock", None))
            res = pipe.retriever.retrieve(question, qvec, active_components=("semantic",),
                                          final_k=k)
            cctx = compress(res, max_tokens=parse_budget(ws.config), target_ratio=0.6)
            if not cctx.text.strip() or ws.llm is None:
                return {"node_ids": [], "selected_evidence_ids": [],
                        "n_retrieved": len(res.items), "n_candidates": res.n_candidates,
                        "answer": None, "context_text": cctx.text,
                        "context_tokens": cctx.n_tokens, "latency_ms": res.latency_ms,
                        "metrics": {}, "active_components": ["semantic"],
                        "candidate_policy": "shared"}
            text = ws.llm.chat([{"role": "system", "content": _ANSWER_SYSTEM},
                                {"role": "user", "content": answer_prompt(cctx.text, question)}],
                               max_tokens=220, temperature=0.0) or ""
            if _is_no_evidence_text(text):
                text = None
            sel = list(dict.fromkeys(
                u.get("node_id") for u in cctx.units if u.get("node_id")))
            return {"node_ids": sel, "selected_evidence_ids": sel,
                    "n_retrieved": len(res.items), "n_candidates": res.n_candidates,
                    "answer": text, "context_text": cctx.text,
                    "context_tokens": cctx.n_tokens,
                    "latency_ms": res.latency_ms, "metrics": {},
                    "active_components": ["semantic"], "candidate_policy": "shared"}
    return fn


if __name__ == "__main__":
    raise SystemExit(main())
