"""Answer-stage comparison: MIRA vs flat retrieval, same LLM answer stage.

Standalone (doesn't re-run retrieval metrics): 50-question subsample, forced
Qwen2.5-1.5B (3B cannot co-reside with the 82k-node workspace in 16 GB RAM),
identical compression + answer prompt for both systems. Writes
data_bench/bench_real_answers.json with token-F1 significance.

Exits nonzero if the LLM fails to load — never silently falls back to the
extractive answerer, that would poison the artifact.
"""
import json
import os
import random
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import core.workspace as cw  # noqa: E402
import config.auto_config as ac  # noqa: E402
cw.DATA_DIR = os.path.join(ROOT, "data_bench")
ac.DATA_DIR = cw.DATA_DIR

import models.model_manager as mm  # noqa: E402

_orig_pick = mm.pick_local_model


def _pick_15b(rc):
    for m in mm.discover_local_gguf():
        if "1.5b" in os.path.basename(m.path).lower():
            return m
    return _orig_pick(rc)


mm.pick_local_model = _pick_15b  # auto_config resolves the attr at call time

from config.auto_config import build_context  # noqa: E402
from core.workspace import Workspace  # noqa: E402
from evaluation.benchmark import answer_retrieve_fn, run_system  # noqa: E402
from evaluation.datasets import resolve_titles_to_ids  # noqa: E402
from scripts.eval_benchmark_real import _flat_answer_fn  # noqa: E402
from scripts.eval_significance import paired_bootstrap, wilcoxon  # noqa: E402

BENCH_JSON = os.path.join(cw.DATA_DIR, "benchmarks", "musique_bench.json")
OUT = os.path.join(cw.DATA_DIR, "bench_real_answers.json")

with open(BENCH_JSON, "r", encoding="utf-8") as fh:
    bench = json.load(fh)
records = [{"question": q["question"], "answer": q["answer"],
            "answer_aliases": q.get("answer_aliases") or [],
            "supporting_titles": q.get("supporting_titles") or []}
           for q in bench["questions"]]

ctx = build_context()
llm_available = bool(ctx.llm and ctx.llm.available)
print(f"llm available: {llm_available}"
      + (f" ({os.path.basename(ctx.llm.model_path)}, gpu_layers="
         f"{ctx.llm.n_gpu_layers_used}, ctx={ctx.llm.n_ctx})" if llm_available else ""))
if not llm_available:
    print("LLM did not load — refusing to produce an extractive-fallback artifact")
    sys.exit(2)

ws = Workspace(embeddings=ctx.embeddings, llm=ctx.llm, config=ctx.cfg)
for r in records:
    r["supporting_ids"] = resolve_titles_to_ids(ws, r["supporting_titles"])

rng = random.Random(77)
sub = rng.sample(records, min(50, len(records)))

systems = {
    "mira_full": answer_retrieve_fn(ws, None, k=8),
    "flat_vector": _flat_answer_fn(ws, bench["paragraphs"], k=8),
}
arows = {}
for name, fn in systems.items():
    print(f"\nanswer run: {name} ({len(sub)} q)", flush=True)
    res = run_system(name, fn, ws.embeddings, sub, k=8)
    arows[name] = res["rows"]
    agg = res["aggregate"]
    print(f"  token_f1={agg.get('answer_token_f1', 0):.4f} "
          f"mrr={agg.get('mrr', 0):.4f} recall={agg.get('retrieval_recall', 0):.4f}",
          flush=True)

bt = paired_bootstrap(arows["mira_full"], arows["flat_vector"], "answer_token_f1")
wx = wilcoxon(arows["mira_full"], arows["flat_vector"], "answer_token_f1")
print(f"\nsignificance (token_f1): bootstrap {bt}\n wilcoxon {wx}")

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump({"n": len(sub),
               "model": os.path.basename(ctx.llm.model_path),
               "gpu_layers": ctx.llm.n_gpu_layers_used, "n_ctx": ctx.llm.n_ctx,
               "rows": arows,
               "significance_token_f1": {"bootstrap": bt, "wilcoxon": wx}},
              fh, indent=1)
print(f"wrote {OUT}")
