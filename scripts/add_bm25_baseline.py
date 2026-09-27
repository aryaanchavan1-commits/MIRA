"""Add BM25 to the completed MuSiQue retrieval benchmark (no re-runs).

Loads the checkpoint, runs BM25 on the same 300 questions, merges rows into
the final artifact, recomputes significance (mira vs bm25), and rewrites
data_bench/bench_real_results.json. Deterministic: no seeds involved.
"""
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import core.workspace as cw  # noqa: E402
import config.auto_config as ac  # noqa: E402
cw.DATA_DIR = os.path.join(ROOT, "data_bench")
ac.DATA_DIR = cw.DATA_DIR

from baselines.bm25 import BM25RAG  # noqa: E402
from config.auto_config import build_context  # noqa: E402
from core.workspace import Workspace  # noqa: E402
from evaluation.benchmark import baseline_retrieve_fn, run_system  # noqa: E402
from scripts.eval_significance import paired_bootstrap, wilcoxon  # noqa: E402

BENCH_JSON = os.path.join(cw.DATA_DIR, "benchmarks", "musique_bench.json")
RESULTS = os.path.join(cw.DATA_DIR, "bench_real_results.json")

with open(BENCH_JSON, "r", encoding="utf-8") as fh:
    bench = json.load(fh)
records = [{"question": q["question"], "answer": q["answer"],
            "supporting_titles": q.get("supporting_titles") or []}
           for q in bench["questions"]]

ctx = build_context()
ws = Workspace(embeddings=ctx.embeddings, llm=None, config=ctx.cfg)
from evaluation.datasets import resolve_titles_to_ids  # noqa: E402
for r in records:
    r["supporting_ids"] = resolve_titles_to_ids(ws, r["supporting_titles"])
n_gold = sum(1 for r in records if r["supporting_ids"])
print(f"{len(records)} questions, {len(ws.frame.nodes)} nodes, gold: {n_gold}")

bm25 = BM25RAG(ws.frame)
print(f"bm25 index built in {bm25.build_ms} ms over {bm25.N} docs")
res = run_system("bm25", baseline_retrieve_fn(bm25, k=8),
                 ws.embeddings, records, k=8)
agg = res["aggregate"]
print(f"bm25  mrr={agg.get('mrr', 0):.4f} recall@8={agg.get('retrieval_recall', 0):.4f} "
      f"latency={agg.get('latency_ms', 0):.1f}ms")

with open(RESULTS, "r", encoding="utf-8") as fh:
    out = json.load(fh)
out["summary"]["bm25"] = {k: agg.get(k, 0) for k in
                          ("mrr", "retrieval_recall", "latency_ms",
                           "n_candidates", "context_tokens")}
out["summary"]["bm25"]["index_build_ms"] = bm25.build_ms
out.setdefault("rows", {})["bm25"] = res["rows"]
# per-question mira rows live in the checkpoint (the recovered results JSON
# keeps aggregates only)
with open(os.path.join(cw.DATA_DIR, "bench_real_checkpoint.json"),
          "r", encoding="utf-8") as fh:
    ckpt_rows = json.load(fh).get("rows", {})
mira = [r for key, rs in ckpt_rows.items()
        if "|" in key and key.split("|")[1] == "mira_full" for r in rs]
assert mira, "no mira rows in checkpoint"
for metric in ("mrr", "retrieval_recall"):
    bt = paired_bootstrap(mira, res["rows"], metric)
    wx = wilcoxon(mira, res["rows"], metric)
    out["significance"][f"mira_vs_bm25_{metric}"] = {"bootstrap": bt, "wilcoxon": wx}
    print(f"significance mira_full vs bm25 [{metric}]: d={bt['mean_diff']} "
          f"CI [{bt['ci_low']}, {bt['ci_high']}] p~{bt['p_value']}")
with open(RESULTS, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print(f"updated {RESULTS}")
