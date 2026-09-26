"""Recover final benchmark results from bench_real_checkpoint.json (no re-run).

The 3-seed retrieval run had already completed when the process died on a
cp1252 print; the incremental checkpoint holds all 9 system-seed row sets.
Recomputes seed means + significance and writes bench_real_results.json.
"""
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.eval_significance import paired_bootstrap, wilcoxon  # noqa: E402

CKPT = os.path.join(ROOT, "data_bench", "bench_real_checkpoint.json")
OUT = os.path.join(ROOT, "data_bench", "bench_real_results.json")

with open(CKPT, "r", encoding="utf-8") as fh:
    ckpt = json.load(fh)

rows = ckpt["rows"]
summaries = ckpt["seed_summaries"]
systems = sorted({s["system"] for s in summaries})
print("systems:", systems, "| summary blocks:", len(summaries))

# --- mean over seeds (same as the runner) ---
summary = {}
for name in systems:
    keys = ("mrr", "retrieval_recall", "latency_ms", "n_candidates", "context_tokens")
    summary[name] = {}
    for key in keys:
        vals = [s[key] for s in summaries if s["system"] == name and key in s]
        summary[name][key] = round(sum(vals) / len(vals), 4) if vals else 0.0
    print(f"  {name:18s} " + "  ".join(f"{k}={v}" for k, v in summary[name].items()))

# --- significance: mira_full vs flat_vector on pooled rows ---
mira = [r for key, rs in rows.items() if key.split("|")[1] == "mira_full" for r in rs]
flat = [r for key, rs in rows.items() if key.split("|")[1] == "flat_vector" for r in rs]
print(f"pooled rows: mira={len(mira)} flat={len(flat)}")

sig = {}
for metric in ("mrr", "retrieval_recall"):
    bt = paired_bootstrap(mira, flat, metric)
    wx = wilcoxon(mira, flat, metric)
    sig[metric] = {"bootstrap": bt, "wilcoxon": wx}
    print(f"\nsignificance mira_full vs flat_vector [{metric}]:")
    print(f"  paired bootstrap: mean_diff={bt['mean_diff']} "
          f"95% CI [{bt['ci_low']}, {bt['ci_high']}] p~{bt['p_value']}")
    print(f"  wilcoxon: {wx}")

out = {"config": ckpt.get("config", {}),
       "recovered_from_checkpoint": True,
       "seed_summaries": summaries,
       "summary": summary, "significance": sig}
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print(f"\nwrote {OUT}")
