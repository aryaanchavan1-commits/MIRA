"""IndicQA retrieval eval: mira_full vs flat_vector vs bm25 on hi + mr.

Retrieval-only (no LLM). Cross-lingual caveat: MiniLM-L6 is English-centric,
so absolute numbers are low for every system; the comparison is between
systems under the SAME embedder. Writes data_indic/indicqa_results.json.
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

from baselines.bm25 import BM25RAG  # noqa: E402
from config.auto_config import build_context  # noqa: E402
from core.workspace import Workspace  # noqa: E402
from evaluation.benchmark import baseline_retrieve_fn, mira_retrieve_fn, run_system  # noqa: E402
from evaluation.datasets import resolve_titles_to_ids  # noqa: E402
from scripts.eval_significance import paired_bootstrap, wilcoxon  # noqa: E402

OUT = os.path.join(ROOT, "data_indic", "indicqa_results.json")


def run_lang(lang: str, ctx) -> dict:
    data_dir = os.path.join(ROOT, "data_indic", lang)
    cw.DATA_DIR = data_dir
    ac.DATA_DIR = data_dir
    src = os.path.join(ROOT, "data_indic", "benchmarks", f"indicqa_{lang}_bench.json")
    with open(src, "r", encoding="utf-8") as fh:
        bench = json.load(fh)
    records = [{"question": q["question"], "answer": q["answer"],
                "supporting_titles": q["supporting_titles"]} for q in bench["questions"]]

    ws = Workspace(embeddings=ctx.embeddings, llm=None, config=ctx.cfg)
    n_gold = 0
    for r in records:
        r["supporting_ids"] = resolve_titles_to_ids(ws, r["supporting_titles"])
        n_gold += bool(r["supporting_ids"])
    print(f"[{lang}] {len(records)} questions, {len(ws.frame.nodes)} nodes, "
          f"gold {n_gold}/{len(records)}", flush=True)
    if not ws.frame.nodes or n_gold == 0:
        print(f"[{lang}] FATAL: empty workspace or zero gold — skipping", flush=True)
        return {"lang": lang, "error": "empty workspace or zero gold"}

    systems = {
        "mira_full": mira_retrieve_fn(ws, None, k=8),
        "flat_vector": baseline_retrieve_fn(
            __import__("baselines.vector_rag", fromlist=["VectorRAG"]).VectorRAG(
                ws.frame, ws.vs), k=8),
        "bm25": baseline_retrieve_fn(BM25RAG(ws.frame), k=8),
    }
    out = {"lang": lang, "n_questions": len(records),
           "n_nodes": len(ws.frame.nodes), "n_gold": n_gold, "summary": {}}
    rows_by_sys = {}
    for name, fn in systems.items():
        res = run_system(name, fn, ctx.embeddings, records, k=8)
        rows_by_sys[name] = res["rows"]
        agg = res["aggregate"]
        out["summary"][name] = {k: agg.get(k, 0) for k in
                                ("mrr", "retrieval_recall", "latency_ms")}
        print(f"  {name:12s} mrr={agg.get('mrr', 0):.4f} "
              f"recall@8={agg.get('retrieval_recall', 0):.4f}", flush=True)

    for metric in ("mrr", "retrieval_recall"):
        bt = paired_bootstrap(rows_by_sys["mira_full"], rows_by_sys["flat_vector"], metric)
        wx = wilcoxon(rows_by_sys["mira_full"], rows_by_sys["flat_vector"], metric)
        out[f"sig_mira_vs_flat_{metric}"] = {"bootstrap": bt, "wilcoxon": wx}
        print(f"  mira vs flat [{metric}]: d={bt['mean_diff']} p~{bt['p_value']}",
              flush=True)
    return out


def main() -> int:
    ctx = build_context()
    results = {}
    for lang in ("hi", "mr"):
        results[lang] = run_lang(lang, ctx)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"note": "Retrieval-only; MiniLM-L6 is English-centric, so "
                          "absolute numbers are low for all systems. The "
                          "comparison is between systems under the same embedder.",
                   "languages": results}, fh, ensure_ascii=False, indent=1)
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
