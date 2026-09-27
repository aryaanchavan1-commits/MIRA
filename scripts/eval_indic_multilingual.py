"""IndicQA with a MULTILINGUAL embedder — flips the cross-lingual result.

Same benches (hi/mr), same eval protocol as eval_indic_retrieval.py, but the
embedding backend is paraphrase-multilingual-MiniLM-L12-v2 instead of the
English-centric all-MiniLM-L6-v2. Fresh workspaces under data_indic_ml/ keep
the two runs independently reproducible.

Prediction from the paper's diagnosis: dense retrieval (MIRA and flat) should
jump from near-random (MRR 0.02-0.04) into BM25's range (0.39-0.45) or better.

Writes data_indic_ml/indicqa_ml_results.json (before/after comparison inside).
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

ML_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
ML_DATA = os.path.join(ROOT, "data_indic_ml")
BENCH_DIR = os.path.join(ROOT, "data_indic", "benchmarks")

from baselines.bm25 import BM25RAG  # noqa: E402
from models.embeddings import EmbeddingBackend  # noqa: E402
from core.workspace import Workspace  # noqa: E402
from evaluation.benchmark import baseline_retrieve_fn, mira_retrieve_fn, run_system  # noqa: E402
from evaluation.datasets import resolve_titles_to_ids  # noqa: E402
from scripts.eval_significance import paired_bootstrap, wilcoxon  # noqa: E402


def run_lang(lang: str, emb: EmbeddingBackend) -> dict:
    data_dir = os.path.join(ML_DATA, lang)
    cw.DATA_DIR = data_dir
    ac.DATA_DIR = data_dir
    src = os.path.join(BENCH_DIR, f"indicqa_{lang}_bench.json")
    with open(src, "r", encoding="utf-8") as fh:
        bench = json.load(fh)
    records = [{"question": q["question"], "answer": q["answer"],
                "supporting_titles": q["supporting_titles"]}
               for q in bench["questions"]]

    os.makedirs(data_dir, exist_ok=True)
    ctx_cfg = {"topology": {"max_rings": 5, "placement_strategy": "hybrid_mira"}}
    ws = Workspace(embeddings=emb, llm=None, config=ctx_cfg)

    n_gold = 0
    for r in records:
        r["supporting_ids"] = resolve_titles_to_ids(ws, r["supporting_titles"])
        n_gold += bool(r["supporting_ids"])
    print(f"[{lang}] {len(records)} questions, {len(ws.frame.nodes)} nodes, "
          f"gold {n_gold}/{len(records)}", flush=True)
    if not ws.frame.nodes or n_gold == 0:
        return {"lang": lang, "error": "empty workspace or zero gold"}

    systems = {
        "mira_full": mira_retrieve_fn(ws, None, k=8),
        "flat_vector": baseline_retrieve_fn(
            __import__("baselines.vector_rag", fromlist=["VectorRAG"]).VectorRAG(
                ws.frame, ws.vs), k=8),
        "bm25": baseline_retrieve_fn(BM25RAG(ws.frame), k=8),
    }
    out = {"lang": lang, "n_questions": len(records),
           "n_nodes": len(ws.frame.nodes), "n_gold": n_gold,
           "embedder": ML_MODEL, "summary": {}}
    rows_by_sys = {}
    for name, fn in systems.items():
        res = run_system(name, fn, emb, records, k=8)
        rows_by_sys[name] = res["rows"]
        agg = res["aggregate"]
        out["summary"][name] = {k: agg.get(k, 0)
                                for k in ("mrr", "retrieval_recall", "latency_ms")}
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
    allow_dl = os.environ.get("MIRA_ALLOW_DOWNLOAD", "1") == "1"
    emb = EmbeddingBackend(ML_MODEL, device="cpu", allow_download=allow_dl)
    emb.encode(["warmup"])  # force load now: fail fast if the model is unavailable
    if not emb.is_real_model:
        print(f"FATAL: multilingual model '{ML_MODEL}' unavailable: {emb.last_error}")
        return 2
    print(f"multilingual embedder: {ML_MODEL} (dim={emb.dim})", flush=True)

    results = {}
    for lang in ("hi", "mr"):
        results[lang] = run_lang(lang, emb)

    prev_path = os.path.join(ROOT, "data_indic", "indicqa_results.json")
    prev = json.load(open(prev_path, encoding="utf-8")) if os.path.exists(prev_path) else {}
    out = {"embedder": ML_MODEL, "languages": results,
           "before": {"embedder": "all-MiniLM-L6-v2 (English-centric)", "languages": prev}}
    os.makedirs(ML_DATA, exist_ok=True)
    with open(os.path.join(ML_DATA, "indicqa_ml_results.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("wrote data_indic_ml/indicqa_ml_results.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
