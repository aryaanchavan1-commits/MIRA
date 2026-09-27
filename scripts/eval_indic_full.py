"""10x IndicQA evaluation: ALL questions over the FULL paragraph corpus.

IndicQA hi/mr ship ~261/250 SQuAD-style paragraphs with ~4-5 questions each;
the 120-question benches sampled a subset. This script uses EVERY question and
EVERY paragraph (corpus ~261/250 docs, questions ~1190/~1145 — the 10x
question scale) with the multilingual embedder, on fresh workspaces under
data_indic_ml/full_{lang}.

Writes data_indic_ml/indicqa_full_results.json with the 120-question results
inlined for comparison.
"""
import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import core.workspace as cw  # noqa: E402
import config.auto_config as ac  # noqa: E402

ML_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
SRC = os.path.join(ROOT, ".tmp", "datasets")
OUT_DIR = os.path.join(ROOT, "data_indic_ml")
BENCH_DIR = os.path.join(OUT_DIR, "benchmarks")
FULL_DATA = os.path.join(OUT_DIR, "full")

from baselines.bm25 import BM25RAG  # noqa: E402
from models.embeddings import EmbeddingBackend  # noqa: E402
from core.workspace import Workspace  # noqa: E402
from evaluation.benchmark import baseline_retrieve_fn, mira_retrieve_fn, run_system  # noqa: E402
from evaluation.datasets import resolve_titles_to_ids  # noqa: E402
from scripts.eval_significance import paired_bootstrap, wilcoxon  # noqa: E402


def build_bench(lang: str) -> str:
    """ALL questions + ALL paragraphs, same bench schema."""
    with open(os.path.join(SRC, f"indicqa.{lang}.json"), encoding="utf-8") as fh:
        data = json.load(fh)["data"]
    title_key = lambda ctx: ctx.strip()[:60]
    titles, questions = {}, []
    seen_para = set()
    for art in data:
        for p in art["paragraphs"]:
            key = title_key(p["context"])
            if key not in titles:
                titles[key] = p["context"][:1500]
            seen_para.add(key)
            for q in p["qas"]:
                if not q.get("answers"):
                    continue
                questions.append({
                    "id": q["id"], "question": q["question"],
                    "answer": q["answers"][0]["text"], "answer_aliases": [],
                    "hops": 1, "supporting_titles": [key],
                    "paragraphs": [],  # filled at eval time from the shared corpus
                })
    os.makedirs(BENCH_DIR, exist_ok=True)
    out = os.path.join(BENCH_DIR, f"indicqa_{lang}_full_bench.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"dataset": f"indicqa_{lang}_full", "config": lang,
                   "split": "full", "questions": questions,
                   "paragraphs": titles}, fh, ensure_ascii=False)
    print(f"[{lang}] full bench: {len(questions)} questions, "
          f"{len(titles)} paragraphs -> {out}", flush=True)
    return out


def ingest(lang: str, src: str, emb) -> None:
    data_dir = os.path.join(FULL_DATA, lang)
    cw.DATA_DIR = data_dir
    ac.DATA_DIR = data_dir
    with open(src, "r", encoding="utf-8") as fh:
        bench = json.load(fh)
    if os.path.exists(os.path.join(data_dir, "mira.db")):
        print(f"[{lang}] workspace exists — skipping ingest", flush=True)
        return
    os.makedirs(data_dir, exist_ok=True)
    ws = Workspace(embeddings=emb, llm=None, config={})
    from ingestion.pipeline import IngestionPipeline  # noqa: E402
    pipe = IngestionPipeline(ws.store, emb, llm=None,
                             config=ws._pipe_config(), bulk_mode=True)
    t0 = time.time()
    for title, text in bench["paragraphs"].items():
        pipe.ingest_text(text, title=f"IndicQA-{lang}: {title}",
                         source_path=f"(indicqa.{lang})")
    flushed = pipe.flush_bulk_vectors()
    ws.reload()
    print(f"[{lang}] ingested {len(bench['paragraphs'])} paragraphs, "
          f"{len(ws.frame.nodes)} nodes, {flushed} vectors "
          f"in {time.time() - t0:.0f}s", flush=True)


def evaluate(lang: str, src: str, emb) -> dict:
    data_dir = os.path.join(FULL_DATA, lang)
    cw.DATA_DIR = data_dir
    ac.DATA_DIR = data_dir
    with open(src, "r", encoding="utf-8") as fh:
        bench = json.load(fh)
    records = [{"question": q["question"], "answer": q["answer"],
                "supporting_titles": q["supporting_titles"]}
               for q in bench["questions"]]
    ws = Workspace(embeddings=emb, llm=None, config={})
    n_gold = 0
    for r in records:
        r["supporting_ids"] = resolve_titles_to_ids(ws, r["supporting_titles"])
        n_gold += bool(r["supporting_ids"])
    print(f"[{lang}] eval: {len(records)} questions, {len(ws.frame.nodes)} nodes, "
          f"gold {n_gold}/{len(records)}", flush=True)

    systems = {
        "mira_full": mira_retrieve_fn(ws, None, k=8),
        "flat_vector": baseline_retrieve_fn(
            __import__("baselines.vector_rag", fromlist=["VectorRAG"]).VectorRAG(
                ws.frame, ws.vs), k=8),
        "bm25": baseline_retrieve_fn(BM25RAG(ws.frame), k=8),
    }
    out = {"lang": lang, "n_questions": len(records),
           "n_paragraphs": len(bench["paragraphs"]),
           "n_nodes": len(ws.frame.nodes), "n_gold": n_gold,
           "embedder": ML_MODEL, "summary": {}}
    rows_by_sys = {}
    for name, fn in systems.items():
        t0 = time.time()
        res = run_system(name, fn, emb, records, k=8)
        rows_by_sys[name] = res["rows"]
        agg = res["aggregate"]
        out["summary"][name] = {k: agg.get(k, 0)
                                for k in ("mrr", "retrieval_recall", "latency_ms")}
        print(f"  {name:12s} mrr={agg.get('mrr', 0):.4f} "
              f"recall@8={agg.get('retrieval_recall', 0):.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    for metric in ("mrr", "retrieval_recall"):
        bt = paired_bootstrap(rows_by_sys["mira_full"], rows_by_sys["flat_vector"], metric)
        out[f"sig_mira_vs_flat_{metric}"] = {"bootstrap": bt}
        print(f"  mira vs flat [{metric}]: d={bt['mean_diff']} p~{bt['p_value']}",
              flush=True)
    return out


def main() -> int:
    allow_dl = os.environ.get("MIRA_ALLOW_DOWNLOAD", "1") == "1"
    emb = EmbeddingBackend(ML_MODEL, device="cpu", allow_download=allow_dl)
    emb.encode(["warmup"])
    if not emb.is_real_model:
        print(f"FATAL: multilingual model unavailable: {emb.last_error}")
        return 2

    langs = {}
    for lang in ("hi", "mr"):
        src = build_bench(lang)
        ingest(lang, src, emb)
        langs[lang] = src

    results = {}
    for lang, src in langs.items():
        results[lang] = evaluate(lang, src, emb)

    prev_small = os.path.join(ROOT, "data_indic_ml", "indicqa_ml_results.json")
    prev = json.load(open(prev_small, encoding="utf-8")) if os.path.exists(prev_small) else {}
    out = {"embedder": ML_MODEL, "scale": "all questions, full paragraph corpus",
           "languages": results,
           "small_120q_run": prev.get("languages", {})}
    with open(os.path.join(OUT_DIR, "indicqa_full_results.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("wrote data_indic_ml/indicqa_full_results.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
