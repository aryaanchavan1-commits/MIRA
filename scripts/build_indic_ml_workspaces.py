"""Ingest IndicQA hi/mr into data_indic_ml/ with the MULTILINGUAL embedder.

Companion to eval_indic_multilingual.py (same model name constant). BM25 is
embedded-model-agnostic, so it is skipped here and copied from the English
run's artifact at analysis time.
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
ML_DATA = os.path.join(ROOT, "data_indic_ml")
BENCH_DIR = os.path.join(ROOT, "data_indic", "benchmarks")

from config.auto_config import build_context  # noqa: E402
from core.workspace import Workspace  # noqa: E402
from evaluation.datasets import resolve_titles_to_ids  # noqa: E402


def build_lang(lang: str, emb) -> int:
    data_dir = os.path.join(ML_DATA, lang)
    cw.DATA_DIR = data_dir
    ac.DATA_DIR = data_dir
    src = os.path.join(BENCH_DIR, f"indicqa_{lang}_bench.json")
    with open(src, "r", encoding="utf-8") as fh:
        bench = json.load(fh)
    paras = bench["paragraphs"]
    print(f"[{lang}] ingesting {len(paras)} paragraphs (multilingual embedder)",
          flush=True)

    os.makedirs(data_dir, exist_ok=True)
    ws = Workspace(embeddings=emb, llm=None, config={})
    from ingestion.pipeline import IngestionPipeline  # noqa: E402
    pipe = IngestionPipeline(ws.store, emb, llm=None,
                             config=ws._pipe_config(), bulk_mode=True)
    t0 = time.time()
    done = failed = 0
    for title, text in paras.items():
        try:
            pipe.ingest_text(text, title=f"IndicQA-{lang}: {title}",
                             source_path=f"(indicqa.{lang})")
            done += 1
        except Exception as exc:
            failed += 1
            print(f"  FAIL {title[:30]!r}: {exc}", flush=True)
    flushed = pipe.flush_bulk_vectors()
    ws.reload()
    gold = sum(1 for q in bench["questions"]
               if resolve_titles_to_ids(ws, q["supporting_titles"]))
    print(f"[{lang}] done: {done} docs ({failed} failed), {len(ws.frame.nodes)} nodes, "
          f"{flushed} vectors, gold {gold}/{len(bench['questions'])} "
          f"in {time.time() - t0:.0f}s", flush=True)
    return 0 if failed == 0 else 1


def main() -> int:
    allow_dl = os.environ.get("MIRA_ALLOW_DOWNLOAD", "1") == "1"
    from models.embeddings import EmbeddingBackend  # noqa: E402
    emb = EmbeddingBackend(ML_MODEL, device="cpu", allow_download=allow_dl)
    emb.encode(["warmup"])
    if not emb.is_real_model:
        print(f"FATAL: multilingual model unavailable: {emb.last_error}")
        return 2
    # The ML backend must be used for BOTH the workspace and the pipeline.
    # Passing build_context()'s default English embedder here is exactly the
    # mismatch the lineage guard warns about (it silently re-embeds English).
    rc = 0
    for lang in ("hi", "mr"):
        rc |= build_lang(lang, emb)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
