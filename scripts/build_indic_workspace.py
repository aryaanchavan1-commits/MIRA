"""Ingest the IndicQA corpora into isolated per-language workspaces.

data_indic/hi/ and data_indic/mr/ — small corpora (240 paragraphs each), so
this runs in a couple of minutes per language.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import core.workspace as cw  # noqa: E402
import config.auto_config as ac  # noqa: E402

from config.auto_config import build_context  # noqa: E402
from core.workspace import Workspace  # noqa: E402
from evaluation.datasets import resolve_titles_to_ids  # noqa: E402


def build_lang(lang: str) -> int:
    data_dir = os.path.join(ROOT, "data_indic", lang)
    cw.DATA_DIR = data_dir
    ac.DATA_DIR = data_dir
    src = os.path.join(ROOT, "data_indic", "benchmarks", f"indicqa_{lang}_bench.json")
    with open(src, "r", encoding="utf-8") as fh:
        bench = json.load(fh)
    paras = bench["paragraphs"]
    print(f"[{lang}] ingesting {len(paras)} paragraphs into {data_dir}", flush=True)

    ctx = build_context()
    ws = Workspace(embeddings=ctx.embeddings, llm=None, config=ctx.cfg)
    from ingestion.pipeline import IngestionPipeline  # noqa: E402
    pipe = IngestionPipeline(ws.store, ws.embeddings, llm=None,
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
    print(f"[{lang}] done: {done} docs ({failed} failed), "
          f"{len(ws.frame.nodes)} nodes, {flushed} vectors, "
          f"gold resolved {gold}/{len(bench['questions'])} "
          f"in {time.time() - t0:.0f}s", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    rc = 0
    for lang in ("hi", "mr"):
        rc |= build_lang(lang)
    raise SystemExit(rc)
