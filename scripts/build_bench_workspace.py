"""Ingest the MuSiQue benchmark corpus into an ISOLATED workspace.

Patches core.workspace.DATA_DIR to data_bench/ so the main corpus and the
benchmark corpus never mix. Embeddings-only (no LLM): deterministic
extraction, and GPU/RAM stays free for the answer-eval run later.

Usage: .venv/Scripts/python.exe scripts/build_bench_workspace.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.workspace as cw  # noqa: E402

# Patch BOTH module globals: core.workspace re-derives its paths from its own
# DATA_DIR at call time, but ingestion/pipeline.py reads config.auto_config's
# copy directly. Patching only one is how bench vectors once leaked into the
# main FAISS index (both files had identical mtimes).
_BENCH_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data_bench")
cw.DATA_DIR = _BENCH_DATA
import config.auto_config as ac  # noqa: E402
ac.DATA_DIR = _BENCH_DATA

from config.auto_config import build_context  # noqa: E402
from core.workspace import Workspace  # noqa: E402

BENCH = os.path.join(_BENCH_DATA, "benchmarks")
SRC = os.path.join(BENCH, "musique_bench.json")


def main() -> int:
    os.makedirs(BENCH, exist_ok=True)
    with open(SRC, "r", encoding="utf-8") as fh:
        bench = json.load(fh)
    paras = bench["paragraphs"]
    print(f"ingesting {len(paras)} paragraphs into isolated db at {_BENCH_DATA}",
          flush=True)

    ctx = build_context()
    ws = Workspace(embeddings=ctx.embeddings, llm=None, config=ctx.cfg)

    # Bulk path: defer per-document vector-index writes and the heavy
    # per-document reload; flush vectors once at the end, then reload once.
    from ingestion.pipeline import IngestionPipeline  # noqa: E402
    pipe = IngestionPipeline(ws.store, ws.embeddings, llm=None,
                             config=ws._pipe_config(), bulk_mode=True)
    t0 = time.time()
    done = failed = 0
    for title, text in paras.items():
        try:
            pipe.ingest_text(text, title=f"MuSiQue: {title}", source_path="(musique)")
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(paras)} docs, {time.time() - t0:.0f}s, "
                      f"{len(pipe.bulk_frame.nodes)} buffered nodes", flush=True)
        except Exception as exc:
            failed += 1
            print(f"  FAIL {title!r}: {exc}", flush=True)
    flushed = pipe.flush_bulk_vectors()
    print(f"flushed {flushed} vectors once (bulk mode)", flush=True)
    ws.reload()
    print(f"done: {done} docs ({failed} failed), {len(ws.frame.nodes)} nodes, "
          f"{ws.vs.size() if ws.vs else 0} vectors in {time.time() - t0:.0f}s",
          flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
