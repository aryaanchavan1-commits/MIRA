"""Ingest the HotpotQA benchmark corpus into its own ISOLATED workspace
(data_hotpot/) — same bulk pipeline as build_bench_workspace.py.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.workspace as cw  # noqa: E402

# Patch BOTH module globals (see build_bench_workspace.py: leaking vectors
# across workspaces is the known failure mode).
_HOT_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data_hotpot")
cw.DATA_DIR = _HOT_DATA
import config.auto_config as ac  # noqa: E402
ac.DATA_DIR = _HOT_DATA

from config.auto_config import build_context  # noqa: E402
from core.workspace import Workspace  # noqa: E402

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data_bench", "benchmarks", "hotpotqa_bench.json")


def main() -> int:
    os.makedirs(_HOT_DATA, exist_ok=True)
    with open(SRC, "r", encoding="utf-8") as fh:
        bench = json.load(fh)
    paras = bench["paragraphs"]
    print(f"ingesting {len(paras)} paragraphs into isolated db at {_HOT_DATA}",
          flush=True)

    ctx = build_context()
    ws = Workspace(embeddings=ctx.embeddings, llm=None, config=ctx.cfg)

    from ingestion.pipeline import IngestionPipeline  # noqa: E402
    pipe = IngestionPipeline(ws.store, ws.embeddings, llm=None,
                             config=ws._pipe_config(), bulk_mode=True)
    t0 = time.time()
    done = failed = 0
    for title, text in paras.items():
        try:
            pipe.ingest_text(text, title=f"HotpotQA: {title}",
                             source_path="(hotpotqa)")
            done += 1
            if done % 100 == 0:
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
