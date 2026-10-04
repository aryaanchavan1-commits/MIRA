"""Build the Catastrophic Forgetting Lab workspace: sequential task shards (§15).

Real MuSiQue data, ingested as **six sequential task shards** so knowledge
acquisition order is controlled and reproducible:

    Task A -> Task B -> Task C -> Task D -> Task E -> Task F
      test      test        test        test        test        test
        A      A + B       A + B + C   A + ... + D  ... + E     A + ... + F

Six rather than four by default: a longer sequence gives the forgetting curve
more post-arrival observation points (a task is re-tested at every later step),
which is the axis the whole study lives on. ``--tasks 4`` reproduces the
shorter sequence.

Design notes (why it is built this way):

* Tasks are disjoint by construction. A question joins task *k* only if **all**
  of its supporting paragraphs are new in shard *k*; paragraphs shared across
  tasks are ingested with the earliest task and their questions are dropped
  from later tasks, so no task is ever handed pre-learned gold evidence.
* Each shard also gets distractors sampled deterministically from the rest of
  the corpus, so retrieval faces interference instead of a clean room.
* Every node is tagged ``metadata.lab_task`` so the lab can replay the
  sequence (or mask later shards) without re-ingesting or re-embedding.
  Ingestion is variant-independent, which is what makes a 10-variant ablation
  affordable: the expensive embedding work is paid exactly once.

Usage:
    .venv/Scripts/python.exe scripts/build_forgetting_lab.py
    .venv/Scripts/python.exe scripts/build_forgetting_lab.py --tasks 4
    .venv/Scripts/python.exe scripts/build_forgetting_lab.py --distractors 150
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.workspace as cw  # noqa: E402

# Patch BOTH module globals (see build_bench_workspace.py for why).
_LAB_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data_lab")
cw.DATA_DIR = _LAB_DATA
import config.auto_config as ac  # noqa: E402
ac.DATA_DIR = _LAB_DATA

from config.auto_config import build_context  # noqa: E402
from core.workspace import Workspace  # noqa: E402

BENCH = os.path.join(os.path.dirname(cw.DATA_DIR.rstrip("\\/")), "data_bench",
                     "benchmarks", "musique_bench.json")
MANIFEST = os.path.join(_LAB_DATA, "lab_manifest.json")


def plan_tasks(questions: list, paragraphs: dict, n_tasks: int, seed: int,
               distractors: int) -> dict:
    """Deterministic, disjoint task plan (no LLM, no embedding needed)."""
    rng = random.Random(seed)
    ordered = sorted(questions, key=lambda q: q["id"])
    rng.shuffle(ordered)
    per = len(ordered) // n_tasks
    groups = [ordered[i * per:(i + 1) * per] for i in range(n_tasks)]

    # earliest task that needs a paragraph owns it
    owner: dict = {}
    for i, group in enumerate(groups):
        for q in group:
            for t in q.get("supporting_titles") or []:
                owner.setdefault(t, i)

    tasks = []
    used: set = set()
    dropped = 0
    for i, group in enumerate(groups):
        titles = [t for t in paragraphs if owner.get(t) == i]
        kept = []
        for q in group:
            support = [t for t in (q.get("supporting_titles") or []) if t in paragraphs]
            if not support:
                dropped += 1
                continue
            kept.append(q)
        tasks.append({"index": i, "name": chr(ord("A") + i), "titles": titles,
                      "questions": kept})
        used.update(titles)

    # A question is TESTED at the step where its last gold paragraph has
    # arrived: earlier if some evidence is already in memory (which is exactly
    # the continual-learning situation), never later than its own task.
    buckets: dict = {i: [] for i in range(n_tasks)}
    for i, group in enumerate(groups):
        for q in group:
            support = [t for t in (q.get("supporting_titles") or []) if t in paragraphs]
            if not support:
                continue
            step = max(min(owner.get(t, i), i) for t in support)
            buckets[step].append(q)
    for i, task in enumerate(tasks):
        task["questions"] = buckets[i]
        task["own_questions"] = len(groups[i])

    # distractors: deterministic sample of unused corpus paragraphs per task
    pool = sorted(t for t in paragraphs if t not in used)
    rng.shuffle(pool)
    for t_i, task in enumerate(tasks):
        task["distractors"] = pool[t_i * distractors:(t_i + 1) * distractors]
    return {"tasks": tasks, "dropped_questions": dropped,
            "corpus_size": len(paragraphs), "seed": seed, "n_tasks": n_tasks}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=6)
    ap.add_argument("--distractors", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plan-only", action="store_true",
                    help="write the task plan and exit (no ingestion)")
    args = ap.parse_args()

    with open(BENCH, "r", encoding="utf-8") as fh:
        bench = json.load(fh)
    paragraphs, questions = bench["paragraphs"], bench["questions"]
    plan = plan_tasks(questions, paragraphs, args.tasks, args.seed, args.distractors)
    os.makedirs(_LAB_DATA, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump({"plan": plan, "questions": [
            {"task": t["index"], "id": q["id"], "question": q["question"],
             "answer": q["answer"], "supporting_titles": q["supporting_titles"]}
            for t in plan["tasks"] for q in t["questions"]]}, fh, indent=1)
    total_docs = sum(len(t["titles"]) + len(t["distractors"]) for t in plan["tasks"])
    print(f"plan: {len(plan['tasks'])} tasks, "
          f"{sum(len(t['questions']) for t in plan['tasks'])} questions "
          f"({plan['dropped_questions']} dropped as pre-learned), "
          f"{total_docs} documents -> {MANIFEST}", flush=True)
    if args.plan_only:
        return 0

    ctx = build_context()
    ws = Workspace(embeddings=ctx.embeddings, llm=None, config=ctx.cfg)
    from ingestion.pipeline import IngestionPipeline  # noqa: E402
    pipe = IngestionPipeline(ws.store, ws.embeddings, llm=None,
                             config=ws._pipe_config(), bulk_mode=True)
    t0 = time.time()
    for task in plan["tasks"]:
        docs = [(t, paragraphs[t]) for t in task["titles"] + task["distractors"]]
        before = set(pipe.bulk_frame.nodes)
        for title, text in docs:
            pipe.ingest_text(text, title=f"MuSiQue: {title}", source_path="(musique)")
        new_ids = [nid for nid in pipe.bulk_frame.nodes if nid not in before]
        with ws.store.tx() as c:
            c.executemany(
                "UPDATE nodes SET metadata = json_set("
                "COALESCE(NULLIF(metadata,''),'{}'),'$.lab_task', ?) WHERE id = ?",
                [(int(task["index"]), nid) for nid in new_ids])
        print(f"  task {task['name']}: {len(docs)} docs, {len(new_ids)} nodes, "
              f"{time.time() - t0:.0f}s", flush=True)
    flushed = pipe.flush_bulk_vectors()
    print(f"flushed {flushed} vectors", flush=True)
    ws.reload()
    with ws.store.tx() as c:
        tagged = c.execute(
            "SELECT lab_task, COUNT(*) FROM (SELECT json_extract(metadata,'$.lab_task') "
            "AS lab_task FROM nodes) WHERE lab_task IS NOT NULL GROUP BY lab_task"
        ).fetchall()
    print(f"done in {time.time() - t0:.0f}s: {len(ws.frame.nodes)} nodes, "
          f"{ws.vs.size() if ws.vs else 0} vectors, per-task nodes "
          f"{ {r[0]: r[1] for r in tagged} }", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
