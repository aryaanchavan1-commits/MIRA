"""Catastrophic Forgetting Lab + BioMIRA ablation (spec §15-17, §26-29).

Replays the sequential learning of Task A → B → C → D and, after every task,
re-tests **all** tasks learned so far:

    after A:  A          after B:  A + B
    after C:  A + B + C  after D:  A + B + C + D

then reports Forgetting_i / Retention_i (evaluation/forgetting.py).

Why this is affordable: ingestion is variant-independent, so the expensive
embedding work is paid once by scripts/build_forgetting_lab.py. Each variant
replays the sequence over a *masked frame* — at step k only nodes from tasks
0..k are visible — so learning order, interference and the dynamics passes all
happen for real, without re-ingesting or re-embedding per variant.

Retrieval-level, not answer-level: no LLM is loaded, so the metrics are MRR /
recall / hit@k over evidence node ids. That is the honest scope for a frozen
1-3B model on a laptop (§20-21).

Usage:
    .venv/Scripts/python.exe scripts/eval_forgetting.py --per-task 40
    .venv/Scripts/python.exe scripts/eval_forgetting.py --variants B_mira,J_full_biomira
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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LAB = os.path.join(_ROOT, "data_lab")
cw.DATA_DIR = _LAB
import config.auto_config as ac  # noqa: E402
ac.DATA_DIR = _LAB

from core import biomira  # noqa: E402
from core.memory import MemoryFrame  # noqa: E402
from core.retrieval import MIRARetriever  # noqa: E402
from core.workspace import Workspace  # noqa: E402
from evaluation.forgetting import forgetting_metrics, summary_row  # noqa: E402
from evaluation.metrics import mrr  # noqa: E402
from scripts.consolidate import build_embeddings  # noqa: E402

MANIFEST = os.path.join(_LAB, "lab_manifest.json")
OUT = os.path.join(_LAB, "forgetting_results.json")

FULL = ("semantic", "structural", "radial", "graph", "importance",
        "confidence", "recency", "path", "activation")

# --- the ablation ladder (spec §17) ---------------------------------------
# Interpretation, stated up front so the numbers cannot be misread:
#   A  = vector RAG floor (semantic only)
#   C/D/E = MIRA with exactly ONE structural signal, to see what each buys
#   B  = full MIRA baseline (all signals, no BioMIRA)
#   F-J = full MIRA plus ONE adaptive mechanism at a time (single-factor),
#         and J = every mechanism together. Improvement is only ever
#         attributed to a component that shows its own isolated effect.
VARIANTS = {
    "A_vector_rag":      {"components": ("semantic",)},
    "B_mira":            {"components": None},
    "C_mira_radial":     {"components": ("semantic", "radial", "structural")},
    "D_mira_graph":      {"components": ("semantic", "graph", "path")},
    "E_mira_hierarchy":  {"components": ("semantic", "structural")},
    "F_mira_decay":      {"components": None, "decay": True},
    "G_mira_consolidation": {"components": None, "consolidation": True},
    "H_mira_replay":     {"components": None, "replay": True},
    "I_bio_dynamics":    {"components": None, "decay": True, "replay": True,
                          "consolidation": True, "kappa": 0.10},
    "J_full_biomira":    {"components": None, "decay": True, "replay": True,
                          "consolidation": True, "kappa": 0.10,
                          "homeostasis": True, "ring_migration": True,
                          "merge_report": True},
}


def variant_config(base: dict, spec: dict) -> dict:
    """Config for one variant: BioMIRA on only if it uses a mechanism."""
    uses_bio = any(spec.get(k) for k in
                   ("decay", "replay", "consolidation", "homeostasis",
                    "ring_migration", "merge_report")) or spec.get("kappa")
    cfg = json.loads(json.dumps(base))          # deep copy
    cfg["biomira"] = dict(base.get("biomira") or {})
    cfg["biomira"].update({
        "enabled": bool(uses_bio),
        "homeostasis": bool(spec.get("homeostasis", True)),
        "ring_migration": bool(spec.get("ring_migration", False)),
    })
    cfg.setdefault("retrieval_score", {})["kappa_stability"] = float(spec.get("kappa", 0.0))
    return cfg


def load_workspace() -> Workspace:
    # embeddings only: build_context() also warms llama.cpp, and a 4 GB VRAM
    # box aborts GGML on the commit limit even though this lab never answers.
    embeddings, cfg = build_embeddings()
    ws = Workspace(embeddings=embeddings, llm=None, config=cfg)
    if not ws.frame.nodes:
        raise SystemExit("lab workspace is empty — run "
                         "scripts/build_forgetting_lab.py first")
    return ws


def task_index(ws) -> dict:
    """node_id -> task index (from the builder's metadata tag)."""
    out = {}
    for nid, node in ws.frame.nodes.items():
        try:
            out[nid] = int(node.metadata.get("lab_task"))
        except (TypeError, ValueError):
            continue
    return out


def masked_frame(ws, idx: dict, upto: int) -> MemoryFrame:
    visible = {nid: t for nid, t in idx.items() if t <= upto}
    f = MemoryFrame()
    for nid, node in ws.frame.nodes.items():
        if nid in visible:
            f.add_node(node)
    for e in ws.frame.edges:
        if e.source_id in visible and e.target_id in visible:
            f.add_edge(e)
    return f


def gold_ids(ws, questions: list, per_task: int, seed: int) -> dict:
    """task -> [{question, supporting_ids}] using the existing title resolver."""
    from evaluation.datasets import resolve_titles_to_ids
    rng = random.Random(seed)
    by_task: dict = {}
    for q in questions:
        ids = resolve_titles_to_ids(ws, q["supporting_titles"])
        if not ids:
            continue
        by_task.setdefault(q["task"], []).append(
            {"question": q["question"], "answer": q["answer"],
             "supporting_ids": ids})
    for task, rows in by_task.items():
        if 0 < per_task < len(rows):
            by_task[task] = rng.sample(rows, per_task)
    return dict(sorted(by_task.items()))


def run_variant(name: str, spec: dict, ws: Workspace, base_cfg: dict, idx: dict,
                golds: dict, n_tasks: int, k: int) -> dict:
    """Replay the sequence for one variant; return lab rows."""
    cfg = variant_config(base_cfg, spec)
    bio = biomira.bio_config(cfg)
    rows: list = []
    dyn: dict = {"decays": 0, "replays": 0, "promotions": 0}
    snapshot_imp = {n.id: n.importance for n in ws.frame.nodes.values()}
    snapshot_ring = {n.id: n.ring for n in ws.frame.nodes.values()}

    for step in range(n_tasks):
        # --- the variant's "sleep" pass over everything learned so far -----
        frame = masked_frame(ws, idx, step)
        if bio["enabled"]:
            if spec.get("decay"):
                dyn["decays"] += 1
                biomira.apply_adaptive_decay(frame, bio)
            if spec.get("consolidation"):
                hist: dict = {}
                for node in frame.nodes.values():
                    s = biomira.state(node)["consolidation_state"]
                    hist[s] = hist.get(s, 0) + 1
                dyn["states"] = hist
            if spec.get("replay"):
                dyn["replays"] += biomira.replay(frame, bio).get("replayed", 0)
            if spec.get("homeostasis"):
                biomira.homeostatize(frame, bio)
            if spec.get("ring_migration"):
                dyn["promotions"] += biomira.migrate_rings(frame, bio).get("promoted", 0)

        from storage.graph_store import GraphStore
        gs = GraphStore()
        gs.build_from([n.to_row() for n in frame.nodes.values()],
                      [e.to_row() for e in frame.edges])
        active = spec["components"]
        # retrieval itself counts as a use: that is the feedback loop (§24)
        retriever = MIRARetriever(frame, ws.vs, gs, cfg)

        for task in range(step + 1):
            recs = golds.get(task, [])
            if not recs:
                continue
            hits, rr, lat = [], [], []
            for rec in recs:
                qv = ws.embeddings.encode([rec["question"]])[0]
                res = retriever.retrieve(rec["question"], qv, active_components=active,
                                         final_k=k)
                gold = set(rec["supporting_ids"])
                ranked = [i.node.id for i in res.items]
                hits.append(len(gold & set(ranked[:k])) / max(1, len(gold)))
                rr.append(mrr(ranked, gold))
                lat.append(res.latency_ms)
                if bio["enabled"]:
                    # the frozen LLM accepts every grounded retrieval, so a hit
                    # counts as successful use; that is the feedback edge of the
                    # architecture (§24 ANSWER -> FEEDBACK -> consolidation)
                    biomira.touch(frame, ranked, bio,
                                  failed=not (gold & set(ranked[:k])))
            rows.append({"step": step, "task": task, "n": len(recs),
                         "mrr": round(sum(rr) / len(rr), 6),
                         "recall": round(sum(hits) / len(hits), 6),
                         "hit_at_k": round(sum(1 for h in hits if h > 0)
                                           / len(hits), 6),
                         "latency_ms": round(sum(lat) / len(lat), 2)})
            # per-question reciprocal ranks only for the final step: that is
            # the headline comparison, and the only place a paired test has
            # the resolution to say anything. Curves stay as means, which is
            # all a forgetting curve needs.
            if step == n_tasks - 1:
                rows[-1]["questions"] = [q["question"] for q in recs]
                rows[-1]["rr"] = [round(x, 5) for x in rr]
                rows[-1]["hit"] = [1 if h > 0 else 0 for h in hits]
        if not bio["enabled"]:
            # keep baseline runs free of any BioMIRA state leaking forward
            for node in frame.nodes.values():
                node.metadata.pop("bio", None)

    for nid, node in ws.frame.nodes.items():
        node.importance = snapshot_imp.get(nid, node.importance)
        node.ring = snapshot_ring.get(nid, node.ring)
        node.metadata.pop("bio", None)
    out = {"variant": name, "spec": spec, "rows": rows,
           "metrics": forgetting_metrics(rows),
           "step_summary": [summary_row(rows, s) for s in range(n_tasks)],
           "dynamics": dyn}
    return out


def _final_question_rows(variant: dict, n_tasks: int) -> list:
    """Flatten the final step into per-question rows for a paired test."""
    out = []
    for r in variant["rows"]:
        if r["step"] != n_tasks - 1 or "rr" not in r:
            continue
        for q, rr, hit in zip(r["questions"], r["rr"], r["hit"]):
            out.append({"question": q, "rr": rr, "hit": hit})
    return out


def significance(results: dict, baseline: str, n_tasks: int) -> dict:
    """Paired bootstrap of every variant against the MIRA baseline.

    Mean differences on aggregate curves can hide per-question reversals, so
    the test is run on the paired per-question reciprocal ranks of the final
    step — the same questions, the same corpus, one mechanism changed.
    """
    from scripts.eval_significance import paired_bootstrap
    base = _final_question_rows(results[baseline], n_tasks)
    out = {"baseline": baseline, "test": "paired bootstrap, 10k resamples, "
                                        "final-step per-question reciprocal rank",
           "vs_baseline": {}}
    for name, v in results.items():
        if name == baseline:
            continue
        out["vs_baseline"][name] = paired_bootstrap(
            _final_question_rows(v, n_tasks), base, "rr")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="all",
                    help="comma-separated variant names, or 'all'")
    ap.add_argument("--per-task", type=int, default=40)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    names = (list(VARIANTS) if args.variants == "all"
             else [v.strip() for v in args.variants.split(",") if v.strip()])
    unknown = set(names) - set(VARIANTS)
    if unknown:
        raise SystemExit(f"unknown variants: {sorted(unknown)}")

    with open(MANIFEST, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    n_tasks = len(manifest["plan"]["tasks"])

    ws = load_workspace()
    base_cfg = ws.config
    idx = task_index(ws)
    tagged = len(idx)
    print(f"lab workspace: {len(ws.frame.nodes)} nodes, {tagged} tagged, "
          f"{n_tasks} tasks, {ws.vs.size() if ws.vs else 0} vectors", flush=True)
    if tagged < len(ws.frame.nodes):
        print(f"warning: {len(ws.frame.nodes) - tagged} untagged nodes are "
              f"treated as task 0", flush=True)
    golds = gold_ids(ws, manifest["questions"], args.per_task, args.seed)
    print("questions per task: " + str({t: len(v) for t, v in golds.items()}),
          flush=True)

    results = {}
    t0 = time.time()
    for name in names:
        v = run_variant(name, VARIANTS[name], ws, base_cfg, idx, golds,
                        n_tasks, args.k)
        results[name] = v
        agg = v["step_summary"][-1] if v["step_summary"] else {}
        print(f"{name:24s} final mrr {agg.get('mrr', 0):.4f} "
              f"recall {agg.get('recall', 0):.4f} "
              f"avg_forgetting {v['metrics']['average_forgetting'].get('mrr', 0):+.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    payload = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "n_tasks": n_tasks, "k": args.k, "per_task": args.per_task,
               "questions_per_task": {str(t): len(v) for t, v in golds.items()},
               "significance": significance(results, "B_mira", n_tasks),
               "scope": "retrieval-level (MRR/recall) over evidence node ids; "
                        "LLM frozen and unused, so this measures memory "
                        "interference, not parametric forgetting",
               "variant_interpretation":
                   "C/D/E isolate one structural signal each; F-I add one "
                   "adaptive mechanism at a time to full MIRA; J adds all",
               "variants": results}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
