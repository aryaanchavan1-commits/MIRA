"""Sleep consolidation for the mandala — offline memory-dynamics pass.

Applies the three mechanisms from core/memory_dynamics.py to a workspace,
in the order sleep would:

  1. REPLAY  — recent retrieval_logs and remembered-exchange docs are
     re-fired through hebbian.reinforce (edges along used paths strengthen,
     untouched edges decay slightly) and reinforce_nodes (importance lift +
     decay-clock reset for used nodes).
  2. DECAY   — Ebbinghaus-style soft forgetting over all nodes (never
     deletes; retention scales with connectivity and confidence).
  3. GISTS   — cluster fact/document nodes per sector, store summary
     "gist" nodes wired with gist_of edges (gist abstraction above detail).
  4. PLACE   — re-run mandala placement + persist (workspace.replace_all),
     which also rebuilds the FAISS index.

Retrieval is measured BEFORE and AFTER on the bench's own questions (same
harness as the paper, gold ids resolved the same way), so the pass earns
its keep measurably or the report says so. Default is a DRY probe; pass
--apply to write. A db backup is taken before any write.

Usage:
  .venv/Scripts/python.exe scripts/consolidate.py                 # probe
  .venv/Scripts/python.exe scripts/consolidate.py --apply --half-life 21
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timedelta

if hasattr(sys.stdout, "reconfigure"):  # cp1252 consoles choke on fancy glyphs
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import core.workspace as cw  # noqa: E402
import config.auto_config as ac  # noqa: E402

from config.auto_config import PROJECT_ROOT, ensure_dirs, load_config  # noqa: E402
from core.hardware import auto_configure, detect_hardware  # noqa: E402
from core.hebbian import reinforce  # noqa: E402
from core.memory_dynamics import apply_decay, build_gists, reinforce_nodes  # noqa: E402
from evaluation.benchmark import mira_retrieve_fn, run_system  # noqa: E402
from evaluation.datasets import resolve_titles_to_ids  # noqa: E402
from scripts.eval_significance import paired_bootstrap  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data_bench")
BENCH = os.path.join(DATA_DIR, "benchmarks", "musique_bench.json")
OUT = os.path.join(DATA_DIR, "consolidation_results.json")


def build_embeddings():
    """Embedding backend only — never the LLM. build_context() also warms the
    llama.cpp model, and under Windows' commit limit that GGML abort kills the
    whole process even though retrieval-only passes don't need an LLM."""
    cfg = load_config(None)
    ensure_dirs()
    hw = detect_hardware(PROJECT_ROOT)
    rc = auto_configure(hw, cfg)
    from models.embeddings import EmbeddingBackend
    allow_dl = (not bool(cfg.get("offline", True))) and \
        bool(cfg.get("models", {}).get("allow_download", False))
    emb = EmbeddingBackend(rc.embedding_model, device=rc.embedding_device,
                           allow_download=allow_dl)
    emb.encode(["warmup"])
    return emb, cfg


def load_bench_records(ws, n: int):
    with open(BENCH, "r", encoding="utf-8") as fh:
        bench = json.load(fh)
    questions = bench["questions"][:n]
    id_by_title = {}
    records = []
    missing = 0
    for q in questions:
        gold = resolve_titles_to_ids(ws, q["supporting_titles"])
        if not gold:
            missing += 1
            continue
        records.append({"id": q["id"], "question": q["question"],
                        "answer": q["answer"], "supporting_ids": gold})
    return records, missing


def stability_records(ws, n: int) -> list:
    """Main-workspace probe: recent chat queries with gold = the top-5 ids
    retrieval returned BEFORE consolidation. Before/after MRR on these is a
    *stability* metric — how much sleep perturbs existing recall."""
    rows = []
    try:
        with ws.store.tx() as c:
            c.execute("SELECT query_hash, meta FROM retrieval_logs "
                      "ORDER BY at DESC LIMIT 400")
            rows = c.fetchall()
    except Exception:
        return []
    seen, records = set(), []
    for qhash, meta_json in rows:
        try:
            meta = json.loads(meta_json or "{}")
        except ValueError:
            continue
        ids = (meta.get("node_ids") or [])[:5]
        if len(ids) < 3 or qhash in seen:
            continue
        seen.add(qhash)
        records.append({"id": qhash, "question": meta.get("query") or qhash,
                        "answer": "", "supporting_ids": ids})
        if len(records) >= n:
            break
    return records


def eval_retrieval(ws, records, k: int = 8) -> dict:
    fn = mira_retrieve_fn(ws, k=k)
    out = run_system("mira_full", fn, ws.embeddings, records, k=k)
    return out


def replay_paths(ws, days: int, apply: bool = False) -> dict:
    """Re-fire recent retrievals: log-derived seed pairs + remembered docs."""
    since = (datetime.utcnow() - timedelta(days=days)).isoformat(timespec="seconds")
    rows = []
    try:
        with ws.store.tx() as c:
            c.execute("SELECT at, query_hash, meta FROM retrieval_logs "
                      "WHERE at >= ? ORDER BY at DESC LIMIT 5000", (since,))
            rows = c.fetchall()
    except Exception:
        rows = []
    seed_ids: list[str] = []
    from_logs = 0
    for at, qhash, meta_json in rows:
        try:
            meta = json.loads(meta_json or "{}")
        except ValueError:
            continue
        ids = (meta.get("node_ids") or [])[:6]
        if ids:
            seed_ids.extend(ids)
            from_logs += 1
    # remembered exchanges carry their own provenance; re-fire their docs
    remembered = [d["id"] for d in ws.store.list_documents()
                  if str(d.get("source_path", "")).startswith("conversation:")]
    doc_nodes = []
    for doc_id in remembered[:50]:
        doc_nodes.extend(n.id for n in ws.frame.nodes.values()
                         if doc_id in (n.source_ids or []))
    seed_ids.extend(doc_nodes[:200])

    seeds = [nid for nid, _ in Counter(seed_ids).most_common(400)
             if nid in ws.frame.nodes]
    if not seeds:
        return {"log_entries": len(rows), "queries_with_ids": from_logs,
                "remembered_docs": len(remembered), "seed_nodes": 0,
                "paths_fired": 0}
    # pair consecutive candidates of the same query as co-activation paths
    paths, cur = [], []
    for nid in seed_ids:
        if nid in ws.frame.nodes:
            cur.append(nid)
            if len(cur) == 2:
                paths.append(cur)
                cur = []
    changed = reinforce(ws.frame, paths, lr=0.10, decay=0.02,
                        store=ws.store if apply else None)
    node_report = reinforce_nodes(ws.frame, seeds, lift=0.06,
                                  persist=ws.store.upsert_node if apply else None)
    return {"log_entries": len(rows), "queries_with_ids": from_logs,
            "remembered_docs": len(remembered), "seed_nodes": len(seeds),
            "paths_fired": len(paths), "edges_changed": len(changed),
            **node_report}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry probe)")
    ap.add_argument("--target", choices=["bench", "main"], default="bench",
                    help="bench = isolated MuSiQue ws; main = live corpus "
                         "(has real chat logs + remembered exchanges for replay)")
    ap.add_argument("--half-life", type=float, default=21.0)
    ap.add_argument("--replay-days", type=int, default=30)
    ap.add_argument("--n", type=int, default=300, help="probe questions")
    ap.add_argument("--max-gists", type=int, default=24)
    args = ap.parse_args()

    global DATA_DIR, BENCH, OUT
    if args.target == "main":
        DATA_DIR = os.path.join(ROOT, "data")
        BENCH = None  # main ws probes on its own recent chat questions instead
        OUT = os.path.join(DATA_DIR, "consolidation_results.json")
    cw.DATA_DIR = DATA_DIR
    ac.DATA_DIR = DATA_DIR
    embeddings, cfg = build_embeddings()
    ws = cw.Workspace(embeddings=embeddings, llm=None, config=cfg)
    n_nodes, n_edges = len(ws.frame.nodes), len(ws.frame.edges)
    print(f"workspace: {n_nodes} nodes, {n_edges} edges, apply={args.apply}")

    # known-failure guard: placement left on a non-hybrid strategy collapses
    # ring-0/1 (3,900 -> 110 on the bench ws), which silently empties the
    # hierarchical candidate stage and halves MRR
    ring01 = sum(1 for n in ws.frame.nodes.values() if n.ring in (0, 1))
    rate = ring01 / max(n_nodes, 1)
    print(f"ring-0/1 rate: {rate:.3%} ({ring01}/{n_nodes})")
    if n_nodes > 2000 and rate < 0.02:
        print("FATAL: ring-0/1 rate < 2% — hierarchical candidate stage would be "
              "near-empty. Re-place (hybrid_mira) or rebuild the workspace first.")
        return 1

    if BENCH is not None:
        records, missing = load_bench_records(ws, args.n)
        print(f"probe records: {len(records)} (unresolved gold: {missing})")
    else:
        records = stability_records(ws, args.n)
        print(f"stability probe: {len(records)} recent queries "
              "(gold = pre-consolidation top-5)")
    if not records:
        print("FATAL: no resolvable gold ids — refusing to consolidate blind")
        return 1

    before = eval_retrieval(ws, records)
    print(f"before: mrr={before['aggregate'].get('mrr')} "
          f"recall={before['aggregate'].get('retrieval_recall')}")

    backup = None
    if args.apply:
        # snapshot BEFORE any write, not after — this is the undo path
        backup = os.path.join(ROOT, ".tmp", "consolidate_backup_mira.db")
        os.makedirs(os.path.dirname(backup), exist_ok=True)
        try:
            with ws.store.tx() as c:
                c.execute("VACUUM INTO ?", (backup,))  # consistent snapshot
        except Exception:
            shutil.copy2(os.path.join(DATA_DIR, "mira.db"), backup)
        print(f"backup: {backup}")

    replay = replay_paths(ws, args.replay_days, apply=args.apply)
    print(f"replay: {replay}")

    decay = apply_decay(ws.frame, half_life_days=args.half_life,
                        dry_run=not args.apply)
    print(f"decay: {decay}")

    def _persist_any(row: dict) -> None:
        """build_gists emits BOTH node rows and {'edge': True, ...} rows;
        route each to the right store call (upsert_node would choke on the
        edge rows — NOT NULL concept — which is exactly what the first
        apply run crashed on)."""
        if row.get("edge"):
            ws.store.add_edge(row["source_id"], row["target_id"],
                              row.get("relation_type", "gist_of"),
                              row.get("weight", 0.8),
                              row.get("confidence", 0.5))
        else:
            ws.store.upsert_node(row)

    gists = {"gists": 0, "members_linked": 0, "sectors": 0}
    if args.apply:
        if not decay.get("dry_run"):
            for node in ws.frame.nodes.values():
                ws.store.upsert_node({**node.to_row(), "_action": "update"})
        gists = build_gists(ws.frame, ws.embeddings,
                            max_gists=args.max_gists, persist=_persist_any)
        # embed gists so placement + index see them
        new_gist_nodes = [n for n in ws.frame.nodes.values()
                          if n.metadata.get("gist") and n.embedding is None]
        if new_gist_nodes:
            texts = [f"{n.concept}. {n.summary}" for n in new_gist_nodes]
            vecs = ws.embeddings.encode(texts)
            for node, vec in zip(new_gist_nodes, vecs):
                node.embedding = vec
        if backup is None:
            backup = os.path.join(ROOT, ".tmp", "consolidate_backup_mira.db")
            try:
                with ws.store.tx() as c:
                    c.execute("VACUUM INTO ?", (backup,))
            except Exception:
                shutil.copy2(os.path.join(DATA_DIR, "mira.db"), backup)
        # NEVER re-place here: the builder's canonical placement is applied
        # PER-DOCUMENT during ingestion, and a global place() call reshapes
        # the whole graph (ring-0/1 collapsed 4,168 -> 633 in the first apply
        # run, dropping MRR 0.70 -> 0.22). Gists are born at ring 1 with the
        # sector of their members; existing placements stay untouched.
        ws.reload()  # rebuild frame/graph/FAISS so gist vectors are searchable
        print("placements preserved (per-doc canonical); index rebuilt for gists")
    else:
        print("dry run: no changes written (gist build skipped; re-run with --apply)")

    after = eval_retrieval(ws, records)
    print(f"after: mrr={after['aggregate'].get('mrr')} "
          f"recall={after['aggregate'].get('retrieval_recall')}")

    sig = {}
    try:
        sig["mrr"] = paired_bootstrap(before["rows"], after["rows"], "mrr")
        sig["retrieval_recall"] = paired_bootstrap(before["rows"], after["rows"],
                                                   "retrieval_recall")
    except Exception as exc:  # significance is additive, never fatal
        sig["error"] = str(exc)

    results = {
        "workspace": {"nodes_before": n_nodes, "edges_before": n_edges,
                      "nodes_after": len(ws.frame.nodes),
                      "edges_after": len(ws.frame.edges)},
        "apply": args.apply,
        "replay": replay, "decay": decay, "gists": gists,
        "before": before["aggregate"], "after": after["aggregate"],
        "significance_before_vs_after": sig,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
