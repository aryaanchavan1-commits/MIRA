"""Simulated aging + sleep: does consolidation recover used memories?

The consolidation pass (scripts/consolidate.py) measured harmless on a FRESH
corpus — nothing had decayed yet, so there was nothing to recover. This
experiment ages the bench workspace 60 simulated days of disuse and asks the
question the benefit hypothesis actually makes:

  1. do aged-but-REPLAYED memories keep their retrieval rank (sleep
     consolidation), while aged-UNREPLAYED ones fade (Ebbinghaus)?
  2. do gist nodes surface MORE as their member details decay (gist
     abstraction outliving verbatim detail)?

Design (aging/replay/decay live in the in-memory frame only; the ONLY db
writes are additive synthetic retrieval_logs marked system='aging_sim'):

  1. BEFORE  — measure the current post-consolidation bench on its own 300
     MuSiQue questions (same harness as the paper) + gist probe.
  2. AGE     — 80% of nodes get updated_at backdated 60 days (seeded RNG,
     deterministic); 20% stay "recently used".
  3. REPLAY  — inject ~60 synthetic retrieval_logs whose node_ids cover HALF
     the aged gold ids (as the answer pipeline's logs would), then run the
     real replay_paths (hebbian paths + importance lift + clock reset).
  4. DECAY   — real apply_decay (Ebbinghaus) over the aged frame.
  5. AFTER   — re-measure; split rows into cohorts by gold-id treatment:
     aged_replayed / aged_faded / recent.

Usage:
  .venv/Scripts/python.exe scripts/eval_aging.py [--n 300] [--age-days 60]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timedelta

if hasattr(sys.stdout, "reconfigure"):  # cp1252 consoles choke on fancy glyphs
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import core.workspace as cw  # noqa: E402
import config.auto_config as ac  # noqa: E402

from config.auto_config import load_config  # noqa: E402
from core.memory_dynamics import apply_decay  # noqa: E402
from core.types import iso_now, stable_hash  # noqa: E402
from scripts.consolidate import (  # noqa: E402
    build_embeddings,
    eval_retrieval,
    load_bench_records,
    replay_paths,
)
from scripts.eval_significance import paired_bootstrap  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data_bench")
OUT = os.path.join(DATA_DIR, "aging_results.json")
K = 8


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def age_frame(frame, fraction: float, days: int, seed: int = 42) -> set:
    """Backdate a seeded fraction of nodes' updated_at (in memory only)."""
    rng = random.Random(seed)
    aged = {nid for nid in sorted(frame.nodes) if rng.random() < fraction}
    old_ts = (datetime.utcnow() - timedelta(days=days)).isoformat(timespec="seconds")
    for nid in aged:
        frame.nodes[nid].updated_at = old_ts
    return aged


def inject_replay_logs(ws, records, aged: set, replay_frac: float = 0.5,
                       n_rows: int = 60, per_row: int = 6, seed: int = 43) -> dict:
    """Write synthetic retrieval logs over HALF the aged gold ids.

    Each row plays one 'session' that co-activated per_row gold nodes — the
    same shape the answer pipeline's real logs have (meta.node_ids), so
    replay_paths() consumes them unmodified. Gold-only ids keep the seed
    set under replay_paths' 400-seed cap, so nothing is silently dropped.
    """
    gold_ids = sorted({nid for r in records for nid in r["supporting_ids"]})
    aged_golds = [nid for nid in gold_ids if nid in aged]
    rng = random.Random(seed)
    rng.shuffle(aged_golds)
    replay_golds = set(aged_golds[:max(1, int(len(aged_golds) * replay_frac))])
    if not replay_golds:
        return {"rows": 0, "aged_golds": len(aged_golds), "replay_golds": 0}
    pool = sorted(replay_golds)
    with ws.store.tx() as c:
        # idempotent restarts: previous runs' synthetic rows are replaced
        c.execute("DELETE FROM retrieval_logs WHERE system='aging_sim'")
        for i in range(n_rows):
            ids = [pool[j % len(pool)] for j in range(i, i + per_row)]
            q = f"aging-sim session {i}: multi-hop lookup"
            c.execute(
                "INSERT INTO retrieval_logs(at,query_hash,system,n_candidates,"
                "n_final,latency_ms,context_tokens,meta) VALUES (?,?,?,?,?,?,?,?)",
                (iso_now(), stable_hash(q), "aging_sim", len(ids), len(ids),
                 12.0, 64,
                 json.dumps({"query": q, "node_ids": ids, "aging_sim": True})))
    return {"rows": n_rows, "aged_golds": len(aged_golds),
            "replay_golds": len(replay_golds)}


def classify_cohorts(records, aged: set, replayed: set) -> dict:
    """question -> cohort, by how its gold ids were treated."""
    coh = {}
    for r in records:
        gold = set(r["supporting_ids"])
        if gold & replayed:
            coh[r["question"]] = "aged_replayed"
        elif gold & aged:
            coh[r["question"]] = "aged_faded"
        else:
            coh[r["question"]] = "recent"
    return coh


def _mean(rows, key: str) -> float:
    vals = [r.get(key) or 0.0 for r in rows]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def cohort_table(rows, cohort_of: dict) -> dict:
    out = {}
    for name in ("aged_replayed", "aged_faded", "recent"):
        subset = [r for r in rows if cohort_of.get(r["question"]) == name]
        out[name] = {"n_records": len(subset),
                     "mrr": _mean(subset, "mrr"),
                     "retrieval_recall": _mean(subset, "retrieval_recall")}
    return out


def gist_probe(ws, gists, k: int = K) -> dict:
    """Are gist nodes retrievable for their own summary, and do they outrank
    their (fading) member details? Run against one retrieval fn per phase."""
    fn = None  # fresh pipeline per call, built below
    rows = []
    for g in gists:
        if fn is None:
            from evaluation.benchmark import mira_retrieve_fn
            fn = mira_retrieve_fn(ws, k=k)
        q = (g.summary or g.concept)[:200]
        qvec = ws.embeddings.encode([q])[0]
        out = fn(q, qvec)
        ids = list(out.get("node_ids") or [])
        grank = ids.index(g.id) + 1 if g.id in ids else 0
        members = [m for m in g.metadata.get("members", []) if m in ids]
        mrank = min((ids.index(m) + 1 for m in members), default=0)
        rows.append({"gist_rank": grank, "member_rank": mrank})
    n = max(len(rows), 1)
    return {
        "n_probes": len(rows),
        "gist_hit": round(sum(1 for r in rows if r["gist_rank"]) / n, 4),
        "gist_mrr": round(sum(1 / r["gist_rank"] for r in rows if r["gist_rank"]) / n, 4),
        "member_hit": round(sum(1 for r in rows if r["member_rank"]) / n, 4),
        "member_mrr": round(sum(1 / r["member_rank"] for r in rows if r["member_rank"]) / n, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=300, help="probe questions")
    ap.add_argument("--age-days", type=int, default=60)
    ap.add_argument("--age-fraction", type=float, default=0.80)
    ap.add_argument("--half-life", type=float, default=21.0)
    ap.add_argument("--gists", type=int, default=40, help="gist-probe samples")
    args = ap.parse_args()

    cw.DATA_DIR = DATA_DIR
    ac.DATA_DIR = DATA_DIR
    cfg = load_config(None)
    embeddings, cfg = build_embeddings()
    ws = cw.Workspace(embeddings=embeddings, llm=None, config=cfg)
    n_nodes, n_edges = len(ws.frame.nodes), len(ws.frame.edges)
    log(f"workspace: {n_nodes} nodes, {n_edges} edges")

    ring01 = sum(1 for n in ws.frame.nodes.values() if n.ring in (0, 1))
    rate = ring01 / max(n_nodes, 1)
    log(f"ring-0/1 rate: {rate:.3%} ({ring01}/{n_nodes})")
    if n_nodes > 2000 and rate < 0.02:
        log("FATAL: ring-0/1 rate < 2% — re-place (hybrid_mira) or rebuild first.")
        return 1

    records, missing = load_bench_records(ws, args.n)
    log(f"probe records: {len(records)} (unresolved gold: {missing})")
    if not records:
        log("FATAL: no resolvable gold ids")
        return 1

    # ---- 1. BEFORE ------------------------------------------------------
    before = eval_retrieval(ws, records)
    log(f"before: mrr={before['aggregate'].get('mrr')} "
        f"recall={before['aggregate'].get('retrieval_recall')}")

    gists = [n for n in ws.frame.nodes.values() if n.metadata.get("gist")]
    rng = random.Random(44)
    probe_gists = rng.sample(gists, min(args.gists, len(gists))) if gists else []
    gp_before = gist_probe(ws, probe_gists)
    log(f"gist probe before: {gp_before}")

    # ---- 2-4. age, replay, decay (in-memory) ----------------------------
    aged = age_frame(ws.frame, args.age_fraction, args.age_days)
    log(f"aged {len(aged)}/{n_nodes} nodes by {args.age_days}d "
        f"(fraction {args.age_fraction:.0%})")

    inj = inject_replay_logs(ws, records, aged)
    log(f"injected replay logs: {inj}")
    replay = replay_paths(ws, 30, apply=False)  # in-memory: no store writes
    log(f"replay: {replay}")

    decay = apply_decay(ws.frame, half_life_days=args.half_life)
    log(f"decay: {decay}")

    # ---- 5. AFTER --------------------------------------------------------
    after = eval_retrieval(ws, records)
    log(f"after: mrr={after['aggregate'].get('mrr')} "
        f"recall={after['aggregate'].get('retrieval_recall')}")

    gp_after = gist_probe(ws, probe_gists)
    log(f"gist probe after: {gp_after}")

    # replayed set = aged golds covered by the injected logs
    replayed = set()
    try:
        with ws.store.tx() as c:
            rows = c.execute(
                "SELECT meta FROM retrieval_logs WHERE system='aging_sim'").fetchall()
            for (meta_json,) in rows:
                meta = json.loads(meta_json or "{}")
                replayed.update((meta.get("node_ids") or []))
    except Exception as exc:
        log(f"WARN: could not re-read injected logs: {exc}")
    cohort_of = classify_cohorts(records, aged, replayed)
    cohorts = {"before": cohort_table(before["rows"], cohort_of),
               "after": cohort_table(after["rows"], cohort_of)}

    sig = {}
    try:
        sig["mrr"] = paired_bootstrap(before["rows"], after["rows"], "mrr")
        sig["retrieval_recall"] = paired_bootstrap(before["rows"], after["rows"],
                                                   "retrieval_recall")
    except Exception as exc:  # significance is additive, never fatal
        sig["error"] = str(exc)

    results = {
        "design": {
            "nodes": n_nodes, "edges": n_edges, "records": len(records),
            "age_days": args.age_days, "age_fraction": args.age_fraction,
            "aged_nodes": len(aged), "half_life_days": args.half_life,
            "injected_logs": inj, "persisted": "synthetic retrieval_logs only; "
            "aging/replay/decay are in-memory (system='aging_sim' rows)",
        },
        "before": before["aggregate"], "after": after["aggregate"],
        "significance_before_vs_after": sig,
        "cohorts": cohorts,
        "replay": replay, "decay": decay,
        "gist_probe": {"n_gists_total": len(gists), "before": gp_before,
                       "after": gp_after},
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    log(f"wrote {OUT}")

    # ---- headline --------------------------------------------------------
    for name in ("aged_replayed", "aged_faded", "recent"):
        b, a = cohorts["before"][name], cohorts["after"][name]
        log(f"cohort {name}: n={a['n_records']} "
            f"mrr {b['mrr']}->{a['mrr']} (d={round(a['mrr'] - b['mrr'], 4)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
