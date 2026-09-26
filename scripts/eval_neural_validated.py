"""Held-out, document-grouped validation of the learned retrieval scorer.

Why this script exists: config.yaml only allows retrieval_score.learned=true
"after a held-out, document-grouped validation run records the scorer and
split". This is that run.

Design:
  - Group = document. Every fact node belongs to exactly one document; no
    document contributes nodes to both train and holdout (no leakage).
  - Train: fit DeltaRuleScorer on queries built from train docs' fact nodes.
  - Holdout: same procedure on holdout docs, but scored with the TRAINED
    weights vs the hand weights. Report MRR/recall deltas with paired
    bootstrap significance per-question.
  - Decision rule is printed, not applied: config stays honest unless the
    holdout says the learned scorer wins.

Usage: .venv/Scripts/python.exe scripts/eval_neural_validated.py [--holdout-frac 0.3]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.auto_config import build_context  # noqa: E402
from core.neural import FEATURES, DeltaRuleScorer  # noqa: E402
from core.retrieval import MIRARetriever  # noqa: E402
from core.workspace import Workspace  # noqa: E402
from scripts.eval_significance import paired_bootstrap  # noqa: E402

OUT = "experiments/neural_validation.json"


def build_pairs(ws, retriever, docs, topk):
    """(features, label, question, node_id) rows for the given documents."""
    X, y, meta = [], [], []
    for doc_id in docs:
        nodes = [n for n in ws.frame.nodes.values()
                 if doc_id in (n.source_ids or [])]
        for n in nodes:
            q = n.concept or (n.summary or "")[:80]
            if not q or not q.strip():
                continue
            qvec = ws.embeddings.encode([q])[0]
            res = retriever.retrieve(q, qvec, active_components=None, final_k=topk)
            pos = {n.id} | set(ws.gs.neighborhood(n.id, radius=1))
            for it in res.items:
                X.append([it.components[f] for f in FEATURES])
                y.append(1.0 if it.node.id in pos else 0.0)
                meta.append({"question": q, "node_id": n.id,
                             "cand_id": it.node.id})
    return X, y, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ctx = build_context()
    ws = Workspace(embeddings=ctx.embeddings, llm=None, config=ctx.cfg)
    retriever = MIRARetriever(ws.frame, ws.vs, ws.gs, ws.config)
    hand_weights = dict(retriever.weights)

    docs = sorted({sid for n in ws.frame.nodes.values()
                   for sid in (n.source_ids or []) if sid.startswith("doc_")})
    rng = random.Random(args.seed)
    rng.shuffle(docs)
    n_hold = max(2, int(len(docs) * args.holdout_frac))
    hold_docs, train_docs = set(docs[:n_hold]), set(docs[n_hold:])
    print(f"{len(docs)} documents -> {len(train_docs)} train / {len(hold_docs)} holdout (grouped by document)")

    print("building training pairs…", flush=True)
    X_tr, y_tr, _ = build_pairs(ws, retriever, train_docs, args.topk)
    print(f"train: {len(y_tr)} pairs ({int(sum(y_tr))} pos)")
    if len(y_tr) < 50 or sum(y_tr) < 5:
        print("insufficient training signal")
        return 1

    scorer = DeltaRuleScorer(epochs=args.epochs)
    hist = scorer.fit(X_tr, y_tr)
    learned_weights = scorer.weights()
    print(f"trained: loss {hist['first_loss']} -> {hist['final_loss']}")
    for f in FEATURES:
        print(f"  {f:12s} hand={hand_weights.get(f, 0):.3f} learned={learned_weights.get(f, 0):.3f}")

    print("building holdout pairs…", flush=True)
    X_ho, y_ho, meta_ho = build_pairs(ws, retriever, hold_docs, args.topk)
    print(f"holdout: {len(y_ho)} pairs ({int(sum(y_ho))} pos)")

    # Per-question MRR under both weightings, on holdout candidates only.
    by_q = {}
    for row, label, m in zip(X_ho, y_ho, meta_ho):
        by_q.setdefault(m["question"], []).append((row, label, m["cand_id"]))

    def mrr_per_q(weights):
        rows = []
        for q, cands in by_q.items():
            ranked = sorted(cands, key=lambda rc: (-sum(w * f for w, f in zip(weights, rc[0])), rc[2]))
            rr = 0.0
            for rank, (_, label, _) in enumerate(ranked, 1):
                if label > 0:
                    rr = 1.0 / rank
                    break
            rows.append({"question": q, "mrr": rr})
        return rows

    rows_hand = mrr_per_q([hand_weights.get(f, 0.0) for f in FEATURES])
    rows_learned = mrr_per_q([learned_weights.get(f, 0.0) for f in FEATURES])
    mean = lambda rs: sum(r["mrr"] for r in rs) / max(1, len(rs))
    bt = paired_bootstrap(rows_learned, rows_hand, "mrr")
    print(f"\nholdout MRR  hand={mean(rows_hand):.4f}  learned={mean(rows_learned):.4f}  "
          f"delta={bt['mean_diff']}  CI95={bt['ci_low']}..{bt['ci_high']}  p≈{bt['p_value']}")

    verdict = (bt["mean_diff"] > 0 and bt["ci_low"] > 0) or bt["p_value"] < 0.05
    print(f"decision: {'ENABLE learned: true' if verdict else 'KEEP learned: false'} "
          "(config is only flipped when the holdout win is significant)")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"seed": args.seed, "holdout_frac": args.holdout_frac,
                   "n_docs": len(docs), "n_train_docs": len(train_docs),
                   "n_holdout_docs": len(hold_docs),
                   "n_train_pairs": len(y_tr), "n_holdout_pairs": len(y_ho),
                   "hand_weights": hand_weights, "learned_weights": learned_weights,
                   "holdout_mrr_hand": round(mean(rows_hand), 4),
                   "holdout_mrr_learned": round(mean(rows_learned), 4),
                   "significance": bt, "verdict_enable": bool(verdict),
                   "epochs": args.epochs}, fh, indent=1)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
