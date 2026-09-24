"""Fit the delta-rule neural scorer on corpus QA data.

Generates real training pairs from the ingested corpus: for each fact node,
its own concept is the query, the node itself and its graph neighbors are
positives, and every other retrieved candidate is a hard negative — exactly
the ranking confusions the scorer must learn to fix. Features are computed
with the real retriever so train/deploy features match exactly. Usage:

    .venv/Scripts/python.exe scripts/train_neural.py [--epochs 60]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.auto_config import DATA_DIR, build_context  # noqa: E402
from core.neural import FEATURES, DeltaRuleScorer  # noqa: E402

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mira.train_neural")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--topk", type=int, default=16)
    args = ap.parse_args()

    ctx = build_context()
    from core.retrieval import MIRARetriever
    from core.workspace import Workspace
    w = Workspace(embeddings=ctx.embeddings, llm=ctx.llm, config=ctx.cfg)

    fact_nodes = [n for n in w.frame.nodes.values() if n.source_ids]
    if len(fact_nodes) < 4:
        print("Need >= 4 fact nodes (ingest documents first).")
        return 1

    retriever = MIRARetriever(w.frame, w.vs, w.gs, w.config)
    X_rows, y_rows = [], []
    for n in fact_nodes:
        q = n.concept or (n.summary or "")[:80]
        if not q.strip():
            continue
        qvec = w.embeddings.encode([q])[0]
        res = retriever.retrieve(q, qvec, active_components=None, final_k=args.topk)
        pos = {n.id} | set(w.gs.neighborhood(n.id, radius=1))
        for it in res.items:
            X_rows.append([it.components[f] for f in FEATURES])
            y_rows.append(1.0 if it.node.id in pos else 0.0)

    import numpy as np
    X = np.asarray(X_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    if y.sum() < 2 or (1 - y).sum() < 2:
        print("Not enough label balance to train (need both classes).")
        return 1

    s = DeltaRuleScorer(epochs=args.epochs)
    hist = s.fit(X, y)
    s.save(os.path.join(DATA_DIR, "neural_scorer.json"))
    print(f"trained on {len(y)} pairs ({int(y.sum())} pos / {int((1 - y).sum())} neg)")
    print(f"loss {hist['first_loss']} -> {hist['final_loss']}")
    print("learned weights:")
    for f, v in s.weights().items():
        print(f"  {f:12s} {v:.4f}")
    print("enable with retrieval_score.learned: true in config/config.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
