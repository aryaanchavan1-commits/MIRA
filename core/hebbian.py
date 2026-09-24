"""Hebbian edge reinforcement — consolidation of used pathways.

Third biological timescale in MIRA:
  • fast:    spreading activation during retrieval      (core/activation.py)
  • medium:  activation dynamics shape each answer      (retrieval scoring)
  • slow:    Hebbian plasticity — edges co-active in
             successful retrievals strengthen           (this module)

"Neurons that fire together wire together" (Hebb 1949). When a retrieval
path leads to an answer the user keeps (or a benchmark labels relevant),
the edges along that path gain weight; all edges decay slightly each
consolidation pass (synaptic homeostasis), so unused paths fade and used
paths dominate future spreading activation.

Honesty (spec §49): a biologically-inspired plasticity rule on graph edge
weights, not a model of synaptic biochemistry.

Conservative by design: bounded, decay-stabilized, and every reinforcement
is logged. Weights are clamped to [0.05, 2.0].
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("mira.hebbian")

W_MIN, W_MAX = 0.05, 2.0


def reinforce(frame, paths: List[List[str]], lr: float = 0.10,
              decay: float = 0.02, store=None) -> List[Tuple[str, str, float]]:
    """Strengthen edges along used paths; decay all edges slightly.

    frame: MemoryFrame (edges updated in place and persisted via store)
    paths: list of node-id chains that produced good answers
    lr:    potentiation rate per co-activation
    decay: homeostatic decay applied to every edge each pass
    store: optional SQLiteStore to persist updated edge weights

    Returns the list of (source, target, new_weight) changes.
    """
    if not paths:
        return []
    touched: Dict[Tuple[str, str], float] = {}
    pairs: List[Tuple[str, str]] = []
    for path in paths:
        for a, b in zip(path, path[1:]):
            pairs.append((a, b))

    # potentiation along used edges
    for a, b in pairs:
        e = frame.get_edge(a, b)
        if e is None:
            continue
        old = e.weight
        new = min(W_MAX, old + lr * (1.0 + 0.5 * e.confidence))
        e.weight = round(new, 4)
        touched[(a, b)] = e.weight

    # homeostatic decay on everything else (normalized so consolidation
    # doesn't inflate the graph's total weight without bound)
    n_reinforced = len(touched)
    if n_reinforced and frame.edges:
        d = decay / math.sqrt(n_reinforced)
        for e in frame.edges:
            if (e.source_id, e.target_id) not in touched:
                e.weight = round(max(W_MIN, e.weight * (1.0 - d)), 4)

    if store is not None:
        for (a, b), w in touched.items():
            try:
                store.update_edge_weight(a, b, w)
            except Exception as exc:
                logger.warning("edge weight persist failed for %s->%s: %s", a, b, exc)
                break
    logger.info("hebbian consolidation: %d edges strengthened along %d paths",
                n_reinforced, len(paths))
    return [(a, b, w) for (a, b), w in touched.items()]
