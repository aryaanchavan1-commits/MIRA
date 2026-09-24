"""Ranking helpers (spec §21/§43).

The component scores themselves are computed in retrieval.py; this module
applies weights to precomputed component dicts so baselines and ablation
runs reuse exactly the same arithmetic (no duplicated scoring logic).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from core.retrieval import ALL_COMPONENTS, RetrievedItem


def apply_weights(components: Dict[str, float],
                  weights: Dict[str, float],
                  active: Optional[Sequence[str]] = None) -> float:
    """Weighted sum over active components, normalized by active weight mass."""
    active = set(active) if active else set(ALL_COMPONENTS)
    wsum = sum(weights.get(c, 0.0) for c in active) or 1.0
    return sum(weights.get(c, 0.0) * components.get(c, 0.0) for c in active) / wsum


def rank_items(items: List[RetrievedItem], weights: Dict[str, float],
               active: Optional[Sequence[str]] = None) -> List[RetrievedItem]:
    for it in items:
        it.score = apply_weights(it.components, weights, active)
    return sorted(items, key=lambda it: -it.score)
