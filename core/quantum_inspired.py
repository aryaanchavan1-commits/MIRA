"""Quantum-inspired membership scoring — CLASSICAL mathematics, optional.

This module implements a *probabilistic amplitude representation* of memory
membership. It runs on ordinary CPUs/GPUs; nothing here is quantum hardware,
quantum speedup, or "quantum memory". The correct reading is:

    quantum-inspired computational representation (H6 in the protocol).

Mechanism. Where ``core.ncm.ConstellationIndex`` weights memberships by
normalized *similarity* (a linear L1 kernel), this backend maps each
membership probability to an amplitude ``a_c = sqrt(w_c)`` and combines
query/node membership with a Born-rule inner product:

    score(q, m) = ( sum_c a_q(c) * a_m(c) )^2

That single square is the entire "quantum-inspired" step: an *interference-
like* nonlinearity that boosts nodes agreeing with the query across SEVERAL
constellations at once (constructive overlap) and suppresses single-
constellation coincidences — the property H6 predicts helps ambiguous,
multi-context queries. For membership concentrated on one constellation both
kernels agree; they differ exactly when membership spreads, which is the
hypothesis being tested. If it does not measure better than the linear kernel
on the protocol's ambiguity benchmark, it is removed — that is what H6
rejection means (docs/EXPERIMENT_PROTOCOL.md).

Complexity is identical to the linear kernel: O(C) per score, O(N·C) for
assignment. No Hilbert-space dynamics is simulated; no quantum claim is made.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from core.ncm import ConstellationIndex, QUERY_KEY


def _amplitudes(weights: Tuple[float, ...]) -> Tuple[float, ...]:
    """Born rule: a_c = sqrt(w_c), the unique map with p(c) = |a_c|^2 = w_c."""
    return tuple(float(np.sqrt(w)) for w in weights)


class BornRuleIndex(ConstellationIndex):
    """Constellation membership scored by a squared-amplitude (Born-rule)
    kernel instead of the linear L1 kernel. Same fit, same budget caps, same
    centroid space; only the score combination differs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.amp_of: Dict[str, Tuple[float, ...]] = {}

    def _assign(self, nid, vec=None):  # type: ignore[override]
        ids, weights = super()._assign(nid, vec)
        if nid != QUERY_KEY:
            self.amp_of[nid] = _amplitudes(weights)
        return ids, weights

    def query_membership(self, qvec) -> Tuple[Tuple[str, ...], Tuple[float, ...]]:  # type: ignore[override]
        ids, weights = self._assign(QUERY_KEY, vec=qvec)
        return ids, weights

    def query_state(self, qvec) -> Tuple[Tuple[str, ...], Tuple[float, ...]]:
        """(ids, amplitudes) for the query."""
        ids, weights = self._assign(QUERY_KEY, vec=qvec)
        return ids, _amplitudes(weights)

    def node_score_born(self, nid: str, q_ids: Tuple[str, ...],
                        q_amp: Tuple[float, ...]) -> float:
        """Born-rule overlap: (sum_c a_q(c) * a_m(c))^2, clipped to [0, 1]."""
        m_ids = self.members.get(nid)
        if not m_ids or not q_ids:
            return 0.0
        m_amp = self.amp_of.get(nid, ())
        if len(m_amp) != len(m_ids):
            return 0.0
        amp = dict(zip(m_ids, m_amp))
        inner = sum(q_amp[i] * amp.get(c, 0.0) for i, c in enumerate(q_ids))
        return float(np.clip(inner * inner, 0.0, 1.0))

    def node_score(self, nid, q_ids, q_w):  # type: ignore[override]
        """Same signature as the base class so retrieval wiring is unchanged;
        converts L1 weights to amplitudes and applies the Born kernel."""
        return self.node_score_born(nid, q_ids, _amplitudes(q_w))


def make_membership_index(frame: MemoryFrame, config: dict,
                          existing: Optional[ConstellationIndex] = None,
                          ) -> Optional[ConstellationIndex]:
    """Factory honoring config ``ncm.backend``: 'linear' (default) | 'born'.

    Returns None when NCM is disabled — the sentinel retrieval.py checks.
    """
    from core.ncm import attach_constellations, ncm_config
    cfg = ncm_config(config)
    if not cfg.get("enabled"):
        return None
    if existing is not None:
        return existing
    if str(cfg.get("backend", "linear")).lower() == "born":
        idx = BornRuleIndex(top_k=cfg["top_k"],
                            max_constellations=cfg["max_constellations"],
                            overlap_floor=cfg["overlap_floor"])
        if idx.fit(frame):          # _assign() fills amp_of during fit
            idx.attach_to_nodes(frame)
        return idx                  # unfitted -> honest zeros
    return attach_constellations(frame, config)
