"""Paired significance tests for benchmark comparisons.

- Paired bootstrap over per-question metric differences (percentile CI).
- Wilcoxon signed-rank on the same pairs (scipy when available).

Both operate per-question, so question difficulty is controlled for — the
same question is compared across systems (paired design).
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple


def per_question(rows_a: List[dict], rows_b: List[dict], metric: str
                 ) -> Tuple[List[float], List[float]]:
    """Align rows by question; return (a_vals, b_vals) for the metric."""
    idx_a = {r.get("question"): r for r in rows_a}
    idx_b = {r.get("question"): r for r in rows_b}
    common = sorted(set(idx_a) & set(idx_b))
    a, b = [], []
    for q in common:
        va, vb = idx_a[q].get(metric), idx_b[q].get(metric)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            a.append(float(va))
            b.append(float(vb))
    return a, b


def paired_bootstrap(rows_a: List[dict], rows_b: List[dict], metric: str,
                     n_boot: int = 10000, seed: int = 42) -> Dict[str, float]:
    """Bootstrap CI for mean(a - b) over shared questions."""
    a, b = per_question(rows_a, rows_b, metric)
    n = len(a)
    if n < 2:
        return {"n_pairs": n, "mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "p_value": 1.0}
    diffs = [x - y for x, y in zip(a, b)]
    if not any(diffs):  # all-zero diffs: no evidence of difference, not max
        return {"n_pairs": n, "mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "p_value": 1.0}
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    mean_diff = sum(diffs) / n
    # two-sided p: how far 0 is inside the bootstrap distribution
    beyond = sum(1 for x in boots if (mean_diff >= 0) == (x < 0))
    p_value = max(2 * beyond / n_boot, 1.0 / n_boot)
    return {"n_pairs": n, "mean_diff": round(mean_diff, 5),
            "ci_low": round(lo, 5), "ci_high": round(hi, 5),
            "p_value": p_value}


def wilcoxon(rows_a: List[dict], rows_b: List[dict], metric: str) -> Dict[str, float]:
    """Wilcoxon signed-rank test on per-question pairs (scipy)."""
    a, b = per_question(rows_a, rows_b, metric)
    out: Dict[str, float] = {"n_pairs": len(a)}
    try:
        from scipy.stats import wilcoxon as _wx
    except ImportError:
        out["error"] = "scipy unavailable"
        return out
    if len(a) < 6:
        out["error"] = "too few pairs"
        return out
    stat, p = _wx(a, b)
    out["statistic"], out["p_value"] = float(stat), float(p)
    return out
