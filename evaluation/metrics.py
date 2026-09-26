"""Metrics (spec §31).

All metrics are computed from actual retrieval/answer output. Answer quality
uses lexical overlap (token-F1) — an offline proxy, not an LLM judge; this is
stated wherever results are shown. No metric is ever fabricated.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


def token_f1(answer: str, gold: str) -> float:
    """Lexical token-F1 between produced answer and reference answer."""
    a, g = tokens(answer), tokens(gold)
    if not a or not g:
        return 0.0
    common = sum((Counter(a) & Counter(g)).values())
    if not common:
        return 0.0
    prec = common / len(a)
    rec = common / len(g)
    return 2 * prec * rec / (prec + rec)


def recall_at_k(retrieved_ids: Sequence[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved_ids: Sequence[str], relevant: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant) / k


def mrr(retrieved_ids: Sequence[str], relevant: Set[str], k: Optional[int] = None) -> float:
    ids = retrieved_ids if k is None else retrieved_ids[:max(0, k)]
    for i, rid in enumerate(ids, start=1):
        if rid in relevant:
            return 1.0 / i
    return 0.0


def context_tokens(context_text: str) -> int:
    """Cheap token estimate (~4 chars/token) for budget accounting."""
    if not (context_text or "").strip():
        return 0
    return max(1, len(context_text) // 4)


def aggregate(rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """Mean of each numeric metric over per-question rows."""
    keys: Set[str] = set()
    rows = list(rows)
    for r in rows:
        keys.update(k for k, v in r.items() if isinstance(v, (int, float)))
    out = {}
    for k in sorted(keys):
        vals = [float(r[k]) for r in rows if k in r]
        if vals:
            out[k] = round(sum(vals) / len(vals), 4)
            out[f"n_scored_{k}"] = len(vals)
    out["n_questions"] = len(rows)
    return out
