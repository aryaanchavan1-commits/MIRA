"""Conflict detection & resolution (spec §18).

Contradictory facts are never overwritten — both stay in memory. At answer
time they are ranked by temporal validity, source authority, confidence,
recency and corroboration. Uncertainty is represented explicitly.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.memory import MemoryNode
from core.types import iso_now, utcnow, datetime

logger = logging.getLogger("mira.conflicts")

# "X is 5" vs "X is 7" on the same subject → same predicate, different value
_VALUE_RE = re.compile(r"^(?P<subject>.+?)\s+(?:is|was|are|=|:)\s+(?P<value>.+)$", re.I)


def _parse_proposition(text: str) -> Optional[Tuple[str, str]]:
    m = _VALUE_RE.match(text.strip())
    if not m:
        return None
    return m.group("subject").strip().lower(), m.group("value").strip().lower()


def detect_conflict(node_a: MemoryNode, node_b: MemoryNode) -> bool:
    """True when same subject, different normalized value, both fact-like."""
    if node_a.memory_type.value not in ("fact", "event") or \
       node_b.memory_type.value not in ("fact", "event"):
        return False
    pa, pb = _parse_proposition(node_a.raw_text or node_a.concept), \
             _parse_proposition(node_b.raw_text or node_b.concept)
    if not pa or not pb or pa[0] != pb[0]:
        return False
    return pa[1] != pb[1]


def find_conflicts(nodes: List[MemoryNode]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    checked: set = set()
    facts = [n for n in nodes if n.memory_type.value in ("fact", "event")]
    for i, a in enumerate(facts):
        for b in facts[i + 1:]:
            key = tuple(sorted((a.id, b.id)))
            if key in checked:
                continue
            checked.add(key)
            if detect_conflict(a, b):
                pairs.append(key)
    return pairs


def resolution_score(node: MemoryNode, source_authority: float = 0.5) -> float:
    """Higher wins when both sides of a conflict are shown (experimental)."""
    recency = 0.5
    try:
        age_h = (utcnow() - datetime.fromisoformat(node.updated_at)).total_seconds() / 3600
        recency = float(np.clip(1.0 - age_h / (24 * 30), 0.0, 1.0))
    except Exception:
        pass
    corroborated = min(1.0, len(node.source_ids) * 0.5)
    active = node.valid_until is None
    return (0.30 * float(active) + 0.25 * source_authority +
            0.20 * float(np.clip(node.confidence, 0, 1)) +
            0.15 * recency + 0.10 * corroborated)


def resolve(nodes: List[MemoryNode],
            authorities: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """For each conflict group, order candidates by resolution score.

    Returns {subject: [node ids in resolution order]}. No deletion — the
    losing side remains in memory with its lower rank (§18).
    """
    authorities = authorities or {}
    conflicts = find_conflicts(nodes)
    groups: Dict[str, List[MemoryNode]] = {}
    for a_id, b_id in conflicts:
        for n in nodes:
            if n.id in (a_id, b_id):
                subj = (_parse_proposition(n.raw_text or n.concept) or ("?", ""))[0]
                groups.setdefault(subj, [])
                if all(m.id != n.id for m in groups[subj]):
                    groups[subj].append(n)
    out: Dict[str, List[str]] = {}
    for subj, members in groups.items():
        members.sort(key=lambda n: -resolution_score(n, authorities.get(n.id, 0.5)))
        out[subj] = [m.id for m in members]
    return out
