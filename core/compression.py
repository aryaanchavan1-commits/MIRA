"""Context compression (spec §23).

raw evidence → deduplicate → rank → compress → final context.
Token estimates are ~4 chars/token. Provenance survives every step: each
output unit carries its node ids and chunk ids.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.retrieval import RetrievedItem, RetrievalResult
from core.types import stable_hash

logger = logging.getLogger("mira.compression")

_TOKEN_RE = re.compile(r"\S+")


def est_tokens(text: str) -> int:
    return max(1, len(_TOKEN_RE.findall(text)))


@dataclass
class CompressedContext:
    text: str = ""
    n_tokens: int = 0
    n_raw_tokens: int = 0
    units: List[Dict[str, Any]] = field(default_factory=list)  # provenance per unit


def compress(result: RetrievalResult, max_tokens: int = 2048,
             target_ratio: float = 0.6) -> CompressedContext:
    """Pack ranked evidence into max_tokens, dedup near-duplicates.

    Sentence-level compression: keep the sentences of each memory most
    relevant to the query, drop duplicates, preserve order by score.
    """
    ctx = CompressedContext()
    seen_hashes: set = set()
    parts: List[str] = []
    budget = max_tokens
    units: List[Dict[str, Any]] = []

    for item in result.items:  # already score-ranked
        n = item.node
        # structural/label nodes (no text of their own) contribute via the
        # graph, not as evidence text — packing them as "[n] <label>" units
        # just poisons the prompt and the extractive fallback
        if not (n.summary or n.raw_text):
            continue
        # dedup identical summaries/raw text
        h = stable_hash((n.summary or n.raw_text)[:2000])
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        # sentence split + selection
        text = n.summary or n.raw_text or n.concept
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            sentences = [n.concept]
        kept: List[str] = []
        unit_tokens = 0
        for idx, s in enumerate(sentences):
            t = est_tokens(s)
            # first sentence always allowed: a unit must carry evidence text,
            # never fall back to the bare concept label (that is how label-only
            # "evidence" used to leak into prompts)
            if idx > 0 and unit_tokens + t > max(16, budget // max(1, len(result.items))):
                break
            if est_tokens(" ".join(kept + [s])) > max_tokens:
                break
            kept.append(s)
            unit_tokens += t
        if not kept:
            continue  # nothing answerable in this unit — contribute no fake evidence
        body = " ".join(kept)
        # topology-aware context: the LLM sees WHERE evidence lives in the
        # mandala (ring = generality, sector = topic), not a flat list
        ring_tag = f"ring{n.ring}" if n.ring is not None else "ring?"
        sector_tag = (n.sector or "general")
        parts.append(f"({ring_tag} · {sector_tag}) {body}")
        units.append({
            "node_id": n.id, "concept": n.concept,
            "chunk_ids": n.source_ids, "ring": n.ring, "sector": n.sector,
            "score": round(item.score, 4), "tokens": unit_tokens,
        })
        budget -= unit_tokens
        if budget <= 32:
            break

    ctx.text = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(parts))
    ctx.n_tokens = est_tokens(ctx.text)
    ctx.units = units
    ctx.n_raw_tokens = sum(est_tokens(i.node.raw_text or i.node.concept) for i in result.items)
    return ctx
