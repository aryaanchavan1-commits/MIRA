"""Context compression (spec §23).

raw evidence → deduplicate → rank → compress → final context.
Token estimates are ~4 chars/token. Provenance survives every step: each
output unit carries its node ids and chunk ids.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from core.retrieval import RetrievalResult
from core.types import stable_hash

logger = logging.getLogger("mira.compression")

_TOKEN_RE = re.compile(r"\S+")
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def est_tokens(text: str) -> int:
    return len(_TOKEN_RE.findall(text or ""))


def _query_relevance(sentence: str, query_tokens: set[str]) -> int:
    if not query_tokens:
        return 0
    sentence_tokens = {t.casefold() for t in _WORD_RE.findall(sentence)}
    return len(query_tokens & sentence_tokens)


@dataclass
class CompressedContext:
    text: str = ""
    n_tokens: int = 0
    n_raw_tokens: int = 0
    units: List[Dict[str, Any]] = field(default_factory=list)  # provenance per unit


def _render_parts(parts: List[str]) -> str:
    return "\n".join(f"[{i + 1}] {part}" for i, part in enumerate(parts))


def compress(result: RetrievalResult, max_tokens: int = 2048,
             target_ratio: float = 0.6) -> CompressedContext:
    """Pack ranked evidence into ``max_tokens`` and keep relevant sentences.

    The numbered citation prefix and mandala tags are part of the evidence sent
    to the model, so every candidate is measured after formatting rather than
    by counting only its body text.
    """
    try:
        max_tokens = max(0, int(max_tokens))
    except (TypeError, ValueError):
        max_tokens = 0

    ctx = CompressedContext()
    seen_hashes: set = set()
    parts: List[str] = []
    units: List[Dict[str, Any]] = []
    query_tokens = {t.casefold() for t in _WORD_RE.findall(
        getattr(result, "query", "") or "")}

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

        text = n.summary or n.raw_text or n.concept
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            sentences = [n.concept]
        if not sentences[0]:
            continue

        # Prefer sentences containing query terms.  A sentence that is too long
        # for the remaining formatted budget is skipped in favour of a shorter
        # relevant one; selected sentences are restored to source order below.
        ranked = sorted(
            enumerate(sentences),
            key=lambda pair: (-_query_relevance(pair[1], query_tokens), pair[0]),
        )
        selected: List[int] = []
        for index, sentence in ranked:
            if sentence in [sentences[i] for i in selected]:
                continue
            candidate_indices = sorted(selected + [index])
            body = " ".join(sentences[i] for i in candidate_indices)
            ring_tag = f"ring{n.ring}" if n.ring is not None else "ring?"
            sector_tag = n.sector or "general"
            candidate_part = f"({ring_tag} · {sector_tag}) {body}"
            candidate_text = _render_parts(parts + [candidate_part])
            if est_tokens(candidate_text) <= max_tokens:
                selected = candidate_indices
        if not selected:
            continue  # nothing answerable in this unit — contribute no fake evidence

        body = " ".join(sentences[i] for i in selected)
        ring_tag = f"ring{n.ring}" if n.ring is not None else "ring?"
        sector_tag = n.sector or "general"
        part = f"({ring_tag} · {sector_tag}) {body}"
        parts.append(part)
        units.append({
            "node_id": n.id, "concept": n.concept,
            "chunk_ids": n.source_ids, "ring": n.ring, "sector": n.sector,
            "score": round(item.score, 4),
            "tokens": est_tokens(f"[{len(parts)}] {part}"),
        })
        # The fully rendered context is within max_tokens by construction.  Stop
        # only when there is no room for another useful excerpt.
        if est_tokens(_render_parts(parts)) >= max_tokens:
            break

    ctx.text = _render_parts(parts)
    ctx.n_tokens = est_tokens(ctx.text)
    ctx.units = units
    ctx.n_raw_tokens = sum(
        est_tokens(i.node.summary or i.node.raw_text or i.node.concept)
        for i in result.items
    )
    return ctx
