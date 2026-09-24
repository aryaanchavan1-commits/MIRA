"""Extraction (spec §16/§17).

Extracts entities, concepts, relations and summaries from chunks. Uses the
local LLM when available; otherwise deterministic fallbacks (n-gram
concepts, co-occurrence relations, lead sentences). Returns provenance-safe
dicts — no execution of document content (§41).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("mira.extract")

_STOP = {
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can",
    "her", "was", "one", "our", "out", "day", "get", "has", "him", "his",
    "how", "man", "new", "now", "old", "see", "two", "way", "who", "its",
    "did", "that", "this", "with", "from", "they", "have", "will", "your",
    "what", "when", "which", "their", "said", "each", "she", "them", "then",
    "were", "been", "more", "these", "those", "such", "into", "than",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _capitalized(text: str) -> List[str]:
    """Candidate entities: capitalized sequences (names), kept in original case."""
    out = []
    for m in re.finditer(r"\b([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\b", text):
        phrase = m.group(1).strip()
        if phrase.lower() not in _STOP and len(phrase) > 2:
            out.append(phrase)
    return out


def extract_concepts_fallback(chunks: List[str], top_k: int = 12) -> List[str]:
    """TF-based concept extraction — deterministic, no model needed."""
    from collections import Counter
    counter: Counter = Counter()
    for c in chunks:
        seen_in_chunk = set()
        for t in _tokens(c):
            if t in _STOP or len(t) < 4:
                continue
            if t not in seen_in_chunk:  # document-frequency style
                counter[t] += 1
                seen_in_chunk.add(t)
    return [tok for tok, _ in counter.most_common(top_k)]


def extract_entities_fallback(chunk: str) -> List[str]:
    out = list(dict.fromkeys(_capitalized(chunk)))[:8]
    if not out:  # lowercase text → fall back to frequent non-stop tokens
        out = extract_concepts_fallback([chunk], top_k=3)
    return out


RELATION_PATTERNS: List[Tuple[str, str]] = [
    (r"([A-Za-z][\w\s-]{2,40}?)\s+is\s+(?:a|an|the)\s+([A-Za-z][\w\s-]{2,40})", "is_a"),
    (r"([A-Za-z][\w\s-]{2,40}?)\s+(?:was\s+)?(?:created|developed|written|founded|built)\s+by\s+([A-Z][\w\s-]{2,40})", "created_by"),
    (r"([A-Za-z][\w\s-]{2,40}?)\s+(?:uses|supports|includes|contains|enables)\s+([A-Za-z][\w\s-]{2,40})", "uses"),
    (r"([A-Za-z][\w\s-]{2,40}?)\s+(?:released|published|introduced)\s+(?:in|on)\s+([0-9]{4})", "released_in"),
]


def extract_relations_fallback(chunk: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for pat, rel in RELATION_PATTERNS:
        for m in re.finditer(pat, chunk):
            subj = m.group(1).strip().rstrip(".,;")
            obj = m.group(2).strip().rstrip(".,;")
            if subj.lower() not in _STOP and obj.lower() not in _STOP:
                out.append({"subject": subj, "relation": rel, "object": obj})
    return out[:6]


def summarize_fallback(chunk: str, max_sentences: int = 2) -> str:
    """Lead-2 extractive summary — deterministic."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk) if s.strip()]
    return " ".join(sents[:max_sentences]) if sents else chunk[:200]


# ---------------------------------------------------------------------------
# LLM-guided extraction (used only when a local model is loaded)
# ---------------------------------------------------------------------------
_LLM_EXTRACT_PROMPT = """Extract structured information from the text below. \
Respond with ONLY a JSON object with keys:
"concepts": up to 5 key concepts (short phrases),
"entities": up to 5 named entities,
"relations": list of {{"subject", "relation", "object"}} triples (max 5),
"summary": one-sentence summary.

Text:
{text}

JSON:"""


def extract_with_llm(chunk: str, llm) -> Optional[Dict[str, Any]]:
    """Returns dict or None on any failure (caller falls back)."""
    if llm is None or not getattr(llm, "available", False):
        return None
    raw = llm.complete(_LLM_EXTRACT_PROMPT.format(text=chunk[:2000]),
                       max_tokens=400, temperature=0.1, stop=["\n\n"])
    if not raw:
        return None
    try:
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else None
    except Exception:
        data = None
    if not isinstance(data, dict):
        return None
    # shape validation — never trust model output blindly
    out: Dict[str, Any] = {}
    if isinstance(data.get("concepts"), list):
        out["concepts"] = [str(c)[:80] for c in data["concepts"]][:5]
    if isinstance(data.get("entities"), list):
        out["entities"] = [str(e)[:80] for e in data["entities"]][:5]
    rels = []
    for r in data.get("relations", []) if isinstance(data.get("relations"), list) else []:
        if isinstance(r, dict) and {"subject", "relation", "object"} <= set(r):
            rels.append({"subject": str(r["subject"])[:80],
                         "relation": str(r["relation"])[:40],
                         "object": str(r["object"])[:80]})
        if len(rels) >= 5:
            break
    out["relations"] = rels
    if isinstance(data.get("summary"), str):
        out["summary"] = data["summary"][:400]
    return out or None
