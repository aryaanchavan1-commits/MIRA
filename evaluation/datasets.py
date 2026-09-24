"""Benchmark dataset adapters (spec §30).

Loads standard multi-hop QA datasets from LOCAL files (never auto-downloaded):
- HotpotQA (distractor/train JSON)
- 2WikiMultiHopQA (HotpotQA-like JSON)
- MuSiQue (JSON with supporting_paragraphs)
- custom MIRA format [{question, answer, supporting_ids?}]

When a live Workspace is provided, supporting-evidence titles are resolved to
actual memory node ids, enabling retrieval recall/MRR. Without it, records
still work for answer-level scoring (token-F1 / judge) but not retrieval metrics.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("mira.datasets")

Record = Dict[str, Any]


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_titles_to_ids(ws, titles: List[str]) -> List[str]:
    """Map document/evidence titles to node ids via the workspace.

    Strategy: find documents by title → their chunk ids → nodes sourced to
    those chunks. Falls back to empty (metrics simply skip retrieval scores).
    """
    if ws is None or not titles:
        return []
    title_lower = {t.lower() for t in titles}
    doc_ids = []
    chunk_ids: set = set()
    for d in ws.store.list_documents():
        if d["title"].lower() in title_lower or os.path.basename(
                d.get("source_path") or "").lower() in title_lower:
            doc_ids.append(d["id"])
            chunk_ids |= {c["id"] for c in ws.store.document_chunks(d["id"])}
    if not chunk_ids:
        return []
    ids = [n.id for n in ws.frame.nodes.values() if set(n.source_ids) & chunk_ids]
    return ids


def load_custom(path: str, ws=None, limit: Optional[int] = None) -> List[Record]:
    data = _read_json(path)
    if isinstance(data, dict):  # allow {"records": [...]}
        data = data.get("records", [])
    out = []
    for d in data[:limit]:
        if d.get("question") and d.get("answer"):
            out.append({"question": d["question"], "answer": d["answer"],
                        "supporting_ids": list(d.get("supporting_ids") or [])})
    return out


def load_hotpotqa(path: str, ws=None, limit: Optional[int] = None) -> List[Record]:
    """HotpotQA: list of {question, answer, supporting_facts: {title: [...]}
    (train format) or [[title, sent_id], ...] (dev format)."""
    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of HotpotQA records")
    out: List[Record] = []
    for d in data:
        if limit and len(out) >= limit:
            break
        q, a = d.get("question"), d.get("answer")
        if not q or a is None:
            continue
        sf = d.get("supporting_facts") or {}
        if isinstance(sf, dict):          # train format
            titles = list(sf.get("title", []))
        elif isinstance(sf, list):        # dev format [[title, sent_id], ...]
            titles = [row[0] for row in sf if row]
        else:
            titles = []
        out.append({"question": q, "answer": a,
                    "supporting_ids": resolve_titles_to_ids(ws, titles)})
    return out


def load_2wiki(path: str, ws=None, limit: Optional[int] = None) -> List[Record]:
    """2WikiMultiHopQA follows HotpotQA's format."""
    return load_hotpotqa(path, ws, limit)


def load_musique(path: str, ws=None, limit: Optional[int] = None) -> List[Record]:
    """MuSiQue: [{question, answer, supporting_paragraphs: [{title, is_supporting}]}]"""
    data = _read_json(path)
    if isinstance(data, dict):
        data = data.get("data", [])
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of MuSiQue records")
    out: List[Record] = []
    for d in data:
        if limit and len(out) >= limit:
            break
        q, a = d.get("question"), d.get("answer")
        if not q or a is None:
            continue
        titles = [p.get("title", "") for p in (d.get("supporting_paragraphs") or [])
                  if p.get("is_supporting")]
        out.append({"question": q, "answer": a,
                    "supporting_ids": resolve_titles_to_ids(ws, titles)})
    return out


LOADERS = {
    "custom": load_custom,
    "hotpotqa": load_hotpotqa,
    "2wiki": load_2wiki,
    "musique": load_musique,
}


def corpus_dataset(ws, limit: Optional[int] = None) -> List[Record]:
    """Generate a QA dataset from the ingested corpus itself.

    Each answerable fact node (has raw_text + sources) contributes one record
    whose question is its concept; the gold answer is the node's evidence
    text. A self-supervised sanity benchmark: measures whether a system can
    find back what it stored — not open-domain QA. Use real datasets
    (HotpotQA, MuSiQue, custom) for headline claims.
    """
    records: List[Record] = []
    for n in ws.frame.nodes.values():
        if not n.source_ids or not (n.raw_text or n.summary):
            continue
        text = (n.raw_text or n.summary or "").strip()
        if len(text) < 25:  # skip fragments with no answerable content
            continue
        records.append({
            "question": n.concept or n.summary[:80],
            "answer": text,
            "supporting_ids": [n.id],
        })
        if limit and len(records) >= limit:
            break
    return records


def load_any(path: str, fmt: str, ws=None, limit: Optional[int] = None) -> List[Record]:
    if fmt == "corpus":
        if ws is None:
            raise ValueError("corpus dataset requires a workspace")
        return corpus_dataset(ws, limit=limit)
    if fmt not in LOADERS:
        raise ValueError(f"unknown dataset format {fmt!r}; choose from {sorted(LOADERS) + ['corpus']}")
    return LOADERS[fmt](path, ws=ws, limit=limit)
