"""Chunking (spec §16).

Recursive character splitter with overlap. Targets ~chunk_size chars but
never splits mid-sentence when avoidable; keeps page numbers for provenance.
"""
from __future__ import annotations

import re
from typing import List, Tuple

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    parts = _SENT_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    if not text or not text.strip():
        return []
    if len(text) <= chunk_size:
        return [text.strip()]
    sentences = _split_sentences(text)
    chunks: List[str] = []
    current: List[str] = []
    size = 0
    for sent in sentences:
        slen = len(sent)
        if size + slen > chunk_size and current:
            chunks.append(" ".join(current).strip())
            # start next chunk with tail overlap
            tail: List[str] = []
            tsize = 0
            for s in reversed(current):
                if tsize + len(s) > overlap:
                    break
                tail.insert(0, s)
                tsize += len(s)
            current = tail
            size = tsize
        # single overlong sentence → hard split
        if slen > chunk_size:
            if current:
                chunks.append(" ".join(current).strip())
                current, size = [], 0
            for i in range(0, slen, chunk_size - overlap):
                piece = sent[i:i + chunk_size].strip()
                if piece:
                    chunks.append(piece)
            continue
        current.append(sent)
        size += slen
    if current:
        chunks.append(" ".join(current).strip())
    return [c for c in chunks if c]


def chunk_pages(pages: List[Tuple[int, str]], chunk_size: int = 800,
                overlap: int = 120) -> List[Tuple[int, str]]:
    """pages: [(page_no, text)] → [(page_no, chunk)]."""
    out: List[Tuple[int, str]] = []
    for page_no, text in pages:
        for chunk in chunk_text(text, chunk_size, overlap):
            out.append((page_no, chunk))
    return out
