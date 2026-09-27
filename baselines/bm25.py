"""Baseline: BM25 lexical ranking (Okapi). Pure Python — no new deps.

Lexical control for the vector baselines: if MIRA's advantage were just
"better matching", a classic lexical ranker should close much of the gap.
Corpus = concept + summary + raw_text per node — the same text the embedder
indexed — so the comparison is apples-to-apples.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List

from core.memory import MemoryFrame, MemoryNode

_TOKEN = re.compile(r"\w+", re.UNICODE)  # unicode-aware: Devanagari et al.


def _tokens(text: str) -> List[str]:
    return _TOKEN.findall((text or "").casefold())


@dataclass
class BaselineResult:
    query: str
    items: List[MemoryNode] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    latency_ms: float = 0.0
    system: str = "bm25"


class BM25RAG:
    """Okapi BM25 (k1=1.5, b=0.75) over the same node texts the embedder saw."""
    system = "bm25"

    def __init__(self, frame: MemoryFrame, k1: float = 1.5, b: float = 0.75):
        self.frame = frame
        self.k1, self.b = k1, b
        t0 = time.perf_counter()
        self._ids: List[str] = []
        self._docs: List[List[str]] = []
        self._index: Dict[str, MemoryNode] = {}
        self._postings: Dict[str, List] = {}
        for n in frame.nodes.values():
            text = " ".join(x for x in (n.concept, n.summary, n.raw_text) if x)
            toks = _tokens(text)
            i = len(self._ids)
            self._ids.append(n.id)
            self._docs.append(toks)
            self._index[n.id] = n
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            for t, f in tf.items():
                self._postings.setdefault(t, []).append((i, f))
        self.N = len(self._docs)
        self._avgdl = (sum(len(d) for d in self._docs) / self.N) if self.N else 0.0
        self._idf = {
            t: math.log(1.0 + (self.N - len(pl) + 0.5) / (len(pl) + 0.5))
            for t, pl in self._postings.items()
        }
        self.build_ms = round((time.perf_counter() - t0) * 1000, 1)

    def retrieve(self, query: str, k: int = 8) -> BaselineResult:
        t0 = time.perf_counter()
        qtoks = _tokens(query)
        res = BaselineResult(query=query)
        if not self.N or not qtoks or self._avgdl == 0:
            res.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            return res
        scores: Dict[int, float] = {}
        for t in qtoks:
            idf = self._idf.get(t)
            if idf is None:
                continue
            for i, f in self._postings.get(t, ()):
                denom = f + self.k1 * (1 - self.b + self.b
                                       * len(self._docs[i]) / self._avgdl)
                scores[i] = scores.get(i, 0.0) + idf * f * (self.k1 + 1) / denom
        order = sorted(scores.items(), key=lambda kv: (-kv[1], self._ids[kv[0]]))
        for i, s in order[:k]:
            res.items.append(self._index[self._ids[i]])
            res.scores.append(float(s))
        res.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return res
