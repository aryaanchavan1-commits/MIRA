"""Baseline C: hierarchical retrieval (spec §28).

Query matches a coarse concept (ring 0-1) → descend to its children and
return leaf evidence. No radial or graph components.
"""
from __future__ import annotations

import re
import time
from typing import List

import numpy as np

from baselines.vector_rag import BaselineResult
from core.memory import MemoryFrame


class HierarchicalRAG:
    system = "hierarchical_rag"

    def __init__(self, frame: MemoryFrame, vector_store=None):
        self.frame = frame
        self._index = {n.id: n for n in frame.nodes.values()}

    def retrieve(self, query_vec: np.ndarray, k: int = 8) -> BaselineResult:
        t0 = time.perf_counter()
        qtokens = set(re.findall(r"[a-z0-9]{3,}", (query_vec if isinstance(query_vec, str)
                                                   else "").lower()))
        # coarse level: ring 0-1 concepts, token overlap
        scored = []
        for n in self.frame.nodes.values():
            if n.ring in (0, 1):
                toks = set(re.findall(r"[a-z0-9]{3,}",
                                      (n.concept + " " + n.summary).lower()))
                if qtokens & toks:
                    scored.append((n.id, len(qtokens & toks)))
        scored.sort(key=lambda kv: -kv[1])
        out_ids: List[str] = []
        for pid, _ in scored[:3]:
            out_ids.append(pid)
            parent = self._index[pid]
            for cid in parent.children[:k]:
                if cid in self._index:
                    out_ids.append(cid)
                    for gcid in self._index[cid].children[:2]:
                        if gcid in self._index:
                            out_ids.append(gcid)
        # fall back to embedding on low token overlap handled by caller (vector)
        res = BaselineResult(query="", system=self.system)
        seen = set()
        for nid in out_ids[:k]:
            if nid in seen:
                continue
            seen.add(nid)
            res.items.append(self._index[nid])
            res.scores.append(1.0 / (1 + len(res.items)))
        res.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return res
