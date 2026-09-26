"""Baseline C: hierarchical retrieval (spec §28).

Query matches a coarse concept (ring 0-1) → descend to its children and
return leaf evidence. No radial or graph components.
"""
from __future__ import annotations

import re
import time
from typing import List

from baselines.vector_rag import BaselineResult
from core.memory import MemoryFrame


class HierarchicalRAG:
    system = "hierarchical_rag"

    def __init__(self, frame: MemoryFrame, vector_store=None):
        self.frame = frame
        self._index = {n.id: n for n in frame.nodes.values()}

    def retrieve(self, query: str, k: int = 8) -> BaselineResult:
        t0 = time.perf_counter()
        qtokens = set(re.findall(r"[a-z0-9]{3,}", (query or "").lower()))
        scored = []
        for n in self.frame.nodes.values():
            if n.ring in (0, 1):
                toks = set(re.findall(r"[a-z0-9]{3,}",
                                      (n.concept + " " + n.summary).lower()))
                overlap = qtokens & toks
                if overlap:
                    scored.append((n.id, len(overlap)))
        scored.sort(key=lambda item: (-item[1], item[0]))
        out_ids: List[str] = []
        for pid, _ in scored[:3]:
            out_ids.append(pid)
            children = sorted(self.frame.children_of(pid))
            for cid in children[:k]:
                if cid in self._index:
                    out_ids.append(cid)
                    grandchildren = sorted(self.frame.children_of(cid))
                    out_ids.extend(gcid for gcid in grandchildren[:2]
                                   if gcid in self._index)
        res = BaselineResult(query=query, system=self.system)
        seen = set()
        for nid in out_ids:
            if nid in seen or nid not in self._index:
                continue
            seen.add(nid)
            res.items.append(self._index[nid])
            res.scores.append(1.0 / (1 + len(res.items)))
            if len(res.items) >= k:
                break
        res.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return res
