"""Training dataset builder (spec §36).

Generates {query, memory_context, retrieval_path, evidence, answer} samples
from the live memory system. Every sample is grounded: the answer text comes
from a real ingested chunk (recorded as evidence), and memory_context is the
actual MIRA retrieval output for the query. Queries are template-generated
from node concepts — the file is marked synthetic; no unsupported
hallucinations are used for training.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from core.workspace import Workspace

_TEMPLATES = [
    "What is {c}?",
    "What does the material say about {c}?",
    "Summarize: {c}.",
]


def build_samples(ws: Workspace, limit: int = 200) -> List[Dict[str, Any]]:
    from core.answer import AnswerPipeline
    pipe = AnswerPipeline(ws.frame, ws.vs, ws.gs, ws.embeddings, llm=None,
                          config=ws.config, doc_titles=ws._doc_titles())
    samples: List[Dict[str, Any]] = []
    for node in ws.frame.nodes.values():
        if len(samples) >= limit:
            break
        if not node.raw_text or not node.source_ids:
            continue  # answers must trace to real evidence
        q = _TEMPLATES[len(samples) % len(_TEMPLATES)].format(c=node.concept)
        ans = pipe.ask(q, system_name="dataset_builder")
        if ans.mode == "no_evidence" or not ans.text.strip():
            continue
        samples.append({
            "query": q,
            "memory_context": ans.memories and json.dumps(ans.memories, ensure_ascii=False) or "",
            "retrieval_path": ans.path_labels,
            "evidence": node.raw_text[:600],
            "answer": ans.text[:600],
            "source_ids": node.source_ids[:3],
            "synthetic": True,
        })
    return samples


def save(samples: List[Dict[str, Any]], path: str) -> int:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(samples, fh, ensure_ascii=False, indent=2)
    return len(samples)
