"""Agent routing layer (the "real agent" capability).

One pipeline decides HOW to answer every query instead of always answering
from ingested memories only:

  1. MEMORY      — retrieval over the mandala scores well → grounded answer
                   with citations (spec §27). Still the priority path.
  2. LIVE WEB    — memory is weak/empty and the user consented → search the
                   web, ingest the top page, re-retrieve, answer with sources.
                   New knowledge becomes permanent mandala memories.
  3. PARAMETRIC  — no consent or no web results → the model answers from its
                   own knowledge, clearly labeled as ungrounded (never cited).

Honesty (spec §49): the answer's origin is always labeled (agent_mode), and
parametric answers never receive fabricated citations.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from core.answer import Answer, AnswerPipeline, parse_budget
from core.types import parse_float

logger = logging.getLogger("mira.agent")


class AgentPipeline:
    def __init__(self, pipe: AnswerPipeline, config: Dict[str, Any],
                 store=None, doc_titles=None):
        self._pipe = pipe
        self.cfg = config or {}
        self.store = store
        self.doc_titles = doc_titles or pipe.doc_titles

    # ------------------------------------------------------------------
    def _parametric(self, question: str) -> Answer:
        """Answer from the model's own knowledge — labeled, never cited."""
        llm = self._pipe.llm
        t0 = time.perf_counter()
        text = ""
        if llm is not None and llm.available:
            text = llm.chat(
                [{"role": "system", "content":
                    "You are a helpful assistant. Answer the question concisely "
                    "in 1-3 sentences."},
                 {"role": "user", "content": question}],
                max_tokens=320, temperature=0.2) or ""
            from core.answer import _degenerate
            if _degenerate(text):
                text = llm.chat(
                    [{"role": "user", "content": question}],
                    max_tokens=320, temperature=0.0) or ""
        if not text:
            text = ("No local LLM is loaded and no matching memories exist. "
                    "Ingest documents or enable web search to answer this.")
        return Answer(
            text=text, mode="llm",
            agent_mode="parametric",
            confidence_note="ungrounded: answered from model knowledge only — "
                            "not verified against memory or the web",
            metrics={"latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                     "n_memories": 0, "context_tokens": 0},
        )

    def _web_answer(self, question: str, weak: Optional[Answer]) -> Optional[Answer]:
        """Search the web, ingest the top page, re-retrieve. None if the web
        path yields nothing better than the weak memory answer."""
        from tools.websearch import fetch_page_text, search
        out = search(question, self.cfg, allow_web=True)
        results = out.get("results") or []
        if not results:
            logger.info("web path: no results (%s)", out.get("note", ""))
            return None
        ingested = 0
        for r in results[:2]:  # ingest top-2 pages, budgeted
            url = r.get("url") or ""
            if not url.startswith(("http://", "https://")):
                continue
            try:
                text = fetch_page_text(url, max_chars=12000)
                if len(text) < 200:
                    continue
                if self._pipe.workspace_ingest is not None:
                    self._pipe.workspace_ingest(text, title=r.get("title") or url)
                    ingested += 1
            except Exception as exc:
                logger.warning("web ingest failed for %s: %s", url, exc)
        if not ingested:
            return None
        fresh = self._pipe.ask(question)  # re-retrieve over the grown memory
        fresh.agent_mode = "web"
        fresh.confidence_note = (
            f"grounded in {ingested} page(s) fetched live from the web and "
            "stored as mandala memories")
        return fresh

    # ------------------------------------------------------------------
    def ask(self, question: str, active_components=None,
            allow_web: Optional[bool] = None) -> Answer:
        t0 = time.perf_counter()
        ans = self._pipe.ask(question, active_components=active_components)

        m = ans.metrics or {}
        top_sem = max((mm.get("components", {}).get("semantic", 0.0)
                       for mm in ans.memories), default=0.0)
        # absolute evidence floor: some real context behind the answer
        # (a ratio would starve small corpora — 50 solid tokens is 2.5% of 2048).
        # 12 tokens ≈ one real sentence; label-only fragments are already
        # impossible (compression skips them).
        enough_ctx = (m.get("context_tokens") or 0) >= 12
        gate = parse_float((self.cfg.get("agent", {}) or {}).get(
            "memory_gate_semantic", 0.45), 0.45)

        strong = (ans.mode in ("llm", "extractive") and top_sem >= gate
                  and enough_ctx)
        if strong:
            ans.agent_mode = "memory"
            return ans

        # weak memory support: escalate (web if consented, else parametric)
        web_ok = allow_web if allow_web is not None else \
            bool((self.cfg.get("web_search", {}) or {}).get("enabled"))
        if web_ok:
            fresh = self._web_answer(question, ans)
            # accept any fresh answer with real evidence — it is grounded in
            # pages we just fetched and stored (mode != no_evidence)
            if fresh is not None and fresh.mode in ("llm", "extractive"):
                fresh.metrics["latency_ms"] = round(
                    (time.perf_counter() - t0) * 1000, 1)
                return fresh
        para = self._parametric(question)
        para.metrics["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return para
