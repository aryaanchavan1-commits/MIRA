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
import re
import time
from threading import RLock
from typing import Any, Dict, Optional

from core.affect import AffectiveState
from core.answer import (
    Answer,
    AnswerPipeline,
    _answer_metrics,
    _completion_budget,
    _degenerate,
    _is_no_evidence_text,
)
from core.types import parse_float

logger = logging.getLogger("mira.agent")

_IDENTITY_QUERY = re.compile(
    r"(?:\bwho\s+(?:created|made|built|developed)\s+(?:you|this\s+(?:ai|system|app)|(?:the\s+)?mira)(?![-'’\w])"
    r"|\bmira\s+(?:was\s+)?(?:created|made|built|developed)\s+by\s+who(?:m)?\b)",
    re.IGNORECASE,
)
_IDENTITY_TEXT = (
    "MIRA was made by Aryan Chavan; it is a local-first Bio-NN-inspired, "
    "mandala-based symbolic memory and retrieval research system, not a "
    "biological brain simulation."
)
_IDENTITY_CONFIDENCE = (
    "high confidence: deterministic built-in project identity; not retrieved "
    "from memory or the web, so no sources were consulted or cited."
)


def _synchronized(method):
    def locked(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    locked.__name__ = getattr(method, "__name__", "locked")
    locked.__doc__ = method.__doc__
    locked.__wrapped__ = method
    return locked


class AgentPipeline:
    def __init__(self, pipe: AnswerPipeline, config: Dict[str, Any],
                 store=None, doc_titles=None, affect_state=None):
        self._pipe = pipe
        self._lock = getattr(pipe, "_lock", None) or RLock()
        self.cfg = config if isinstance(config, dict) else {}
        self.store = store
        self.doc_titles = doc_titles or pipe.doc_titles
        # Workspace passes its single state here.  Standalone agents get a
        # local fallback so their Answers still carry a useful route snapshot.
        self.affect_state = affect_state or AffectiveState()
        self.affect = self.affect_state

    def _finish(self, answer: Answer) -> Answer:
        """Record the final route, then freeze that state for this Answer."""
        try:
            answer.affect_snapshot = self.affect_state.update_from_answer(answer)
        except Exception as exc:  # affect is advisory; never break answering
            logger.warning("affect update failed: %s", exc)
        return answer

    def _reinforce_grounded(self) -> None:
        callback = getattr(self._pipe, "reinforce_last_result", None)
        if callable(callback):
            callback()

    @staticmethod
    def _evidence_memories(answer: Answer):
        """Return only records whose excerpts reached the answer stage."""
        memories = list(getattr(answer, "memories", None) or [])
        if hasattr(answer, "selected_evidence_ids"):
            selected = getattr(answer, "selected_evidence_ids", None) or []
            metrics = getattr(answer, "metrics", None) or {}
            explicit = bool(getattr(answer, "_selection_explicit", False))
            if explicit or selected or "n_selected_evidence" in metrics:
                allowed = set(selected)
                return [memory for memory in memories
                        if isinstance(memory, dict) and memory.get("id") in allowed]
            # A hand-built legacy Answer has no evidence metric; retain its
            # historical interpretation for compatibility.
        return memories

    # ------------------------------------------------------------------
    @staticmethod
    def _identity_answer(question: str) -> Optional[Answer]:
        """Return the canonical local project identity, without retrieval."""
        if not isinstance(question, str) or not _IDENTITY_QUERY.search(question):
            return None
        return Answer(
            text=_IDENTITY_TEXT,
            mode="deterministic",
            agent_mode="identity",
            confidence_note=_IDENTITY_CONFIDENCE,
            metrics=_answer_metrics(
                llm_mode="deterministic",
            ),
            selected_evidence_ids=[],
        )

    def _parametric(self, question: str) -> Answer:
        """Answer from the model's own knowledge — labeled, never cited."""
        llm = self._pipe.llm
        t0 = time.perf_counter()
        text = ""
        if llm is not None and getattr(llm, "available", False):
            messages = [
                {"role": "system", "content":
                    "You are a helpful assistant. Answer the question concisely "
                    "in 1-3 sentences."},
                {"role": "user", "content": question},
            ]
            budget = _completion_budget(llm, messages, 320, self.cfg)
            if budget:
                text = llm.chat(messages, max_tokens=budget, temperature=0.2) or ""
            if _degenerate(text):
                retry = [{"role": "user", "content": question}]
                budget = _completion_budget(llm, retry, 320, self.cfg)
                if budget:
                    text = llm.chat(retry, max_tokens=budget, temperature=0.0) or ""
            if _degenerate(text) or _is_no_evidence_text(text):
                text = ""
        llm_generated = bool(text)
        if not text:
            text = ("No local LLM is loaded and no matching memories exist. "
                    "Ingest documents or enable web search to answer this.")
        return Answer(
            text=text, mode="llm" if llm_generated else "no_evidence",
            agent_mode="parametric",
            confidence_note=("ungrounded: answered from model knowledge only — "
                             "not verified against memory or the web"
                             if llm_generated else
                             "ungrounded: no local LLM produced a parametric answer"),
            metrics=_answer_metrics(
                latency_ms=(time.perf_counter() - t0) * 1000,
                llm_mode="parametric" if llm_generated else "no_evidence",
            ),
            selected_evidence_ids=[],
        )

    def _web_answer(self, question: str, weak: Optional[Answer],
                    active_components=None) -> Optional[Answer]:
        """Search the web, ingest the top page, re-retrieve. None if the web
        path yields nothing better than the weak memory answer."""
        from tools.websearch import fetch_page_text, search, web_allowed
        if not web_allowed(self.cfg, allow_web=True):
            return None
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
                text = fetch_page_text(
                    url, max_chars=12000, config=self.cfg, allow_web=True)
                if len(text) < 200:
                    continue
                if self._pipe.workspace_ingest is not None:
                    self._pipe.workspace_ingest(
                        text, title=r.get("title") or url, source_path=url)
                    ingested += 1
            except Exception as exc:
                logger.warning("web ingest failed for %s: %s", url, exc)
        if not ingested:
            return None
        # Preserve the caller's ablation set across the web round-trip.  A
        # re-retrieval with the default full stack would silently turn an
        # ablation run into a different experiment.
        fresh = self._pipe.ask(question, active_components=active_components)
        fresh.agent_mode = "web"
        fresh.confidence_note = (
            f"grounded in {ingested} page(s) fetched live from the web and "
            "stored as mandala memories")
        return fresh

    # ------------------------------------------------------------------
    @_synchronized
    def ask(self, question: str, active_components=None,
            allow_web: Optional[bool] = None) -> Answer:
        # Validate consent before doing any work, even when the query later
        # takes the identity or strong-memory route.
        from tools.websearch import web_allowed
        web_ok = web_allowed(self.cfg, allow_web)

        identity = self._identity_answer(question)
        if identity is not None:
            return self._finish(identity)

        t0 = time.perf_counter()
        ans = self._pipe.ask(question, active_components=active_components)

        m = ans.metrics or {}
        evidence = self._evidence_memories(ans)
        top_sem = max((mm.get("components", {}).get("semantic", 0.0)
                       for mm in evidence), default=0.0)
        # Absolute evidence floor: some real context behind the answer.  The
        # selected-evidence list is authoritative; a high-scoring candidate
        # dropped by compression cannot make a weak answer look grounded.
        context_text = str(getattr(ans, "context_text", "") or "")
        enough_ctx = (
            bool(evidence)
            and (m.get("context_tokens") or 0) >= 12
            and (not context_text or bool(context_text.strip()))
        )
        gate = parse_float((self.cfg.get("agent", {}) or {}).get(
            "memory_gate_semantic", 0.45), 0.45)

        strong = (ans.mode in ("llm", "extractive") and top_sem >= gate
                  and enough_ctx)
        if strong:
            ans.agent_mode = "memory"
            self._reinforce_grounded()
            return self._finish(ans)

        # Weak memory support: escalate only through the validated consent
        # gate, otherwise return an explicitly ungrounded parametric answer.
        if web_ok:
            fresh = self._web_answer(question, ans, active_components=active_components)
            # Accept only a fresh answer with selected evidence, never merely a
            # generated string or a candidate excluded during compression.
            fresh_evidence = self._evidence_memories(fresh) if fresh is not None else []
            if (fresh is not None and fresh.mode in ("llm", "extractive")
                    and fresh_evidence
                    and (fresh.metrics.get("context_tokens") or 0) >= 12):
                self._reinforce_grounded()
                fresh.metrics["latency_ms"] = round(
                    (time.perf_counter() - t0) * 1000, 1)
                return self._finish(fresh)
        para = self._parametric(question)
        para.metrics["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return self._finish(para)
