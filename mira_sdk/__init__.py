"""MIRA SDK — inspectable long-term memory for AI agents.

A thin, honest facade over ``core.workspace.Workspace`` (the same engine the
FastAPI console drives), bootstrapped exactly like ``server.py`` via
``config.auto_config.build_context``. Thread-safe: every call takes the
workspace lock.

    from mira_sdk import MemoryEngine

    engine = MemoryEngine()               # uses the repo's config
    engine.add("The Eiffel Tower is in Paris.", title="facts")
    hits = engine.retrieve("Where is the Eiffel Tower?")
    print(hits["results"][0]["explanation"])

What it deliberately is NOT: a cloud client, a training loop, or a promise of
"forever memory". Memory is persistent *and versioned*; deletion is logical;
explanations are the retrieval components themselves.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__ = ["MemoryEngine", "__version__"]
__version__ = "0.1.0"


class MemoryEngine:
    """Local-first memory engine. All methods are thread-safe."""

    def __init__(self, config_path: Optional[str] = None,
                 llm: bool = False, load_llm: bool = True) -> None:
        """Bootstrap exactly like the server. ``llm=False`` (default) keeps
        the engine retrieval-only — no model is loaded unless you ask().
        ``load_llm=False`` skips the local GGUF load entirely (retrieval-only
        callers on RAM-tight machines)."""
        from config.auto_config import build_context
        self._ctx = build_context(config_path, force_llm_reload=llm,
                                  load_llm=load_llm)
        from core.workspace import Workspace
        self._ws = Workspace(embeddings=self._ctx.embeddings,
                             llm=self._ctx.llm, config=self._ctx.cfg)

    # -- write path -----------------------------------------------------
    def add(self, text: str, title: str = "sdk", source: str = "(sdk)",
            ) -> Dict[str, Any]:
        """Ingest text; returns ingestion stats (nodes, edges, docs)."""
        return self._ws.ingest_text(text, title=title, source_path=source)

    # -- read path --------------------------------------------------------
    def retrieve(self, question: str, k: Optional[int] = None,
                 components: Optional[List[str]] = None) -> Dict[str, Any]:
        """Retrieve with per-component explanations. No LLM is invoked."""
        res = self._ws.answer_pipeline.retriever.retrieve(
            question, self._ctx.embeddings.encode([question])[0],
            active_components=",".join(components) if components else None,
            final_k=k)
        doc_sources = self._ws._doc_sources()
        return {
            "query": question,
            "n_candidates": res.n_candidates,
            "latency_ms": res.latency_ms,
            "results": [{
                "id": it.node.id,
                "concept": it.node.concept,
                "text": it.node.summary or it.node.raw_text,
                "score": round(it.score, 6),
                "explanation": {c: round(v, 6)
                                for c, v in it.components.items()},
                "path": it.path,
                "source": doc_sources.get((it.node.source_ids or [""])[0],
                                          it.node.source_ids),
            } for it in res.items],
        }

    def explain(self, question: str, k: Optional[int] = None) -> Dict[str, Any]:
        """Alias of retrieve() — explicit for auditability workflows."""
        return self.retrieve(question, k=k)

    def audit(self, question: str, k: Optional[int] = None) -> Dict[str, Any]:
        """Full audit record for one retrieval (S6 in paper/steps.md).

        Everything a reviewer needs to justify this answer path: the scoring
        configuration actually in effect, per-component scores and evidence
        paths for every hit, document sources, and the state of the version
        store (chain count, record count). One call, exportable JSON."""
        res = self.retrieve(question, k=k)
        chain = getattr(self._ws, "version_chain", None)
        rc = self._ws.config.get("retrieval_score", {}) if isinstance(
            self._ws.config, dict) else {}
        return {
            **res,
            "audit": {
                "at": self._now(),
                "weights_in_effect": {c: round(w, 6)
                                      for c, w in
                                      self._ws.answer_pipeline.retriever.weights.items()},
                "retrieval_config": {kk: self._ws.answer_pipeline.retriever.__dict__.get(kk)
                                     for kk in ("candidate_k", "final_k", "max_hops")},
                "store": self._ws.stats(),
                "version_store": (None if chain is None else {
                    "subjects": len(chain.versions),
                    "records": len(chain),
                    "append_only": True}),
            },
        }

    @staticmethod
    def _now() -> str:
        from core.types import iso_now
        return iso_now()

    def ask(self, question: str, allow_web: Optional[bool] = None) -> Dict[str, Any]:
        """Full answer pipeline (loads the local LLM on first call).
        Answers are grounded in retrieved evidence or explicitly labeled."""
        ans = self._ws.ask(question, allow_web=allow_web)
        return {
            "answer": ans.text,
            "mode": ans.mode,                    # llm | extractive | no_evidence
            "agent_mode": ans.agent_mode,        # memory | web | parametric
            "sources": ans.sources,
            "memories": ans.memories,
            "note": ans.confidence_note,
        }

    # -- versioning ------------------------------------------------------
    def remember(self, subject: str, content: str, source: str = "sdk",
                 provenance: Optional[List[str]] = None) -> Dict[str, Any]:
        """Append a version for a subject (never overwrites)."""
        chain = self._ensure_chain()
        return chain.update(subject, content, source=source,
                            provenance=provenance)

    def history(self, subject: str) -> List[Dict[str, Any]]:
        """Immutable version history for a subject ([] if never versioned)."""
        chain = self._ensure_chain()
        return chain.history(subject)

    def rollback(self, subject: str, version: int) -> Dict[str, Any]:
        """Restore version n as the new head — append-only."""
        chain = self._ensure_chain()
        rec = chain.rollback(subject, version)
        if rec is None:
            raise ValueError(f"no version {version} for {subject!r}")
        return rec

    def _ensure_chain(self):
        chain = getattr(self._ws, "version_chain", None)
        if chain is None:
            from core.ncm import VersionChain
            cfg = self._ws.config.get("ncm", {}) if isinstance(
                self._ws.config, dict) else {}
            chain = VersionChain(max_depth=cfg.get("max_depth", 4),
                                 store=self._ws.store)  # persists across restarts
            self._ws.version_chain = chain
        return chain

    # -- maintenance -------------------------------------------------------
    def consolidate(self, days: int = 30, apply: bool = False) -> Dict[str, Any]:
        """Sleep-consolidation pass (decay + replay + gists). Probe mode
        unless apply=True; the same pass the /api/memory/consolidate drives."""
        from scripts.consolidate import replay_paths
        return replay_paths(self._ws, days, apply=apply)

    def stats(self) -> Dict[str, Any]:
        return self._ws.stats()

    def close(self) -> None:
        self._ws.close()
