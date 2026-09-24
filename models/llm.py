"""LLM backend (spec §24).

llama.cpp (llama-cpp-python) over GGUF. Load strategy is defensive:
try requested GPU offload → retry CPU-only → mark unavailable. A missing
LLM never crashes the app; callers get `available=False` and use the
deterministic extractive fallback (marked as such in the UI).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("mira.llm")

try:
    from llama_cpp import Llama
    _LLAMA_IMPORT_OK = True
except Exception as _exc:
    Llama = None
    _LLAMA_IMPORT_OK = False
    _IMPORT_ERROR = _exc


class LLMBackend:
    def __init__(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = 0,
                 n_threads: int = 4, n_batch: int = 128, verbose: bool = False):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.requested_gpu_layers = n_gpu_layers
        self.n_gpu_layers_used = 0
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.available = False
        self.load_error = ""
        self._llm: Any = None
        # llama_cpp.Llama is NOT thread-safe: concurrent calls corrupt decoder
        # state and crash the whole process with a GGML assert (repack.cpp).
        # The server runs sync handlers in a threadpool, so every generation
        # must serialize on this lock.
        self._lock = threading.Lock()
        self._load(verbose)

    def _load(self, verbose: bool) -> None:
        if not _LLAMA_IMPORT_OK:
            self.load_error = f"llama-cpp-python not importable: {_IMPORT_ERROR}"
            logger.warning(self.load_error)
            return
        if not os.path.exists(self.model_path):
            self.load_error = f"model file missing: {self.model_path}"
            logger.warning(self.load_error)
            return
        # Ladder: requested GPU offload → CPU-only → smaller contexts.
        # Windows commit-limit failures (os error 1455) shrink with the KV cache,
        # so a smaller n_ctx succeeds where the full one cannot.
        attempts: List[tuple] = []
        if self.requested_gpu_layers != 0:
            attempts.append((self.requested_gpu_layers, self.n_ctx))
        attempts.append((0, self.n_ctx))
        for smaller_ctx in (1536, 1024, 768):
            if smaller_ctx < self.n_ctx:
                attempts.append((0, smaller_ctx))
        for gpu_layers, n_ctx in attempts:
            try:
                t0 = time.perf_counter()
                self._llm = Llama(
                    model_path=self.model_path,
                    n_ctx=n_ctx,
                    n_gpu_layers=gpu_layers,
                    n_threads=self.n_threads,
                    n_batch=64,
                    n_ubatch=64,  # small physical batch avoids the Q4_K_M CPU
                                  # repack buffer assert (repack.cpp:4238)
                    verbose=verbose,
                )
                self.n_gpu_layers_used = gpu_layers
                self.n_ctx = n_ctx
                self.available = True
                logger.info("llm loaded", extra={
                    "path": os.path.basename(self.model_path),
                    "gpu_layers": gpu_layers, "ctx": n_ctx,
                    "seconds": round(time.perf_counter() - t0, 1),
                })
                return
            except Exception as exc:
                self.load_error = str(exc)
                logger.warning("llm load failed with gpu_layers=%s ctx=%s: %s",
                               gpu_layers, n_ctx, exc)
                self._llm = None
        self.available = False

    # ---- generation ----
    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 512,
             temperature: float = 0.2, stop: Optional[List[str]] = None) -> str:
        """messages: [{role, content}]. Returns "" when unavailable."""
        if not self.available:
            return ""
        with self._lock:  # llama context is stateful — serialize access
            try:
                out = self._llm.create_chat_completion(
                    messages=messages, max_tokens=max_tokens,
                    temperature=temperature, stop=stop or None,
                )
                text = out["choices"][0]["message"]["content"]
                return (text or "").strip()
            except Exception as exc:
                logger.warning("chat failed: %s", exc)
                return ""

    def complete(self, prompt: str, max_tokens: int = 256,
                 temperature: float = 0.2, stop: Optional[List[str]] = None) -> str:
        if not self.available:
            return ""
        with self._lock:  # llama context is stateful — serialize access
            try:
                out = self._llm(prompt, max_tokens=max_tokens, temperature=temperature,
                                stop=stop or None)
                return (out["choices"][0]["text"] or "").strip()
            except Exception as exc:
                logger.warning("complete failed: %s", exc)
                return ""

    def info(self) -> Dict[str, Any]:
        return {
            "model": os.path.basename(self.model_path) if self.model_path else "",
            "available": self.available,
            "gpu_layers": self.n_gpu_layers_used,
            "ctx": self.n_ctx,
            "threads": self.n_threads,
            "error": self.load_error,
        }


_ANSWER_TEMPLATE = """Answer the question using ONLY the provided memory excerpts. \
If the excerpts do not contain the answer, say exactly: "The memories do not contain enough information."

Each excerpt is tagged with its position in the memory mandala:
(ringN · sector) — ring 0 = core concepts, ring 4 = raw source documents;
sector = the topic region. Prefer core-ring evidence for definitions and
outer-ring evidence for specific details.

Memory excerpts:
{context}

Question: {question}

Answer (concise, cite excerpt numbers like [1] when used):"""


def answer_prompt(context: str, question: str) -> str:
    return _ANSWER_TEMPLATE.format(context=context, question=question)


def extractive_answer(question: str, context: str, units: List[Dict]) -> Dict[str, Any]:
    """Deterministic no-LLM fallback: top evidence sentences as the answer.

    Clearly labeled extractive — never presented as generative output.
    """
    sentences: List[str] = []
    for line in context.splitlines():
        line = line.strip()
        if line:
            sentences.append(line)
    top = " ".join(sentences[:3]) if sentences else \
        "The memories do not contain enough information."
    used = [u.get("node_id") for u in units[:3]]
    return {"answer": top, "mode": "extractive", "used_nodes": used}
