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


_MEMORY_ERROR_MARKERS = (
    "unable to allocate", "failed to create llama_context",
    "failed to load model from file", "out of memory", "os error 1455",
    "bad_alloc", "cannot allocate",
)


def _explain(errors: List[str]) -> str:
    """Turn a raw llama.cpp failure into something the operator can act on.

    Every failure seen in the wild here is memory, not a broken model file, so
    name the real constraint and what to do about it instead of surfacing
    "Failed to load model from file" and silently degrading to extractive
    answers.
    """
    if not errors:
        return "no candidate model loaded"
    tail = "; ".join(errors[-2:])
    low = tail.lower()
    if not any(m in low for m in _MEMORY_ERROR_MARKERS):
        return tail
    headroom = ""
    try:
        from core.hardware import detect_ram
        ram = detect_ram()
        commit = ram.get("commit_available")
        if commit:
            headroom = (f" Windows commit headroom is {commit:.2f} GB "
                        f"(RAM available {ram.get('available', 0):.2f} GB) — "
                        "the commit limit, not physical RAM, is the binding "
                        "constraint. ")
    except Exception:  # noqa: BLE001 - reporting must never raise
        pass
    return (f"out of memory loading the model. " + headroom +
            "Close memory-heavy apps or raise the pagefile size, then restart; "
            "MIRA falls back to a smaller local GGUF automatically once there "
            f"is room. Last errors: {tail}")


class LLMBackend:
    # Bounded so a broken/oversized file cannot stall server startup.
    _LOAD_ATTEMPT_BUDGET = 12

    def __init__(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = 0,
                 n_threads: int = 4, n_batch: int = 128, verbose: bool = False):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.requested_gpu_layers = n_gpu_layers
        self.n_gpu_layers_used = 0
        self.n_threads = n_threads
        try:
            self.n_batch = max(1, int(n_batch))
        except (TypeError, ValueError):
            self.n_batch = 64
        self.available = False
        self.load_error = ""
        self._llm: Any = None
        # llama_cpp.Llama is NOT thread-safe: concurrent calls corrupt decoder
        # state and crash the whole process with a GGML assert (repack.cpp).
        # The server runs sync handlers in a threadpool, so every generation
        # must serialize on this lock.
        self._lock = threading.Lock()
        self._load(verbose)

    def _candidate_models(self) -> List[str]:
        """Requested model first, then smaller local GGUFs.

        Selection already budgets against the commit limit, but a box can
        still be tighter than the budget believed (another process commits
        between probe and load). Falling back across model files is what
        keeps the app answering instead of dropping to the extractive path.
        """
        paths: List[str] = []
        seen = set()

        def _add(path: str) -> None:
            # Compare normalized keys: the requested path arrives with forward
            # slashes while discovery returns backslashes, so a plain string
            # check let the same file be queued twice and burned the budget.
            key = os.path.normcase(os.path.abspath(path))
            if key not in seen:
                seen.add(key)
                paths.append(path)

        if self.model_path:
            _add(self.model_path)
        try:
            from models.model_manager import discover_local_gguf
            for m in sorted(discover_local_gguf(),
                            key=lambda m: os.path.getsize(m.path)):
                _add(m.path)
        except Exception as exc:  # noqa: BLE001 - discovery is best effort
            logger.warning("model discovery failed: %s", exc)
        return [p for p in paths if os.path.exists(p)]

    def _load(self, verbose: bool) -> None:
        if not _LLAMA_IMPORT_OK:
            self.load_error = f"llama-cpp-python not importable: {_IMPORT_ERROR}"
            logger.warning(self.load_error)
            return
        if not os.path.exists(self.model_path):
            self.load_error = f"model file missing: {self.model_path}"
            logger.warning(self.load_error)
            return
        # Ladder over (model file x gpu-offload x context):
        #   requested file: GPU offload -> CPU-only -> smaller contexts
        #   then the next smaller local GGUF, and so on.
        # Windows commit-limit failures (os error 1455) shrink with the KV
        # cache, so a smaller n_ctx succeeds where the full one cannot.
        #
        # A file that cannot be loaded at all is skipped after ONE attempt —
        # retrying it is wasted startup time — but the ladder still moves on
        # to a smaller model, which is what actually keeps a memory-starved
        # box answering instead of degrading to the extractive fallback.
        errors: List[str] = []
        gpu_opts = ([self.requested_gpu_layers] if self.requested_gpu_layers else []) + [0]
        ctx_opts = [self.n_ctx] + [c for c in (2048, 1536, 1024, 768, 512) if c < self.n_ctx]
        budget = self._LOAD_ATTEMPT_BUDGET
        for path in self._candidate_models():
            for gpu_layers in gpu_opts:
                for n_ctx in ctx_opts:
                    if len(errors) >= budget:
                        break
                    try:
                        t0 = time.perf_counter()
                        self._llm = Llama(
                            model_path=path,
                            n_ctx=n_ctx,
                            n_gpu_layers=gpu_layers,
                            n_threads=self.n_threads,
                            n_batch=self.n_batch,
                            n_ubatch=min(64, self.n_batch),  # keep physical batch bounded
                            verbose=verbose,
                        )
                        self.model_path = path
                        self.n_gpu_layers_used = gpu_layers
                        self.n_ctx = n_ctx
                        self.available = True
                        self.load_error = ""
                        if errors:
                            logger.info("llm loaded after %d failed attempt(s): %s",
                                        len(errors), os.path.basename(path))
                        logger.info("llm loaded", extra={
                            "path": os.path.basename(path),
                            "gpu_layers": gpu_layers, "ctx": n_ctx,
                            "seconds": round(time.perf_counter() - t0, 1),
                        })
                        return
                    except Exception as exc:
                        errors.append(f"{os.path.basename(path)} gpu={gpu_layers} "
                                      f"ctx={n_ctx}: {exc}")
                        logger.warning("llm load failed with %s gpu_layers=%s ctx=%s: %s",
                                       os.path.basename(path), gpu_layers, n_ctx, exc)
                        self._llm = None
                        # File-level failure: no context or offload change will
                        # rescue this file, so move to the next candidate.
                        if "failed to load model from file" in str(exc).lower():
                            break
                    if len(errors) >= budget:
                        break
                if len(errors) >= budget:
                    break
        self.available = False
        self.load_error = _explain(errors)
        logger.error("llm unavailable after %d attempt(s): %s",
                     len(errors), self.load_error)

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
