"""Embedding backend (spec §13/§38).

sentence-transformers when available; deterministic hashing fallback when the
model can't be loaded (offline first-run, no disk space). The fallback keeps
the whole pipeline runnable — it's a real, if weaker, embedding: char n-gram
hashing, normalized. Every record of which backend was used is kept in
metadata so experiments remain comparable.
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import List, Optional

import numpy as np

logger = logging.getLogger("mira.embeddings")

FALLBACK_DIM = 384


class EagerEmbeddings:
    """Proxies an EmbeddingBackend so the first call forces the load.

    The underlying backend loads lazily; without this, a thread that calls
    info() while another is mid-import sees "hashing" and the lineage guard
    re-embeds everything with the fallback. Wrapping in an eager proxy fixes
    the race without touching call sites.
    """

    def __init__(self, inner: "EmbeddingBackend"):
        self._inner = inner

    def _eager(self) -> "EmbeddingBackend":
        self._inner._load()
        return self._inner

    def info(self) -> dict:
        return self._eager().info()

    def encode(self, texts, batch_size: int = 32, show_progress: bool = False):
        return self._eager().encode(texts, batch_size=batch_size,
                                    show_progress=show_progress)

    def __getattr__(self, name):
        return getattr(self._eager(), name)


class EmbeddingBackend:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 device: str = "cpu", allow_download: bool = False):
        self.model_name = model_name
        self.device = device
        self.allow_download = allow_download
        self.model = None
        self.backend_kind = "hashing"  # set to "st" on successful load
        self.dim = FALLBACK_DIM
        self._load_attempted = False

    def _load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        if not self.allow_download:
            # strict offline: HF hub must not be contacted; cached models still load
            os.environ["HF_HUB_OFFLINE"] = "1"
        # one retry: this machine has shown transient sentence-transformers
        # import failures (AV/file-lock); without the retry a transient blip
        # poisons the embedding lineage and re-embeds all nodes with hashing
        devices = [self.device] if self.device == "cpu" else [self.device, "cpu"]
        last_exc: Exception | None = None
        for attempt in (1, 2):
            try:
                self._attempt_load(devices)
                return
            except Exception as exc:
                last_exc = exc
                self.model = None
                logger.warning("embedding load attempt %d failed: %s", attempt, exc)
        self.backend_kind = "hashing"
        self.dim = FALLBACK_DIM
        logger.warning("embedding model load failed (%s) — using hashing fallback "
                       "(retrieval quality degraded, pipeline still functional)", last_exc)

    def _attempt_load(self, devices: List[str]) -> None:
        from sentence_transformers import SentenceTransformer  # may fail transiently
        last_exc: Exception | None = None
        for dev in devices:
            try:
                model = SentenceTransformer(self.model_name, device=dev)
                self.dim = int(model.get_sentence_embedding_dimension())
                self.model = model
                self.backend_kind = "st"
                self.device = dev
                logger.info("embedding model loaded",
                            extra={"model": self.model_name, "device": dev, "dim": self.dim})
                return
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(f"all devices failed: {last_exc}")

    @property
    def is_real_model(self) -> bool:
        return self.backend_kind == "st"

    def info(self) -> dict:
        self._load()  # lazy backend means info() must trigger the real load
        return {"model": self.model_name, "backend": self.backend_kind,
                "device": self.device if self.backend_kind == "st" else "cpu",
                "dim": self.dim}

    def encode(self, texts: List[str], batch_size: int = 32,
               show_progress: bool = False) -> np.ndarray:
        """Returns (n, dim) float32, L2-normalized."""
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        self._load()
        if self.model is not None:
            try:
                vecs = self.model.encode(
                    texts, batch_size=batch_size, show_progress_bar=show_progress,
                    normalize_embeddings=True, convert_to_numpy=True,
                )
                return np.ascontiguousarray(vecs, dtype="float32")
            except Exception as exc:
                logger.warning("st encode failed (%s) — falling back to hashing", exc)
        return self._hash_encode(texts)

    # ---- deterministic fallback ----
    def _hash_encode(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            t = text.lower()
            grams = [t[j:j + 3] for j in range(max(1, len(t) - 2))]
            for g in grams:
                h = hashlib.md5(g.encode("utf-8", "ignore")).digest()
                idx = int.from_bytes(h[:2], "little") % self.dim
                sign = 1.0 if h[2] % 2 == 0 else -1.0
                out[i, idx] += sign
            # token unigrams weigh double
            for tok in t.split():
                h = hashlib.md5(tok.encode("utf-8", "ignore")).digest()
                idx = int.from_bytes(h[:2], "little") % self.dim
                sign = 1.0 if h[2] % 2 == 0 else -1.0
                out[i, idx] += 2.0 * sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (out / norms).astype("float32")


_default: Optional[EmbeddingBackend] = None


def get_embeddings() -> EmbeddingBackend:
    global _default
    if _default is None:
        raise RuntimeError("embeddings not initialized — call init_embeddings() first")
    return _default


def init_embeddings(model_name: str, device: str = "cpu",
                    allow_download: bool = False) -> EmbeddingBackend:
    global _default
    be = EmbeddingBackend(model_name, device=device, allow_download=allow_download)
    if model_name in ("stub", "hashing"):
        be._load_attempted = True   # tests: never contact HF, stay deterministic
    _default = EagerEmbeddings(be)
    return _default
