"""Auto-configuration (spec §38).

detect hardware → calculate safe memory budget → select embedding model →
select LLM → select quantization → select context length → select GPU
layers → select threads → select batch size → validate → launch.

A MIRAContext singleton carries the resolved configuration through the app
(Streamlit reruns reuse it via st.cache_resource).
"""
from __future__ import annotations

import logging
import os
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from core.hardware import HardwareProfile, RuntimeConfig, auto_configure, detect_hardware, validate_runtime
from models import model_manager

logger = logging.getLogger("mira.autoconfig")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "mira.db")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")

# keep the HF cache on the project drive — the system drive is often too small
os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, ".hf_cache"))


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    path = path or os.path.join(PROJECT_ROOT, "config", "config.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg


def ensure_dirs() -> None:
    for d in (DATA_DIR, LOGS_DIR, MODELS_DIR, EXPERIMENTS_DIR,
              os.path.join(DATA_DIR, "uploads"), os.path.join(DATA_DIR, "indexes")):
        os.makedirs(d, exist_ok=True)


@dataclass
class MIRAContext:
    hw: HardwareProfile
    cfg: Dict[str, Any]
    rc: RuntimeConfig
    warnings: List[str] = field(default_factory=list)
    llm: Any = None                # LLMBackend or None
    embeddings: Any = None         # EmbeddingBackend
    local_model: Any = None        # LocalModel or None
    offline: bool = True

    @property
    def has_llm(self) -> bool:
        return self.llm is not None and self.llm.available


def build_context(config_path: Optional[str] = None,
                  force_llm_reload: bool = False,
                  _cache: Dict[str, "MIRAContext"] = {}) -> MIRAContext:
    """Resolve the full runtime context. Cached per config path per process."""
    key = config_path or "default"
    if key in _cache and not force_llm_reload:
        return _cache[key]

    cfg = load_config(config_path)
    ensure_dirs()

    hw = detect_hardware(PROJECT_ROOT)
    rc = auto_configure(hw, cfg)
    warnings = validate_runtime(rc, hw)

    ctx = MIRAContext(hw=hw, cfg=cfg, rc=rc, warnings=warnings,
                      offline=bool(cfg.get("offline", True)))

    # embeddings (device chosen by auto_configure)
    from models.embeddings import EmbeddingBackend
    emb_cfg = cfg.get("models", {}).get("embeddings", {}) or {}
    allow_dl = (not ctx.offline) and bool(cfg.get("models", {}).get("allow_download", False))
    ctx.embeddings = EmbeddingBackend(
        rc.embedding_model, device=rc.embedding_device, allow_download=allow_dl)
    # Warm the embedding model NOW. torch reserves memory at first use; if the
    # LLM (llama.cpp) grabs its allocation first, a tight Windows commit limit
    # makes torch fail with os error 1455 and retrieval silently degrades to
    # the hashing fallback. Embeddings first = both fit.
    try:
        ctx.embeddings.encode(["warmup"])
    except Exception as exc:
        logger.warning("embedding warmup failed: %s", exc)

    # LLM: local GGUF only (remote APIs are gated off, spec §5 priority 4)
    if rc.llm_backend == "llama_cpp":
        local = model_manager.pick_local_model(rc)
        ctx.local_model = local
        if local is not None:
            from models.llm import LLMBackend
            ctx.llm = LLMBackend(
                model_path=local.path, n_ctx=rc.n_ctx,
                n_gpu_layers=rc.n_gpu_layers, n_threads=rc.n_threads,
                n_batch=rc.n_batch,
            )
            if not ctx.llm.available:
                warnings.append("LLM failed to load — answers use the deterministic "
                                "extractive fallback (see Models page / logs)")
        else:
            note = ("no local GGUF found — run Models page to download one, or "
                    "answers will use the deterministic extractive fallback")
            warnings.append(note)
            logger.info(note)

    for w in warnings:
        logger.warning(w)
    _cache[key] = ctx
    return ctx


def runtime_summary(ctx: MIRAContext) -> Dict[str, Any]:
    return {
        "hardware": ctx.hw.to_dict(),
        "runtime": ctx.rc.to_dict(),
        "llm": ctx.llm.info() if ctx.llm else {"available": False},
        "embeddings": ctx.embeddings.info() if ctx.embeddings else {},
        "local_model": ctx.local_model.name if ctx.local_model else None,
        "warnings": ctx.warnings,
        "offline": ctx.offline,
    }
