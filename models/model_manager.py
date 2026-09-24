"""Model manager (spec §5).

Model selection ladder:
1. local GGUF already on disk (models/ dir, auto-discovered)
2. compatible downloadable model (requires explicit user confirmation)
3. lightweight recommended model (largest that fits, per auto_config)
4. remote API — NOT implemented (config gate exists, defaults off)

Downloads are always gated: disk + memory pre-check, explicit confirmation,
HF token taken from the environment (never hardcoded). All models live in
models/ on the project drive.
"""
from __future__ import annotations

import glob
import logging
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("mira.models")

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

HF_GGUF_REPOS: Dict[str, str] = {
    # param-size label → HF repo containing GGUF files (Q4_K_M preferred)
    "0.5B": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    "1B": "bartowski/Llama-3.2-1B-Instruct-GGUF",
    "1.5B": "Qwen/Qwen2.5-1.5B~Instruct-GGUF",
    "2B": "bartowski/gemma-2-2b-it-GGUF",
    "3B": "bartowski/Llama-3.2-3B-Instruct-GGUF",
}

Q4_VARIANTS = ["Q4_K_M", "Q4_K_S", "Q4_0"]


@dataclass
class LocalModel:
    path: str
    size_gb: float
    params_b: float = 0.0
    quant: str = ""

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


def discover_local_gguf(models_dir: str = MODELS_DIR) -> List[LocalModel]:
    out: List[LocalModel] = []
    if not os.path.isdir(models_dir):
        return out
    for path in glob.glob(os.path.join(models_dir, "**", "*.gguf"), recursive=True):
        try:
            out.append(LocalModel(path=path, size_gb=os.path.getsize(path) / 2**30))
        except OSError:
            continue
    out.sort(key=lambda m: -m.size_gb)
    return out


def _params_from_size(size_gb: float) -> float:
    """Q4_K_M ≈ 4.8 bits/weight → params ≈ size * 8 / 4.8."""
    return round(size_gb * 8 / 4.8, 1)


def pick_local_model(rc) -> Optional[LocalModel]:
    """Choose the largest local model whose estimated need fits the runtime budget."""
    best: Optional[LocalModel] = None
    for m in discover_local_gguf():
        params = m.params_b or _params_from_size(m.size_gb)
        need = params * 4.8 / 8 + 0.8
        budget = rc.usable_vram_gb + rc.usable_ram_gb
        if need <= budget and (best is None or m.size_gb > best.size_gb):
            best = m
            best.params_b = params
            best.quant = "Q4_K_M(est)"
    return best


def estimate_disk_need(params_b: float, quant: str = "Q4_K_M") -> float:
    bits = {"Q4_K_M": 4.8, "Q5_K_M": 5.7, "Q8_0": 8.5}.get(quant, 4.8)
    return params_b * bits / 8


def precheck_download(params_b: float, quant: str, hw, cfg: Dict[str, Any],
                      models_dir: str = MODELS_DIR) -> Tuple[bool, str]:
    """(ok, message). Checks disk + memory budget before ANY download."""
    disk_need = estimate_disk_need(params_b, quant)
    max_disk = cfg.get("resources", {}).get("max_model_disk_gb", "auto")
    if str(max_disk).lower() != "auto":
        try:
            disk_need_limit = float(max_disk)
            if disk_need > disk_need_limit:
                return False, f"model needs {disk_need:.1f} GB disk > limit {disk_need_limit} GB"
        except ValueError:
            pass
    free = shutil.disk_usage(models_dir if os.path.isdir(models_dir) else os.getcwd()).free / 2**30
    if free < disk_need * 1.2:
        return False, f"insufficient disk: {free:.1f} GB free, need {disk_need:.1f} GB (+20% margin)"
    ram_need = disk_need + 0.8
    budget = rc_budget(hw, cfg)
    if ram_need > budget:
        return False, (f"model needs ~{ram_need:.1f} GB RAM/VRAM, budget {budget:.1f} GB — "
                       f"pick a smaller model or lower performance mode")
    return True, f"ok: {disk_need:.1f} GB disk, ~{ram_need:.1f} GB memory budget"


def rc_budget(hw, cfg: Dict[str, Any]) -> float:
    from core.hardware import PROFILE_FRACTIONS
    mode = str(cfg.get("performance_mode", "balanced")).lower()
    mode = mode if mode in PROFILE_FRACTIONS else "balanced"
    return hw.usable_ram_gb(mode) + hw.usable_vram_gb(mode)


def download_gguf(repo_id: str, filename: str, models_dir: str = MODELS_DIR,
                  token: Optional[str] = None) -> str:
    """Confirmed download path. Caller must run precheck + get user consent first."""
    os.makedirs(models_dir, exist_ok=True)
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id=repo_id, filename=filename, local_dir=models_dir, token=token,
    )
    logger.info("model downloaded", extra={"repo": repo_id, "file": filename})
    return path


def hf_token_from_env() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
