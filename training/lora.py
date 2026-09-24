"""Optional QLoRA fine-tuning (spec §35).

STRICTLY optional and safety-gated. Training activates only when GPU/VRAM,
RAM, disk, model compatibility, and dataset size all pass precheck; otherwise
it prints SKIP and exits without touching anything. On the reference machine
(RTX 3050 Laptop 4 GB) only 1B-3B bases with QLoRA (4-bit base, small batch,
gradient accumulation, gradient checkpointing) can qualify — and even then
this is experimental. Baselines come first; never fine-tune before they exist.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("mira.training")

MIN_SAMPLES = 50
MIN_VRAM_GB = 5.0        # 4-bit 1-3B + LoRA adapters + optimizer + activations
MIN_RAM_GB = 12.0
MIN_DISK_GB = 6.0
SUPPORTED_ARCH_HINTS = ("llama", "qwen2", "phi", "gemma", "mistral", "tinyllama")


def precheck(hw, cfg: Dict[str, Any], dataset_path: str,
             base_model_path: str) -> Tuple[bool, str]:
    """(ok, reason). Conservative by design — false SKIPs are fine."""
    if not os.path.exists(dataset_path):
        return False, f"dataset missing: {dataset_path}"
    try:
        with open(dataset_path, "r", encoding="utf-8") as fh:
            n = len(json.load(fh))
    except Exception as exc:
        return False, f"dataset unreadable: {exc}"
    if n < MIN_SAMPLES:
        return False, f"dataset too small: {n} < {MIN_SAMPLES} samples"

    vram = hw.usable_vram_gb("research")
    ram = hw.usable_ram_gb("research")
    disk = hw.free_disk_gb
    if vram < MIN_VRAM_GB:
        return False, (f"insufficient VRAM: {vram:.1f} GB usable < {MIN_VRAM_GB} GB — "
                       "SKIP TRAINING, external memory still works (spec §35)")
    if ram < MIN_RAM_GB:
        return False, f"insufficient RAM: {ram:.1f} GB usable < {MIN_RAM_GB} GB"
    if disk < MIN_DISK_GB:
        return False, f"insufficient disk: {disk:.1f} GB < {MIN_DISK_GB} GB"
    if not base_model_path.lower().endswith(".gguf"):
        return False, "only local GGUF base models supported for QLoRA conversion"
    return True, f"ok: {n} samples, VRAM {vram:.1f} GB, RAM {ram:.1f} GB"


def run_qlora(hw, cfg: Dict[str, Any], dataset_path: str,
              base_model_path: str, out_dir: str,
              dry_run: bool = True) -> Dict[str, Any]:
    """Runs QLoRA only if precheck passes AND dry_run=False. Default is a
    no-op dry run that reports what would happen. torch/peft/trl imports are
    deferred so this module never breaks machines without training deps."""
    ok, reason = precheck(hw, cfg, dataset_path, base_model_path)
    report: Dict[str, Any] = {"precheck_ok": ok, "reason": reason,
                              "executed": False}
    if not ok:
        logger.info("training skipped: %s", reason)
        return report
    if dry_run:
        report["would_do"] = [
            "load base in 4-bit (bitsandbytes NF4)",
            "attach LoRA r=16 alpha=32 on attn+mlp projections",
            "bf16 autocast, per-device batch 1, grad-accum 16, grad checkpointing",
            "cosine LR 2e-4, 1-3 epochs, eval on held-out split",
        ]
        return report
    try:
        import torch  # noqa: F401
        import peft  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:
        report["reason"] = f"training deps missing ({exc}) — SKIP"
        return report
    # ponytail: real trainer loop intentionally out of scope for this prototype —
    # wire an HF Trainer here when the research plan reaches phase 12 for real.
    report["reason"] = "trainer loop not wired in this prototype — see report['would_do']"
    return report
