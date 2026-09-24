"""Model evaluation (spec §37).

Compare Base SLM + Vector RAG / Graph RAG / MIRA vs Fine-tuned SLM + MIRA.
This module wires the comparison harness; it only reports measurements that
actually exist. If no fine-tuned adapter is present, it says so.
"""
from __future__ import annotations

from typing import Any, Dict

from evaluation.ablation import run_all_systems


def compare_models(ws, records, k: int = 8,
                   finetuned_ws=None) -> Dict[str, Any]:
    """Base vs fine-tuned SLM over identical retrieval + dataset."""
    out: Dict[str, Any] = {
        "base_slm": run_all_systems(ws, records, k=k,
                                    include_baselines=False,
                                    include_ablations=False),
    }
    if finetuned_ws is not None:
        out["finetuned_slm"] = run_all_systems(
            finetuned_ws, records, k=k, include_baselines=False,
            include_ablations=False)
    else:
        out["finetuned_slm"] = "not available — no fine-tuned adapter loaded"
    return out
