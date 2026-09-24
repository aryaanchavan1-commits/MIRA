"""Experiment persistence + export (spec §32/§33).

Every run stores: id, timestamp, config, retrieval params, weights, hardware
profile, software versions, seed, results. Artifacts land in
experiments/EXP-XXXX/ and a row goes into SQLite for the UI.
"""
from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.auto_config import EXPERIMENTS_DIR, PROJECT_ROOT

EXPERIMENTS_DIR = Path(EXPERIMENTS_DIR)  # auto_config exports str; report uses pathlib

METRIC_ORDER = ["retrieval_recall", "mrr", "answer_token_f1", "context_tokens",
                "latency_ms", "questions_per_s", "n_questions"]


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def _next_exp_dir() -> Path:
    existing = [p.name for p in EXPERIMENTS_DIR.iterdir()
                if p.is_dir() and p.name.startswith("EXP-")] \
        if EXPERIMENTS_DIR.exists() else []
    n = max([int(p.split("-")[1]) for p in existing], default=0) + 1
    d = EXPERIMENTS_DIR / f"EXP-{n:04d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_experiment(name: str, config: Dict[str, Any],
                    results: Dict[str, Dict[str, Any]],
                    store=None, seed: int = 42,
                    hw=None) -> str:
    """Persist one experiment run. Returns the experiment id."""
    d = _next_exp_dir()
    stamp = datetime.now(timezone.utc).isoformat()

    meta = {
        "experiment_id": d.name,
        "name": name,
        "created_at": stamp,
        "git_commit": _git_commit(),
        "seed": seed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hardware": hw.to_dict() if hw else {},
        "config": config,
        "software": {"numpy": _ver("numpy"), "faiss": _ver("faiss"),
                     "networkx": _ver("networkx")},
    }
    (d / "config.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (d / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # CSV: one row per system
    with open(d / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["system"] + [m for m in METRIC_ORDER])
        for sysname, res in results.items():
            agg = res.get("aggregate", {})
            w.writerow([sysname] + [agg.get(m, "") for m in METRIC_ORDER])

    (d / "summary.md").write_text(
        _summary_md(meta, results), encoding="utf-8")

    if store is not None:
        store.save_experiment(name=name, config=meta, results=results,
                              git_commit=meta["git_commit"],
                              hardware=meta["hardware"], seed=seed)
    return d.name


def _ver(pkg: str) -> str:
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:
        return "?"


def _summary_md(meta: Dict[str, Any], results: Dict[str, Dict[str, Any]]) -> str:
    lines = [f"# {meta['name']}", "",
             f"- experiment: `{meta['experiment_id']}`",
             f"- created: {meta['created_at']}",
             f"- git: `{meta['git_commit']}` · seed: {meta['seed']}", "",
             "| system | " + " | ".join(METRIC_ORDER) + " |",
             "|---|" + "---|" * len(METRIC_ORDER)]
    for sysname, res in results.items():
        agg = res.get("aggregate", {})
        lines.append(f"| {sysname} | " +
                     " | ".join(str(agg.get(m, "—")) for m in METRIC_ORDER) + " |")
    lines += ["", "Metrics are computed from actual retrieval output. "
              "answer_token_f1 is a lexical proxy, not an LLM judge.", ""]
    return "\n".join(lines)


def comparison_table(results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rows for dashboard charts: MIRA variants vs baselines."""
    rows = []
    for sysname, res in results.items():
        agg = res.get("aggregate", {})
        rows.append({"system": sysname, **{m: agg.get(m) for m in METRIC_ORDER}})
    return rows
