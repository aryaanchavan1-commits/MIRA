"""BioMIRA — biologically inspired adaptive memory layer over MIRA.

Research question (tested, never assumed):

    "Can biologically inspired adaptive memory dynamics reduce catastrophic
     forgetting in a structured long-term memory system for small/local AI
     models?"

Everything here is a *computational abstraction* inspired by cognitive memory,
not a claim of biological equivalence. Parameters are configurable and the
whole layer is gated by ``biomira.enabled`` — with it off, every function is a
no-op and MIRA behaves exactly as before (§1 fair baseline).

Design constraints this module keeps (spec §20-25):

* **No new storage system.** Per-node dynamics live in ``node.metadata["bio"]``,
  a JSON blob the SQLite store already round-trips. No schema migration, no
  second graph, no second vector index, existing rows lazily get defaults.
* **No new model, no GPU.** NumPy + SQLite only; runs on CPU (spec §20-21).
* **No deletion.** Decay lowers retrieval probability and stability; it never
  removes a memory (spec §7). Consolidated memories are *protected*, never
  immutable — conflicts, corrections and merges still apply (§8, §11).
* **Every pass is explicit and reportable.** No hidden magic; the scripts and
  API return before/after reports so results can be reported honestly (§23).

Field map — existing schema reused, nothing duplicated:

    activation           -> bio.activation           (this module)
    stability            -> bio.stability            (this module)
    importance           -> node.importance          (existing)
    confidence           -> node.confidence          (existing)
    access_count         -> bio.access_count         (this module)
    last_accessed        -> bio.last_accessed        (this module)
    consolidation_score  -> bio.consolidation_score  (this module)
    decay_rate           -> bio.decay_rate           (this module)
    creation_time        -> node.created_at          (existing)
    last_modified        -> node.updated_at          (existing)
    memory_version       -> bio.memory_version       (this module)

Mechanism → existing MIRA code (do not re-implement):

    sparse/LIF activation     core/activation.py   (SpreadingActivation)
    Hebbian association       core/hebbian.py      (reinforce)
    STDP-inspired association core/hebbian.py      (stdp_update)
    gist abstraction          core/memory_dynamics.py (build_gists)

This module supplies what was missing: bounded stability dynamics, adaptive
per-memory decay, homeostatic normalization, explicit consolidation states,
a size-bounded replay buffer with an explicit selection policy, and
activation-driven radial migration.
"""
from __future__ import annotations

import datetime
import logging
import math
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from core.memory import MemoryFrame, MemoryNode
from core.memory_dynamics import decay_factor
from core.types import iso_now

logger = logging.getLogger("mira.biomira")

# Consolidation lifecycle (§8). Higher rank = more protected.
STATES: Tuple[str, ...] = ("new", "candidate", "stable", "consolidated")
STATE_RANK: Dict[str, int] = {s: i for i, s in enumerate(STATES)}

DEFAULTS: Dict[str, Any] = {
    "enabled": False,              # §1: MIRA must run without BioMIRA
    # --- stability (§2) -------------------------------------------------
    "stability_gain": 0.12,       # reinforcement per successful retrieval
    "stability_decay_per_day": 0.01,
    "stability_utility_importance": 0.30,
    "stability_utility_confidence": 0.25,
    "stability_utility_activation": 0.20,
    "stability_utility_degree": 0.15,
    "stability_utility_access": 0.10,
    # --- adaptive decay (§7) -------------------------------------------
    "half_life_days": 21.0,
    "decay_floor": 0.15,           # never below floor × original importance
    "decay_min_importance": 0.05,
    "decay_protection_gain": 0.35,  # per consolidation rank (slower decay)
    "decay_stability_gain": 1.20,
    "decay_importance_gain": 0.60,
    "decay_access_gain": 0.80,
    "decay_access_ref": 8.0,
    "decay_mult_min": 0.25,
    "decay_mult_max": 8.0,
    # --- consolidation (§8) --------------------------------------------
    "consolidate_candidate_access": 2,
    "consolidate_stable_access": 5,
    "consolidate_consolidated_access": 10,
    "consolidate_candidate_score": 0.45,
    "consolidate_stable_score": 0.60,
    "consolidate_consolidated_score": 0.72,
    "consolidate_score_access": 0.35,   # score weights (must sum ~1)
    "consolidate_score_importance": 0.20,
    "consolidate_score_confidence": 0.15,
    "consolidate_score_degree": 0.15,
    "consolidate_score_stability": 0.15,
    "consolidate_degree_ref": 8.0,
    # --- homeostatic normalization (§6) ---------------------------------
    "homeostasis": True,
    "homeostasis_percentile": 95.0,
    "homeostasis_squeeze": 0.25,        # excess above cap is scaled by this
    "homeostasis_importance_cap": 0.98,  # hard ceiling regardless of percentile
    # --- replay buffer (§9) ---------------------------------------------
    "replay_buffer_size": 500,
    "replay_cooldown_days": 7.0,
    "replay_batch": 64,
    "replay_risk_weight": 0.35,       # at-risk but valuable
    "replay_use_weight": 0.20,        # proven useful
    "replay_stability_weight": 0.15,
    "replay_degree_weight": 0.10,
    "replay_failure_weight": 0.10,    # involved in retrieval failures
    "replay_state_weight": 0.10,
    # --- radial integration (§13) --------------------------------------
    "ring_migration": False,          # experimental variable; opt-in
    "migrate_min_accesses": 3,
    "migrate_min_stability": 0.55,
    "migrate_max_per_pass": 64,
    # --- merging (§12) --------------------------------------------------
    "merge_similarity": 0.92,         # cosine above which two nodes are dupes
    "merge_max_pairs": 500,
}


def bio_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolve BioMIRA parameters: defaults <- biomira block <- enabled gate."""
    params = dict(DEFAULTS)
    config = config or {}
    # accept either a full MIRA config or an already-resolved params dict
    block = config.get("biomira") or (config if "enabled" in config else {}) or {}
    for k, v in block.items():
        if k in params:
            params[k] = v
    # `biomira.enabled` is the single switch; an explicit BIOMIRA_ENABLED-style
    # top-level key is honoured too so experiments can flip it without editing
    # YAML.
    if "BIOMIRA_ENABLED" in (config or {}):
        params["enabled"] = bool((config or {})["BIOMIRA_ENABLED"])
    params["enabled"] = bool(params["enabled"])
    return params


def enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    return bool(bio_config(config)["enabled"])


# ---------------------------------------------------------------------------
# per-node dynamics state (lives in node.metadata["bio"])
# ---------------------------------------------------------------------------

def _default_state() -> Dict[str, Any]:
    return {
        "activation": 0.0,
        "stability": 0.5,
        "consolidation_score": 0.0,
        "consolidation_state": "new",
        "access_count": 0,
        "failures": 0,
        "last_accessed": None,
        "last_replayed": None,
        "last_decay_at": None,
        "decay_rate": 0.0,
        "memory_version": 1,
        "superseded_by": None,
    }


def state(node: MemoryNode) -> Dict[str, Any]:
    """Dynamics state for a node, defaulted. Does not mutate or persist."""
    raw = node.metadata.get("bio")
    if not isinstance(raw, dict):
        raw = {}
    out = _default_state()
    out.update({k: v for k, v in raw.items() if k in out})
    return out


def _write(node: MemoryNode, st: Dict[str, Any]) -> None:
    node.metadata["bio"] = dict(st)


def _f(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _parse_ts(value: Any) -> Optional[datetime.datetime]:
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _align(ts: Optional[datetime.datetime],
           now: datetime.datetime) -> Optional[datetime.datetime]:
    """naive/aware mismatch guard (same trick as memory_dynamics)."""
    if ts is None:
        return None
    if (ts.tzinfo is None) != (now.tzinfo is None):
        return ts.replace(tzinfo=None) if ts.tzinfo is None else ts.replace(tzinfo=now.tzinfo)
    return ts


def _degree_map(frame: MemoryFrame) -> Dict[str, int]:
    deg: Dict[str, int] = defaultdict(int)
    for e in frame.edges:
        deg[e.source_id] += 1
        deg[e.target_id] += 1
    return deg


def consolidation_score(node: MemoryNode, st: Dict[str, Any], degree: int,
                        cfg: Dict[str, Any]) -> float:
    """Bounded [0,1] consolidation score from the §8 evidence signals."""
    acc_ref = max(1.0, _f(cfg["consolidate_consolidated_access"], 10))
    score = (
        _f(cfg["consolidate_score_access"]) * min(_f(st["access_count"]) / acc_ref, 1.0)
        + _f(cfg["consolidate_score_importance"]) * float(np.clip(node.importance, 0, 1))
        + _f(cfg["consolidate_score_confidence"]) * float(np.clip(node.confidence, 0, 1))
        + _f(cfg["consolidate_score_degree"]) * min(degree / max(1.0, _f(cfg["consolidate_degree_ref"], 8.0)), 1.0)
        + _f(cfg["consolidate_score_stability"]) * float(np.clip(st["stability"], 0, 1))
    )
    return float(np.clip(score, 0.0, 1.0))


def consolidation_state_for(node: MemoryNode, st: Dict[str, Any], degree: int,
                            cfg: Dict[str, Any]) -> str:
    """Lifecycle transition NEW → CANDIDATE → STABLE → CONSOLIDATED (§8)."""
    acc = _f(st["access_count"])
    score = consolidation_score(node, st, degree, cfg)
    if acc >= _f(cfg["consolidate_consolidated_access"], 10) and \
            score >= _f(cfg["consolidate_consolidated_score"], 0.72):
        return "consolidated"
    if acc >= _f(cfg["consolidate_stable_access"], 5) and \
            score >= _f(cfg["consolidate_stable_score"], 0.60):
        return "stable"
    if acc >= _f(cfg["consolidate_candidate_access"], 2) or \
            score >= _f(cfg["consolidate_candidate_score"], 0.45):
        return "candidate"
    return "new"


# ---------------------------------------------------------------------------
# strength / retention — §7 adaptive decay
# ---------------------------------------------------------------------------

def strength(node: MemoryNode, cfg: Dict[str, Any],
             degree: int = 0) -> float:
    """How much protection a memory earns, as a half-life multiplier.

    High-value, well-wired, frequently used, consolidated memories decay
    slowly; isolated, unused ones decay fast. Never deletes anything (§7).
    """
    st = state(node)
    rank = STATE_RANK.get(st["consolidation_state"], 0)
    access_term = min(math.log1p(_f(st["access_count"]))
                      / math.log1p(max(1.0, _f(cfg["decay_access_ref"], 8.0))), 1.0)
    mult = (
        1.0
        + _f(cfg["decay_protection_gain"]) * rank
        + _f(cfg["decay_stability_gain"]) * float(np.clip(st["stability"], 0, 1))
        + _f(cfg["decay_importance_gain"]) * (float(np.clip(node.importance, 0, 1)) - 0.5)
        + _f(cfg["decay_access_gain"]) * access_term
        + 0.10 * min(degree / 8.0, 1.0)
    )
    return float(np.clip(mult, _f(cfg["decay_mult_min"], 0.25),
                         _f(cfg["decay_mult_max"], 8.0)))


def retention(node: MemoryNode, cfg: Dict[str, Any],
              now: Optional[datetime.datetime] = None,
              degree: int = 0,
              min_age_ts: Optional[datetime.datetime] = None) -> float:
    """Retention factor in [0,1] — probability mass the memory keeps.

    ``min_age_ts`` (the newest of last-use / last-decay) overrides the age
    anchor so a repeat pass charges only the interval since it last ran.
    """
    now = now or datetime.datetime.utcnow()
    half_life = max(1e-6, _f(cfg["half_life_days"], 21.0)) * strength(node, cfg, degree)
    if min_age_ts is not None:
        if (min_age_ts.tzinfo is None) != (now.tzinfo is None):
            now = now.replace(tzinfo=None) if min_age_ts.tzinfo is None \
                else now.replace(tzinfo=min_age_ts.tzinfo)
        node.metadata["_decay_age_days"] = max(
            0.0, (now - min_age_ts).total_seconds() / 86400.0)
    try:
        r = decay_factor(node, now, half_life_days=half_life, stability_boost=0.0)
    finally:
        node.metadata.pop("_decay_age_days", None)
    return float(np.clip(r, 0.0, 1.0))


# ---------------------------------------------------------------------------
# §2/§8 access bookkeeping: stability update + lifecycle transition
# ---------------------------------------------------------------------------

def touch(frame: MemoryFrame, node_ids: Iterable[str],
          cfg: Optional[Dict[str, Any]] = None,
          now: Optional[datetime.datetime] = None,
          persist: Optional[Callable[[Dict[str, Any]], None]] = None,
          failed: bool = False) -> Dict[str, Any]:
    """Record retrieval use (or a retrieval failure) and update dynamics.

    ``stability(t+1) = clip(stability(t) + reinforcement - decay, 0, 1)``
    """
    cfg = bio_config(cfg)
    now = now or datetime.datetime.utcnow()
    if not cfg.get("enabled", False):
        return {"skipped": "biomira disabled", "touched": 0, "promoted": 0,
                "states": {}, "failed": bool(failed)}
    deg = _degree_map(frame)
    counts: Dict[str, int] = defaultdict(int)
    states: Dict[str, int] = defaultdict(int)
    promoted = 0
    touched = 0
    for nid in node_ids:
        node = frame.nodes.get(nid)
        if node is None:
            continue
        touched += 1
        st = state(node)
        degree = deg.get(nid, 0)
        if failed:
            st["failures"] = int(_f(st["failures"])) + 1
            delta = -_f(cfg["stability_gain"])
        else:
            st["access_count"] = int(_f(st["access_count"])) + 1
            st["last_accessed"] = iso_now()
            utility = (
                _f(cfg["stability_utility_importance"]) * float(np.clip(node.importance, 0, 1))
                + _f(cfg["stability_utility_confidence"]) * float(np.clip(node.confidence, 0, 1))
                + _f(cfg["stability_utility_activation"]) * float(np.clip(st["activation"], 0, 1))
                + _f(cfg["stability_utility_degree"]) * min(degree / 8.0, 1.0)
                + _f(cfg["stability_utility_access"]) * min(_f(st["access_count"]) / 10.0, 1.0)
            )
            delta = _f(cfg["stability_gain"]) * utility
        last = _align(_parse_ts(st["last_accessed"]), now)
        elapsed = max(0.0, (now - last).total_seconds() / 86400.0) if last else 0.0
        decay = _f(cfg["stability_decay_per_day"]) * elapsed
        st["stability"] = float(np.clip(_f(st["stability"], 0.5) + delta - decay, 0.0, 1.0))
        st["consolidation_score"] = round(consolidation_score(node, st, degree, cfg), 4)
        new_state = consolidation_state_for(node, st, degree, cfg)
        if STATE_RANK.get(new_state, 0) > STATE_RANK.get(st["consolidation_state"], 0):
            promoted += 1
        st["consolidation_state"] = new_state
        _write(node, st)
        counts[new_state] += 1
        if persist is not None and not failed:
            persist({**node.to_row(), "_action": "update"})
    report = {"touched": touched, "promoted": promoted,
              "states": dict(counts), "failed": bool(failed)}
    logger.info("biomira touch: %s", report)
    return report


def set_activation(frame: MemoryFrame, values: Dict[str, float]) -> int:
    """Persist the activation produced by core/activation.py (§3)."""
    n = 0
    for nid, v in values.items():
        node = frame.nodes.get(nid)
        if node is None:
            continue
        st = state(node)
        st["activation"] = float(np.clip(_f(v), 0.0, 1.0))
        _write(node, st)
        n += 1
    return n


# ---------------------------------------------------------------------------
# §7 adaptive decay pass
# ---------------------------------------------------------------------------

def apply_adaptive_decay(frame: MemoryFrame, cfg: Optional[Dict[str, Any]] = None,
                         now: Optional[datetime.datetime] = None,
                         dry_run: bool = False,
                         persist: Optional[Callable[[Dict[str, Any]], None]] = None
                         ) -> Dict[str, Any]:
    """Soft decay every memory by its own retention. Never deletes.

    ``persist(node_row)`` is optional write-back, matching reinforce_nodes.
    """
    cfg = bio_config(cfg)
    if not cfg.get("enabled", False):
        return {"skipped": "biomira disabled", "decayed": 0, "nodes": len(frame.nodes)}
    now = now or datetime.datetime.utcnow()
    deg = _degree_map(frame)
    floor = _f(cfg["decay_floor"], 0.15)
    min_importance = _f(cfg["decay_min_importance"], 0.05)
    changed = 0
    total_r = 0.0
    min_seen, max_seen = 1.0, 0.0
    buckets: Dict[str, int] = defaultdict(int)
    stamp = iso_now()
    for node in frame.nodes.values():
        st = state(node)
        # charge only the interval since the last decay pass: a repeated sleep
        # must not re-charge the full age, or a month of nightly passes erases
        # everything (the same root fix as memory_dynamics.apply_decay).
        # All stamps are UTC in this system, so compare tz-stripped (naive vs
        # aware ISO strings would otherwise be incomparable).
        stamps = [t.replace(tzinfo=None) if t.tzinfo is not None else t
                  for t in (_parse_ts(node.updated_at),
                            _parse_ts(st.get("last_decay_at"))) if t]
        now_cmp = now.replace(tzinfo=None) if now.tzinfo is not None else now
        last = max(stamps) if stamps else None
        r = retention(node, cfg, now_cmp, deg.get(node.id, 0),
                      min_age_ts=last) if last is not None else 1.0
        total_r += r
        min_seen, max_seen = min(min_seen, r), max(max_seen, r)
        buckets[st["consolidation_state"]] += 1
        if r >= 0.999:
            if not dry_run:
                st["last_decay_at"] = stamp
                _write(node, st)
            continue
        # consolidated memories decay, but slower (§8: protected, not immutable)
        st["decay_rate"] = round(1.0 - r, 4)
        if not dry_run:
            st["last_decay_at"] = stamp
            _write(node, st)
            node.importance = round(max(node.importance * r, node.importance * floor,
                                        min_importance), 4)
            if persist is not None:
                persist({**node.to_row(), "_action": "update"})
        changed += 1
    n = max(1, len(frame.nodes))
    report = {"nodes": len(frame.nodes), "decayed": changed,
              "retention_mean": round(total_r / n, 4),
              "retention_min": round(min_seen, 4),
              "retention_max": round(max_seen, 4),
              "half_life_days": _f(cfg["half_life_days"], 21.0),
              "states": dict(buckets), "dry_run": dry_run}
    logger.info("biomira adaptive decay: %s", report)
    return report


# ---------------------------------------------------------------------------
# §6 homeostatic normalization — popularity must not monopolize retrieval
# ---------------------------------------------------------------------------

def homeostatize(frame: MemoryFrame, cfg: Optional[Dict[str, Any]] = None
                 ) -> Dict[str, Any]:
    """Soft-cap the top tail of importance/activation (bounded, reversible).

    Compression is multiplicative above the cap, so ordering is preserved and
    a popular memory can dominate but never become the only retrievable one.
    """
    cfg = bio_config(cfg)
    if not cfg.get("homeostasis", True):
        return {"skipped": "homeostasis off", "compressed": 0}
    pctl = float(np.clip(_f(cfg["homeostasis_percentile"], 95.0), 50.0, 99.9))
    squeeze = float(np.clip(_f(cfg["homeostasis_squeeze"], 0.25), 0.01, 1.0))
    imp_cap = float(np.clip(_f(cfg["homeostasis_importance_cap"], 0.98), 0.5, 1.0))
    if not frame.nodes:
        return {"compressed": 0}
    imps = np.array([float(np.clip(n.importance, 0, 1)) for n in frame.nodes.values()])
    soft_cap = float(np.percentile(imps, pctl))
    cap = min(imp_cap, max(soft_cap, 0.5))
    compressed = 0
    for n in frame.nodes.values():
        if n.importance > cap:
            n.importance = round(min(imp_cap, cap + (n.importance - cap) * squeeze), 4)
            compressed += 1
        st = state(n)
        if st["activation"] > 1.0:
            st["activation"] = 1.0
        _write(n, st)
    return {"cap": round(cap, 4), "percentile": pctl,
            "compressed": compressed, "nodes": len(frame.nodes)}


# ---------------------------------------------------------------------------
# §9 replay buffer — bounded, explicitly prioritized, never the whole DB
# ---------------------------------------------------------------------------

def replay_priority(node: MemoryNode, st: Dict[str, Any], cfg: Dict[str, Any],
                    now: datetime.datetime, degree: int) -> float:
    """Priority = at-risk-but-valuable + proven-useful + failure-driven."""
    r = retention(node, cfg, now, degree)
    value = float(np.clip(node.importance, 0, 1))
    risk = (1.0 - r) * value
    use = min(_f(st["access_count"]) / max(1.0, _f(cfg["decay_access_ref"], 8.0)), 1.0)
    return float(np.clip(
        _f(cfg["replay_risk_weight"]) * risk
        + _f(cfg["replay_use_weight"]) * use
        + _f(cfg["replay_stability_weight"]) * float(np.clip(st["stability"], 0, 1))
        + _f(cfg["replay_degree_weight"]) * min(degree / 8.0, 1.0)
        + _f(cfg["replay_failure_weight"]) * min(_f(st["failures"]) / 3.0, 1.0)
        + _f(cfg["replay_state_weight"]) * (STATE_RANK.get(st["consolidation_state"], 0) / 3.0),
        0.0, 1.0))


def select_replay(frame: MemoryFrame, cfg: Optional[Dict[str, Any]] = None,
                  now: Optional[datetime.datetime] = None,
                  size: Optional[int] = None) -> List[str]:
    """Top-k replay candidates (the buffer). Bounded by replay_buffer_size."""
    cfg = bio_config(cfg)
    now = now or datetime.datetime.utcnow()
    size = int(size if size is not None else _f(cfg["replay_buffer_size"], 500))
    size = max(0, min(size, len(frame.nodes)))
    if not size:
        return []
    cooldown = _f(cfg["replay_cooldown_days"], 7.0)
    deg = _degree_map(frame)
    scored: List[Tuple[float, str]] = []
    for nid, node in frame.nodes.items():
        st = state(node)
        last = _align(_parse_ts(st["last_replayed"]), now)
        if last is not None and (now - last).total_seconds() / 86400.0 < cooldown:
            continue
        scored.append((replay_priority(node, st, cfg, now, deg.get(nid, 0)), nid))
    scored.sort(key=lambda kv: (-round(kv[0], 6), kv[1]))
    return [nid for _, nid in scored[:size]]


def replay(frame: MemoryFrame, cfg: Optional[Dict[str, Any]] = None,
           persist: Optional[Callable[[Dict[str, Any]], None]] = None,
           now: Optional[datetime.datetime] = None) -> Dict[str, Any]:
    """Reactivate selected old memories: reset decay clock, lift stability.

    Reuses MIRA's reinforcement so the replayed memories follow exactly the
    same dynamics as a real retrieval.
    """
    from core.memory_dynamics import reinforce_nodes
    cfg = bio_config(cfg)
    if not cfg.get("enabled", False):
        return {"skipped": "biomira disabled", "replayed": 0}
    now = now or datetime.datetime.utcnow()
    ids = select_replay(frame, cfg, now)
    if not ids:
        return {"replayed": 0, "buffer": len(frame.nodes)}
    reinforce_nodes(frame, ids, persist=persist)
    stamp = iso_now()
    for nid in ids:
        node = frame.nodes.get(nid)
        if node is None:
            continue
        st = state(node)
        st["last_replayed"] = stamp
        st["activation"] = 1.0
        st["stability"] = float(np.clip(_f(st["stability"], 0.5)
                                        + _f(cfg["stability_gain"]), 0.0, 1.0))
        st["consolidation_score"] = round(
            consolidation_score(node, st, _degree_map(frame).get(nid, 0), cfg), 4)
        st["consolidation_state"] = consolidation_state_for(
            node, st, _degree_map(frame).get(nid, 0), cfg)
        _write(node, st)
    report = {"replayed": len(ids), "buffer": len(frame.nodes),
              "batch": int(_f(cfg["replay_batch"], 64)),
              "at": stamp}
    logger.info("biomira replay: %s", report)
    return report


# ---------------------------------------------------------------------------
# §13 radial integration — activation moves memories between rings
# ---------------------------------------------------------------------------

def migrate_rings(frame: MemoryFrame, cfg: Optional[Dict[str, Any]] = None,
                  now: Optional[datetime.datetime] = None,
                  persist: Optional[Callable[[Dict[str, Any]], None]] = None
                  ) -> Dict[str, Any]:
    """Promote repeatedly used, stable memories inward; demote faded ones.

    ponytail: promotion is capped at ``migrate_max_per_pass`` and requires
    real access history. An uncapped global promotion collapses the inner
    rings on large workspaces (measured on the 83k-node bench: ring-0/1
    share fell to ~1.5% and MRR with it). Upgrade path: incremental
    promotion driven by the replay buffer rather than a full-frame sweep.
    """
    cfg = bio_config(cfg)
    if not cfg.get("ring_migration", False):
        return {"skipped": "ring migration off"}
    now = now or datetime.datetime.utcnow()
    max_ring = max(1, int((cfg.get("max_rings") or 5)) - 1)
    budget = int(_f(cfg["migrate_max_per_pass"], 64))
    deg = _degree_map(frame)
    min_acc = _f(cfg["migrate_min_accesses"], 3)
    min_stab = _f(cfg["migrate_min_stability"], 0.55)
    promoted: List[str] = []
    demoted: List[str] = []
    for nid, node in sorted(frame.nodes.items()):
        if node.ring is None:
            continue
        st = state(node)
        if (node.ring > 0 and _f(st["access_count"]) >= min_acc
                and _f(st["stability"]) >= min_stab
                and len(promoted) < budget):
            node.ring -= 1
            node.updated_at = iso_now()
            promoted.append(nid)
        elif (node.ring < max_ring and _f(node.importance) <= _f(cfg["decay_min_importance"], 0.05)
              and retention(node, cfg, now, deg.get(nid, 0)) <= 1e-3):
            node.ring += 1
            demoted.append(nid)
    for nid in promoted + demoted:
        node = frame.nodes.get(nid)
        if node is not None and persist is not None:
            persist({**node.to_row(), "_action": "update"})
    report = {"promoted": len(promoted), "demoted": len(demoted),
              "budget": budget, "max_ring": max_ring}
    logger.info("biomira ring migration: %s", report)
    return report


# ---------------------------------------------------------------------------
# §11/§12 versioning + merge detection
# ---------------------------------------------------------------------------

def mark_node_superseded(node: MemoryNode, new_id: str) -> MemoryNode:
    """Version one memory as history of another (no frame, no store).

    This is the single place supersession is written, so the conflict path
    (core/updater.py) and the frame-level helper below cannot drift apart.
    """
    st = state(node)
    st["superseded_by"] = new_id
    st["memory_version"] = int(_f(st["memory_version"], 1)) + 1
    st["consolidation_state"] = "consolidated"  # protected history (§8/§11)
    st["stability"] = 1.0
    _write(node, st)
    node.metadata["superseded_reason"] = f"superseded_by:{new_id}"
    node.updated_at = iso_now()
    return node


def mark_superseded(frame: MemoryFrame, old_id: str, new_id: str,
                    persist: Optional[Callable[[Dict[str, Any]], None]] = None
                    ) -> Optional[str]:
    """A correction does not erase the old fact: version it and link forward.

    The superseded memory keeps its ``valid_until`` window so historical
    questions still resolve to the old value (§11).
    """
    old = frame.nodes.get(old_id)
    if old is None or new_id not in frame.nodes:
        return None
    mark_node_superseded(old, new_id)
    if persist is not None:
        persist({**old.to_row(), "_action": "update"})
    return old_id


def detect_merge_candidates(frame: MemoryFrame, cfg: Optional[Dict[str, Any]] = None,
                            embeddings: Any = None) -> Dict[str, Any]:
    """Report duplicate / related / conflicting pairs. Never merges silently.

    Provenance is never destroyed: application goes through
    ``core.updater.MemoryUpdater.merge``, which keeps ``source_ids`` and
    blocks contradictory merges (§12).
    """
    from core.conflicts import detect_conflict
    cfg = bio_config(cfg)
    thr = float(np.clip(_f(cfg["merge_similarity"], 0.92), 0.0, 1.0))
    max_pairs = int(_f(cfg["merge_max_pairs"], 500))
    facts = [n for n in frame.nodes.values()
             if n.embedding is not None and embeddings is not None]
    if len(facts) < 2:
        return {"pairs": 0, "duplicates": [], "conflicts": [],
                "note": "embeddings required for semantic merge detection"}
    X = np.vstack([n.embedding for n in facts])
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-9)
    sims = X @ X.T
    np.fill_diagonal(sims, -1.0)
    dups: List[Dict[str, Any]] = []
    conflicts: List[Dict[str, Any]] = []
    seen: set = set()
    order = np.dstack(np.unravel_index(np.argsort(-sims, axis=None), sims.shape))[0]
    for i, j in order:
        if len(dups) + len(conflicts) >= max_pairs:
            break
        a, b = facts[int(i)], facts[int(j)]
        if a.id > b.id:
            a, b = b, a
        key = (a.id, b.id)
        if key in seen or sims[int(i), int(j)] < thr:
            continue
        seen.add(key)
        pair = {"a": a.id, "b": b.id,
                "similarity": round(float(sims[int(i), int(j)]), 4),
                "a_concept": a.concept[:120], "b_concept": b.concept[:120]}
        if detect_conflict(a, b):
            conflicts.append(pair)
        else:
            dups.append(pair)
    return {"pairs": len(dups) + len(conflicts), "duplicates": dups,
            "conflicts": conflicts, "threshold": thr,
            "note": "conflicts must go through updater.merge (keeps both sides)"}


# ---------------------------------------------------------------------------
# §19 transparency — why was this retrieved / decayed / consolidated
# ---------------------------------------------------------------------------

def explain_node(node: MemoryNode, cfg: Optional[Dict[str, Any]] = None,
                 now: Optional[datetime.datetime] = None,
                 degree: Optional[int] = None) -> Dict[str, Any]:
    """Per-node rationale. Everything the UI shows is computed here."""
    cfg = bio_config(cfg)
    now = now or datetime.datetime.utcnow()
    st = state(node)
    deg = degree or 0
    r = retention(node, cfg, now, deg)
    contrib = {
        "importance": float(np.clip(node.importance, 0, 1)),
        "confidence": float(np.clip(node.confidence, 0, 1)),
        "stability": float(np.clip(st["stability"], 0, 1)),
        "activation": float(np.clip(st["activation"], 0, 1)),
        "access_count": min(_f(st["access_count"]) / 10.0, 1.0),
        "degree": min(deg / 8.0, 1.0),
    }
    score = consolidation_score(node, st, deg, cfg)
    return {
        "node_id": node.id,
        "concept": node.concept[:160],
        "ring": node.ring,
        "sector": node.sector,
        "state": st["consolidation_state"],
        "activation": round(contrib["activation"], 4),
        "stability": round(contrib["stability"], 4),
        "importance": round(contrib["importance"], 4),
        "confidence": round(contrib["confidence"], 4),
        "access_count": int(_f(st["access_count"])),
        "failures": int(_f(st["failures"])),
        "consolidation_score": round(score, 4),
        "consolidation_state": st["consolidation_state"],
        "retention": round(r, 4),
        "decay_rate": round(_f(st["decay_rate"]), 4),
        "strength_multiplier": round(strength(node, cfg, deg), 4),
        "memory_version": int(_f(st["memory_version"], 1)),
        "superseded_by": st["superseded_by"],
        "last_accessed": st["last_accessed"],
        "last_replayed": st["last_replayed"],
        "why_decayed": _why(st, r),
        "why_consolidated": _why_state(st, score, node, cfg),
    }


def _why(st: Dict[str, Any], r: float) -> str:
    bits = []
    if _f(st["access_count"]) == 0:
        bits.append("never retrieved")
    else:
        bits.append(f"{int(_f(st['access_count']))} retrievals")
    last = st["last_accessed"]
    bits.append("last used " + str(last) if last else "no recorded use")
    bits.append(f"retention {r:.2f}")
    if _f(st["decay_rate"]) > 0:
        bits.append(f"last decay pass removed {100 * _f(st['decay_rate']):.0f}%")
    return "; ".join(bits)


def _why_state(st: Dict[str, Any], score: float, node: MemoryNode,
               cfg: Dict[str, Any]) -> str:
    target = st["consolidation_state"]
    nxt = {"new": "candidate", "candidate": "stable",
           "stable": "consolidated", "consolidated": None}[target]
    base = (f"score {score:.2f} from access {int(_f(st['access_count']))}, "
            f"importance {node.importance:.2f}, confidence {node.confidence:.2f}, "
            f"stability {float(np.clip(st['stability'], 0, 1)):.2f}")
    if nxt is None:
        return base + " — fully consolidated (decays slower; still editable)"
    return base + f" — needs {nxt}"


# ---------------------------------------------------------------------------
# dashboard aggregate for the UI / API (§18)
# ---------------------------------------------------------------------------

def summary(frame: MemoryFrame, cfg: Optional[Dict[str, Any]] = None,
            now: Optional[datetime.datetime] = None,
            top_k: int = 20) -> Dict[str, Any]:
    """One-pass aggregate: state histogram, active set, at-risk set."""
    cfg = bio_config(cfg)
    now = now or datetime.datetime.utcnow()
    deg = _degree_map(frame)
    hist: Dict[str, int] = {s: 0 for s in STATES}
    activated: List[Tuple[float, str, str]] = []
    at_risk: List[Tuple[float, str, str]] = []
    replays = 0
    failures = 0
    total_acc = 0.0
    for nid, node in frame.nodes.items():
        st = state(node)
        hist[st["consolidation_state"]] = hist.get(st["consolidation_state"], 0) + 1
        if _f(st["activation"]) > 0:
            activated.append((_f(st["activation"]), nid, node.concept[:80]))
        r = retention(node, cfg, now, deg.get(nid, 0))
        if r < 0.5:
            at_risk.append((r, nid, node.concept[:80]))
        if st["last_replayed"]:
            replays += 1
        failures += int(_f(st["failures"]))
        total_acc += _f(st["access_count"])
    activated.sort(key=lambda t: (-t[0], t[1]))
    at_risk.sort()
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "nodes": len(frame.nodes),
        "edges": len(frame.edges),
        "states": hist,
        "mean_activation": round(sum(a for a, _, _ in activated) / max(1, len(activated)), 4),
        "active_count": len(activated),
        "at_risk_count": len(at_risk),
        "mean_access_count": round(total_acc / max(1, len(frame.nodes)), 4),
        "replayed_nodes": replays,
        "failure_count": failures,
        "top_active": [{"node_id": i, "activation": round(a, 4), "concept": c}
                       for a, i, c in activated[:top_k]],
        "top_at_risk": [{"node_id": i, "retention": round(r, 4), "concept": c}
                        for r, i, c in at_risk[:top_k]],
        "replay_buffer_size": int(_f(cfg["replay_buffer_size"], 500)),
    }
