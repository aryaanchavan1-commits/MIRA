"""Memory dynamics: forgetting, reinforcement, gist abstraction ("sleep").

Three mechanisms that move MIRA from "inspired by human memory" toward
modeling human memory:

1. **Forgetting (soft decay)** — Ebbinghaus-style: importance fades with time
   since last use, exponentially, scaled by stability (well-connected,
   high-confidence memories decay slower). Never deletes; retrieval
   probability fades. ``updated_at`` timestamps are the activity signal.

2. **Reinforcement** — a retrieval that USES a node resets its decay clock
   (bump ``updated_at``) and lifts importance toward its ceiling. Reuses the
   same Hebbian-consolidation trigger the answer pipeline already fires on
   grounded answers.

3. **Gist abstraction ("sleep")** — offline pass that clusters fact nodes by
   semantic sector, summarizes each cluster, and stores the summary as a
   higher-ring SEMANTIC "gist" node wired to its members with "gist_of"
   edges. Human memory keeps the summary and loses the verbatim detail;
   this gives retrieval a cheap general-level hit above detail nodes.

All three are explicit, benchmarkable passes — never hidden magic. The
consolidation script reports before/after retrieval metrics so the paper can
claim (or disclaim) the benefit.
"""
from __future__ import annotations

import datetime
import logging
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from core.memory import MemoryEdge, MemoryFrame, MemoryNode, MemoryType
from core.types import iso_now

logger = logging.getLogger("mira.dynamics")


# ---------------------------------------------------------------------------
# 1. forgetting — Ebbinghaus-style soft decay
# ---------------------------------------------------------------------------

def _parse_ts(value: str) -> Optional[datetime.datetime]:
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def decay_factor(node: MemoryNode, now: datetime.datetime,
                 half_life_days: float = 21.0,
                 stability_boost: float = 0.5) -> float:
    """Retention factor in [0, 1] — how much of the memory survives.

    R(t) = exp(-t / tau) with tau scaled by stability: degree and confidence
    act as consolidators (a well-wired memory decays slower), mirroring the
    standard finding that spaced, connected traces persist longer.
    """
    ts = _parse_ts(node.updated_at) or _parse_ts(node.created_at)
    if ts is None:
        return 1.0
    if (ts.tzinfo is None) != (now.tzinfo is None):
        now = now.replace(tzinfo=None) if ts.tzinfo is None else now.replace(tzinfo=ts.tzinfo)
    # apply_decay overrides the age with "elapsed since the last decay pass"
    # so repeated sleeps charge each interval once (see _last_activity)
    override = node.metadata.get("_decay_age_days")
    days = float(override) if override is not None else max(
        0.0, (now - ts).total_seconds() / 86400.0)
    days = max(0.0, days)
    degree = node.metadata.get("_degree", 0)  # set by caller (apply_decay)
    stability = 1.0 + stability_boost * (min(degree, 8) / 8.0) * node.confidence
    tau = max(half_life_days, 1e-6) / np.log(2.0) * stability
    return float(np.exp(-days / tau))


def _last_activity(node: MemoryNode) -> Optional[datetime.datetime]:
    """The newest of last-modified and last-decayed.

    A decay pass must charge only the interval SINCE it last ran, not the
    full age again — otherwise a nightly consolidate multiplies a memory by
    R(age) every night and a month of sleep erases it. Clock resets from use
    (reinforce_nodes touches updated_at) still count as activity; created_at
    is only a fallback when updated_at is missing.
    """
    base = _parse_ts(node.updated_at) or _parse_ts(node.created_at)
    last_decay = _parse_ts(node.metadata.get("last_decay_at"))
    stamps = [s for s in (base, last_decay) if s is not None]
    return max(stamps) if stamps else None


def apply_decay(frame: MemoryFrame, now: Optional[datetime.datetime] = None,
                half_life_days: float = 21.0, floor: float = 0.15,
                min_importance: float = 0.05, dry_run: bool = False
                ) -> Dict[str, Any]:
    """Soften every node's importance by retention; never below ``floor`` ×
    original, never deletes. Returns a report (also used by the paper)."""
    now = now or datetime.datetime.utcnow()
    degree: Dict[str, int] = defaultdict(int)
    for e in frame.edges:
        degree[e.source_id] += 1
        degree[e.target_id] += 1

    changed, total_r, n = 0, 0.0, 0
    min_seen, max_seen = 1.0, 0.0
    stamp = now.isoformat(timespec="seconds")
    for node in frame.nodes.values():
        node.metadata["_degree"] = degree.get(node.id, 0)
        ts = _last_activity(node)
        if ts is not None:
            if (ts.tzinfo is None) != (now.tzinfo is None):
                now = now.replace(tzinfo=None) if ts.tzinfo is None \
                    else now.replace(tzinfo=ts.tzinfo)
            node.metadata["_decay_age_days"] = max(
                0.0, (now - ts).total_seconds() / 86400.0)
        r = decay_factor(node, now, half_life_days=half_life_days)
        node.metadata.pop("_degree", None)
        node.metadata.pop("_decay_age_days", None)
        if not dry_run:
            node.metadata["last_decay_at"] = stamp
        total_r += r
        n += 1
        min_seen, max_seen = min(min_seen, r), max(max_seen, r)
        if r >= 0.999:
            continue
        faded = node.importance * r
        node.importance = round(max(faded, node.importance * floor,
                                    min_importance), 4)
        changed += 1
    report = {"nodes": n, "decayed": changed,
              "retention_mean": round(total_r / max(n, 1), 4),
              "retention_min": round(min_seen, 4),
              "retention_max": round(max_seen, 4),
              "half_life_days": half_life_days, "dry_run": dry_run}
    logger.info("decay pass: %s", report)
    return report


# ---------------------------------------------------------------------------
# 2. reinforcement — use strengthens and resets the decay clock
# ---------------------------------------------------------------------------

def reinforce_nodes(frame: MemoryFrame, node_ids: Iterable[str],
                    lift: float = 0.06, ceiling: float = 0.98,
                    persist=None) -> Dict[str, Any]:
    """Lift importance toward ceiling and touch ``updated_at`` so the decay
    clock restarts. ``persist(node_row)`` is optional for write-back."""
    now = iso_now()
    touched = 0
    for nid in set(node_ids):
        node = frame.nodes.get(nid)
        if node is None:
            continue
        node.importance = round(min(ceiling, node.importance + lift
                                    * (1.2 - node.importance)), 4)
        node.updated_at = now
        touched += 1
        if persist is not None:
            persist({**node.to_row(), "_action": "update"})
    return {"reinforced": touched, "lift": lift}


# ---------------------------------------------------------------------------
# 3. gist abstraction — summaries above details ("sleep" output)
# ---------------------------------------------------------------------------

def _truncate(text: str, limit: int = 220) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def build_gists(frame: MemoryFrame, embeddings, min_members: int = 3,
                max_gists: int = 24, persist=None) -> Dict[str, Any]:
    """Cluster fact/document nodes per sector by embedding centroid proximity
    (greedy, deterministic) and store one SEMANTIC gist node per cluster,
    wired to members with ``gist_of`` edges. Gists carry a mid importance and
    ring is recomputed by ``place()`` by the caller."""
    by_sector: Dict[str, List[MemoryNode]] = defaultdict(list)
    for n in frame.nodes.values():
        if n.memory_type in (MemoryType.FACT, MemoryType.DOCUMENT,
                             MemoryType.EPISODIC) and n.embedding is not None:
            by_sector[n.sector or "general"].append(n)

    created = 0
    members_total = 0
    existing_concepts = {n.concept.lower() for n in frame.nodes.values()}
    new_edges: List[MemoryEdge] = []

    for sector, nodes in sorted(by_sector.items()):
        if len(nodes) < min_members:
            continue
        X = np.stack([n.embedding for n in nodes])
        # greedy farthest-point seeds, deterministic
        k = max(1, min(max_gists // max(len(by_sector), 1) or 1,
                       len(nodes) // min_members))
        if k == 0:
            continue
        centroid = X.mean(axis=0)
        order = np.argsort(-np.linalg.norm(X - centroid, axis=1))
        seeds = [int(order[0])]
        while len(seeds) < k:
            dists = np.min([np.linalg.norm(X - X[s], axis=1) for s in seeds],
                           axis=0)
            nxt = int(np.argmax(dists))
            if dists[nxt] < 1e-6 or nxt in seeds:
                break
            seeds.append(nxt)
        for s in seeds:
            sim = X @ X[s]
            members = [nodes[i] for i in np.argsort(-sim)[:min_members]
                       if sim[i] > 0.25]
            if len(members) < min_members:
                continue
            summary = _truncate("; ".join(
                (m.summary or m.raw_text or m.concept) for m in members[:4]))
            concept = f"Gist[{sector}]: {_truncate(members[0].concept, 48)}"
            if concept.lower() in existing_concepts:
                continue
            existing_concepts.add(concept.lower())
            gist = MemoryNode(
                concept=concept, memory_type=MemoryType.SEMANTIC,
                summary=summary, raw_text="",
                parent_id=members[0].parent_id,
                importance=round(float(np.mean([m.importance for m in members])) + 0.05, 4),
                confidence=round(float(np.mean([m.confidence for m in members])), 4),
                source_ids=sorted({sid for m in members for sid in m.source_ids})[:8],
                ring=1, depth=1,  # §9: gists ARE major concepts, one ring under the core
                metadata={"gist": True, "sector": sector,
                          "members": [m.id for m in members]},
            )
            frame.add_node(gist)
            gist_edges = [
                {"edge": True, "source_id": gist.id, "target_id": m.id,
                 "relation_type": "gist_of", "weight": 0.8,
                 "confidence": gist.confidence}
                for m in members
            ]
            for e in gist_edges:
                frame.add_edge(MemoryEdge(
                    source_id=e["source_id"], target_id=e["target_id"],
                    relation_type="gist_of", weight=e["weight"],
                    confidence=e["confidence"]))
            new_edges.extend(gist_edges)
            if persist is not None:
                persist({**gist.to_row(), "_action": "create"})
                for e in gist_edges:
                    persist(e)
            created += 1
            members_total += len(members)
    report = {"gists": created, "members_linked": members_total,
              "sectors": len(by_sector)}
    logger.info("gist pass: %s", report)
    return report
