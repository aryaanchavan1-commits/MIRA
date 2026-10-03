"""Unit tests for core/memory_dynamics.py — decay, reinforcement, gists."""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory import MemoryFrame, MemoryNode, MemoryType  # noqa: E402
from core.memory_dynamics import (  # noqa: E402
    apply_decay,
    build_gists,
    decay_factor,
    reinforce_nodes,
)


def _node(nid: str, *, days_old: float = 0.0, importance: float = 0.8,
          confidence: float = 0.5, sector: str = "alpha", vec=None,
          mtype: MemoryType = MemoryType.FACT) -> MemoryNode:
    ts = (datetime.utcnow() - timedelta(days=days_old)).isoformat(timespec="seconds")
    return MemoryNode(
        id=nid, concept=f"concept {nid}", memory_type=mtype,
        summary=f"summary of {nid}", sector=sector,
        importance=importance, confidence=confidence,
        created_at=ts, updated_at=ts,
        embedding=None if vec is None else np.asarray(vec, dtype=np.float32),
    )


def test_decay_fades_old_weak_memory_and_never_deletes() -> None:
    frame = MemoryFrame()
    frame.add_node(_node("old", days_old=90, importance=0.6, confidence=0.2))
    frame.add_node(_node("fresh", days_old=0, importance=0.6, confidence=0.9))
    n0 = len(frame.nodes)
    report = apply_decay(frame, half_life_days=21.0)
    assert len(frame.nodes) == n0, "decay must never delete"
    assert report["decayed"] >= 1
    assert 0.0 < report["retention_mean"] < 1.0
    old = frame.nodes["old"]
    fresh = frame.nodes["fresh"]
    assert old.importance < 0.6, "old, weakly-connected node must fade"
    assert fresh.importance == 0.6, "fresh node should be untouched"
    assert old.importance >= 0.6 * 0.15 - 1e-6, "floor respected"


def test_decay_factor_scales_with_stability() -> None:
    old = _node("a", days_old=60, confidence=0.9)
    old.metadata["_degree"] = 8
    weak = _node("b", days_old=60, confidence=0.1)
    weak.metadata["_degree"] = 0
    now = datetime.utcnow()
    r_connected = decay_factor(old, now)
    r_isolated = decay_factor(weak, now)
    assert r_connected > r_isolated, "well-wired memories decay slower"


def test_reinforce_lifts_and_resets_clock() -> None:
    frame = MemoryFrame()
    stale = _node("stale", days_old=45, importance=0.4)
    frame.add_node(stale)
    before_ts = stale.updated_at
    persisted = []
    report = reinforce_nodes(frame, ["stale", "missing"], lift=0.06,
                             persist=persisted.append)
    assert report["reinforced"] == 1, "unknown ids skipped"
    node = frame.nodes["stale"]
    assert node.importance > 0.4
    assert node.importance <= 0.98, "ceiling respected"
    assert node.updated_at >= before_ts, "decay clock reset"
    assert len(persisted) == 1 and persisted[0]["id"] == "stale"


def test_build_gists_creates_summary_nodes_with_edges() -> None:
    frame = MemoryFrame()
    vec = [1.0, 0.0, 0.0]
    for i in range(6):
        frame.add_node(_node(f"n{i}", vec=vec, sector="alpha"))
    frame.add_node(_node("far", vec=[0.0, 1.0, 0.0], sector="alpha"))
    persisted = []
    report = build_gists(frame, embeddings=None, min_members=3,
                         max_gists=24, persist=persisted.append)
    assert report["gists"] == 1, "identical vectors dedupe to one gist"
    gists = [n for n in frame.nodes.values() if n.metadata.get("gist")]
    assert len(gists) == 1
    gist = gists[0]
    edges = [e for e in frame.edges if e.source_id == gist.id
             and e.relation_type == "gist_of"]
    assert len(edges) == 3, "min_members members linked"
    assert gist.memory_type == MemoryType.SEMANTIC
    assert gist.embedding is None, "embedding stays a caller concern"
    created = [r for r in persisted if not r.get("edge")]
    assert any(r["id"] == gist.id and r["_action"] == "create"
               for r in created), "persist saw the gist row"


def test_build_gists_skips_small_clusters() -> None:
    frame = MemoryFrame()
    for i in range(2):
        frame.add_node(_node(f"s{i}", vec=[1.0, 0.0], sector="beta"))
    report = build_gists(frame, embeddings=None, min_members=3)
    assert report["gists"] == 0


def test_repeated_decay_passes_charge_each_interval_once() -> None:
    """Two immediate passes must not double-count: decay is charged per
    interval since the last pass, so an immediate second pass is a no-op and
    a nightly consolidate cannot compound a month of age into one night."""
    f = MemoryFrame()
    n = _node("old", vec=[1.0, 0.0])
    n.updated_at = (datetime.utcnow()
                    - timedelta(days=30)).isoformat(timespec="seconds")
    f.add_node(n)

    r1 = apply_decay(f, half_life_days=30.0)
    imp_after_one = f.nodes["old"].importance
    assert imp_after_one < _node("old").importance, "aged memory must fade"

    r2 = apply_decay(f, half_life_days=30.0)   # immediate second pass
    assert f.nodes["old"].importance == imp_after_one, \
        "a same-moment repeat pass must charge zero extra interval"
    assert r2["retention_mean"] >= 0.999
    assert r1["retention_mean"] < 1.0

    # 10 more days pass: the next pass charges exactly those 10 days, not 40
    f.nodes["old"].updated_at = (datetime.utcnow()
                                 - timedelta(days=10)).isoformat(timespec="seconds")
    f.nodes["old"].metadata["last_decay_at"] = (datetime.utcnow()
                                                - timedelta(days=10)
                                                ).isoformat(timespec="seconds")
    apply_decay(f, half_life_days=30.0)
    expected = imp_after_one * (0.5 ** (10 / 30.0))
    assert abs(f.nodes["old"].importance - expected) < 0.02, \
        (f.nodes["old"].importance, expected)


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)}/{len(tests)} memory-dynamics tests passed")


if __name__ == "__main__":
    _run_all()
