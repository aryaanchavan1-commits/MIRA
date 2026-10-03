"""Unit tests for core/biomira.py — the adaptive memory layer.

Covers every mechanism in the spec plus the §1 guarantee that plain MIRA is
unaffected when the layer is disabled.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import biomira as B  # noqa: E402
from core.memory import MemoryEdge, MemoryFrame, MemoryNode, MemoryType  # noqa: E402

ON = {"biomira": {"enabled": True}}


def _cfg(**over) -> dict:
    return B.bio_config({"biomira": {"enabled": True, **over}})


def _node(nid: str, *, days_old: float = 0.0, importance: float = 0.6,
          confidence: float = 0.6, ring: int = 1, vec=None,
          mtype: MemoryType = MemoryType.FACT,
          concept: str = "") -> MemoryNode:
    ts = (datetime.utcnow() - timedelta(days=days_old)).isoformat(timespec="seconds")
    return MemoryNode(
        id=nid, concept=concept or f"concept {nid}", memory_type=mtype,
        summary=f"summary of {nid}", ring=ring, sector="alpha",
        importance=importance, confidence=confidence,
        created_at=ts, updated_at=ts,
        embedding=None if vec is None else np.asarray(vec, dtype=np.float32),
    )


def _frame(n: int = 4, **kw) -> MemoryFrame:
    f = MemoryFrame()
    for i in range(n):
        f.add_node(_node(f"n{i}", days_old=i * 30, vec=[1.0, i * 0.01], **kw))
    f.add_edge(MemoryEdge("n0", "n1"))
    return f


# --- §1 the layer must be a strict no-op when disabled ---------------------

def test_disabled_layer_is_a_noop() -> None:
    f = _frame()
    before = {n.id: (n.importance, n.ring, dict(n.metadata)) for n in f.nodes.values()}
    assert not B.enabled({})
    for report in (B.apply_adaptive_decay(f, B.bio_config()),
                   B.replay(f, B.bio_config()),
                   B.migrate_rings(f, B.bio_config())):
        assert "skipped" in report, report
    assert "skipped" in B.touch(f, ["n0"], B.bio_config())
    for nid, (imp, ring, meta) in before.items():
        assert f.nodes[nid].importance == imp
        assert f.nodes[nid].ring == ring
        assert f.nodes[nid].metadata == meta, "disabled layer must not write state"
    assert B.summary(f, B.bio_config())["enabled"] is False


def test_stability_component_is_zero_when_disabled() -> None:
    """The baseline must be bit-identical: kappa_stability defaults to 0."""
    from core.retrieval import MIRARetriever  # import cost only, no model load
    w = {"retrieval_score": {}, "biomira": {"enabled": False}}
    r = MIRARetriever.__new__(MIRARetriever)
    r.config = w
    r.bio = B.bio_config(w)
    r.bio_enabled = bool(r.bio["enabled"])
    assert r._stability(_node("x")) == 0.0
    w2 = {"biomira": {"enabled": True}}
    r.bio = B.bio_config(w2)
    r.bio_enabled = True
    n = _node("y")
    B.touch(MemoryFrame(), [], B.bio_config())  # no-op, keeps state() importable
    n.metadata["bio"] = {"stability": 0.8}
    assert r._stability(n) == 0.8


# --- §2 stability ----------------------------------------------------------

def test_stability_rises_with_use_and_is_bounded() -> None:
    f = _frame()
    cfg = _cfg()
    n0 = B.state(f.nodes["n0"])["stability"]
    for _ in range(40):
        B.touch(f, ["n0", "n1"], cfg)
    st = B.state(f.nodes["n0"])
    assert st["stability"] > n0, "repeated retrieval must raise stability"
    assert 0.0 <= st["stability"] <= 1.0, "stability must stay bounded"
    assert st["access_count"] == 40
    assert st["last_accessed"], "last_accessed must be recorded"


def test_failed_retrieval_lowers_stability() -> None:
    f = _frame()
    cfg = _cfg()
    for _ in range(3):
        B.touch(f, ["n2"], cfg)
    good = B.state(f.nodes["n2"])["stability"]
    B.touch(f, ["n2"], cfg, failed=True)
    after = B.state(f.nodes["n2"])
    assert after["stability"] < good, "failure must reduce stability"
    assert after["failures"] == 1
    assert after["access_count"] == 3, "a failure is not a use"


# --- §7 adaptive decay -----------------------------------------------------

def test_adaptive_decay_fades_but_never_deletes() -> None:
    f = _frame()
    n0 = len(f.nodes)
    report = B.apply_adaptive_decay(f, _cfg())
    assert len(f.nodes) == n0, "decay must never delete a memory"
    assert report["decayed"] >= 1
    assert 0.0 < report["retention_mean"] < 1.0
    assert f.nodes["n2"].importance < 0.6, "aged memory must fade"
    assert f.nodes["n0"].importance == 0.6, "a just-used memory keeps its importance"
    assert f.nodes["n3"].importance >= 0.6 * 0.15 - 1e-9, "floor must hold"


def test_consolidated_memories_decay_slower_than_new_ones() -> None:
    cfg = _cfg()
    a, b = _node("prot", days_old=90), _node("fresh_new", days_old=90)
    B._write(a, {"consolidation_state": "consolidated", "stability": 0.9,
                 "access_count": 30, "stability_decay_rate": 0})
    now = datetime.utcnow()
    assert B.retention(b, cfg, now) < B.retention(a, cfg, now), \
        "protection must slow decay (spec §8: protected, not immutable)"


def test_strength_is_bounded() -> None:
    cfg = _cfg()
    hot = _node("hot", importance=1.0, confidence=1.0)
    B._write(hot, {"stability": 1.0, "access_count": 10_000,
                   "consolidation_state": "consolidated"})
    cold = _node("cold", importance=0.0, confidence=0.0)
    assert cfg["decay_mult_max"] >= B.strength(hot, cfg) >= cfg["decay_mult_min"]
    assert cfg["decay_mult_min"] <= B.strength(cold, cfg) <= cfg["decay_mult_max"]


# --- §8 consolidation lifecycle -------------------------------------------

def test_consolidation_lifecycle_advances_with_evidence() -> None:
    f = _frame()
    cfg = _cfg()
    assert B.state(f.nodes["n0"])["consolidation_state"] == "new"
    B.touch(f, ["n0"], cfg)
    assert B.state(f.nodes["n0"])["consolidation_state"] in ("new", "candidate")
    for _ in range(30):
        B.touch(f, ["n0", "n1"], cfg)
    st = B.state(f.nodes["n0"])
    assert st["consolidation_state"] == "consolidated", st
    assert 0.0 <= st["consolidation_score"] <= 1.0
    ranks = [B.STATE_RANK[s] for s in B.STATES]
    assert ranks == sorted(ranks), "state order must be monotonic"


def test_consolidated_state_does_not_block_correction() -> None:
    """Protected ≠ immutable (§8): a superseded fact is still versioned."""
    f = _frame()
    old, new = f.nodes["n0"], _node("n9", concept="new value")
    f.add_node(new)
    for _ in range(30):
        B.touch(f, [old.id], _cfg())
    assert B.mark_superseded(f, old.id, new.id) == old.id
    st = B.state(old)
    assert st["superseded_by"] == new.id
    assert st["memory_version"] == 2, "correction bumps the version"
    assert old.concept == "concept n0", "the old memory is preserved, not erased"


# --- §6 homeostatic normalization -----------------------------------------

def test_homeostasis_stops_one_memory_from_monopolizing() -> None:
    f = MemoryFrame()
    for i in range(20):
        f.add_node(_node(f"m{i}", importance=0.5, vec=[1.0, i * 0.01]))
    f.nodes["m0"].importance = 1.0
    rep = B.homeostatize(f, _cfg())
    assert rep["compressed"] >= 1
    assert f.nodes["m0"].importance < 1.0, "the top tail must be compressed"
    assert f.nodes["m0"].importance >= f.nodes["m5"].importance, \
        "homeostasis compresses, it does not reorder"
    assert all(0.0 <= n.importance <= 1.0 for n in f.nodes.values())


def test_homeostasis_can_be_disabled() -> None:
    f = MemoryFrame()
    f.add_node(_node("m0", importance=1.0))
    assert "skipped" in B.homeostatize(f, _cfg(homeostasis=False))


# --- §9 replay buffer ------------------------------------------------------

def test_replay_buffer_is_bounded_and_prioritizes_at_risk() -> None:
    f = _frame(6)
    cfg = _cfg(replay_buffer_size=3)
    ids = B.select_replay(f, cfg)
    assert len(ids) == 3, "buffer must be bounded, never the whole DB"
    # the oldest valuable memory is the most at-risk and must be chosen
    assert ids[0] == "n5", ids
    rep = B.replay(f, cfg)
    assert rep["replayed"] == 3
    for nid in ids:
        st = B.state(f.nodes[nid])
        assert st["last_replayed"], "replayed nodes are stamped for cooldown"
    nxt = B.select_replay(f, cfg)
    assert not set(ids) & set(nxt), "cooldown must exclude recent replays"
    assert len(nxt) <= 3, "the buffer stays bounded across rounds"


def test_replay_lifts_protected_state() -> None:
    f = _frame(3)
    cfg = _cfg()
    target = f.nodes["n2"]
    before = B.state(target)["stability"]
    B.replay(f, cfg)
    assert B.state(target)["stability"] > before
    assert B.state(target)["activation"] == 1.0


# --- §13 radial integration -----------------------------------------------

def test_ring_migration_moves_used_memories_inward_only() -> None:
    f = MemoryFrame()
    used, faded = _node("used", ring=3, importance=0.8), _node("faded", ring=1, importance=0.05)
    for n in (used, faded):
        f.add_node(n)
    cfg = _cfg(ring_migration=True, max_rings=5)
    B._write(faded, {"importance": 0.05})
    faded.importance = 0.05
    faded.updated_at = (datetime.utcnow() - timedelta(days=4000)).isoformat()
    for _ in range(10):
        B.touch(f, [used.id], cfg)
    rep = B.migrate_rings(f, cfg)
    assert rep["promoted"] == 1, rep
    assert used.ring == 2, "repeated use pulls a memory toward the core"
    assert "skipped" in B.migrate_rings(f, B.bio_config())
    assert "skipped" in B.migrate_rings(f, _cfg(ring_migration=False))


def test_ring_migration_respects_per_pass_budget() -> None:
    f = MemoryFrame()
    cfg = _cfg(ring_migration=True, migrate_max_per_pass=5, max_rings=5)
    for i in range(40):
        f.add_node(_node(f"r{i}", ring=3, importance=0.9))
        B._write(f.nodes[f"r{i}"], {"stability": 0.9, "access_count": 9,
                                     "consolidation_state": "stable"})
    rep = B.migrate_rings(f, cfg)
    assert rep["promoted"] <= 5, rep
    assert rep["budget"] == 5


# --- §12 merging -----------------------------------------------------------

def test_merge_detection_separates_duplicates_from_conflicts() -> None:
    f = MemoryFrame()
    f.add_node(_node("d1", vec=[1.0, 0.0, 0.0]))
    f.add_node(_node("d2", vec=[1.0, 0.001, 0.0], concept="same thing"))
    f.add_node(_node("c1", concept="city mayor is Ann",
                     mtype=MemoryType.FACT, vec=[0.0, 1.0, 0.0]))
    f.add_node(_node("c2", concept="city mayor is Bob",
                     mtype=MemoryType.FACT, vec=[0.0, 1.0, 0.001]))
    rep = B.detect_merge_candidates(f, _cfg(merge_similarity=0.9), embeddings=object())
    assert rep["pairs"] >= 2, rep
    assert rep["duplicates"], "near-identical facts are merge candidates"
    assert rep["conflicts"], "same subject, different value is a conflict"
    assert "keeps both sides" in rep["note"], "merging must preserve provenance"


def test_merge_detection_requires_embeddings() -> None:
    assert B.detect_merge_candidates(_frame(), _cfg(), embeddings=None)["pairs"] == 0


# --- §19 explanation + dashboard ------------------------------------------

def test_explain_node_answers_why_decayed_and_why_consolidated() -> None:
    f = _frame()
    cfg = _cfg()
    for _ in range(30):
        B.touch(f, ["n0"], cfg)
    ex = B.explain_node(f.nodes["n0"], cfg, degree=1)
    for key in ("activation", "stability", "importance", "confidence",
                "consolidation_score", "consolidation_state", "retention",
                "decay_rate", "memory_version", "why_decayed",
                "why_consolidated", "ring", "sector"):
        assert key in ex, key
    assert ex["why_decayed"] and "retrievals" in ex["why_decayed"]
    assert "score" in ex["why_consolidated"]


def test_summary_reports_the_whole_layer() -> None:
    f = _frame(5)
    cfg = _cfg()
    B.touch(f, ["n0", "n1"], cfg)
    B.set_activation(f, {"n0": 0.9})
    s = B.summary(f, cfg)
    assert s["nodes"] == 5 and s["edges"] == 1
    assert s["active_count"] == 1
    assert s["top_active"][0]["node_id"] == "n0"
    assert s["at_risk_count"] >= 1, "aged memories must show up as at risk"
    assert set(s["states"]) == set(B.STATES)
    assert s["replay_buffer_size"] == 500


def test_state_defaults_for_legacy_rows() -> None:
    """Existing workspaces have no bio metadata; state must still work."""
    n = MemoryNode(concept="legacy")
    st = B.state(n)
    assert st["consolidation_state"] == "new" and st["memory_version"] == 1
    assert "bio" not in n.metadata, "reading state must not write metadata"
    assert n.metadata.get("document_id") is None


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)}/{len(tests)} biomira tests passed")


if __name__ == "__main__":
    _run_all()
