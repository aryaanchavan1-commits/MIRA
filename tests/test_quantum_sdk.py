"""Quantum-inspired backend + SDK tests (spec §20/§30): kernel separation,
the interference property that H6 predicts, default-off, and SDK read path."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core.quantum_inspired import BornRuleIndex, _amplitudes
from tests.test_ncm import constellation_world, _embed, TOPICS

FAILURES = []


def check(name, fn):
    try:
        fn()
        print(f"PASS {name}")
    except Exception as exc:  # noqa: BLE001 — harness must not die early
        FAILURES.append((name, exc))
        print(f"FAIL {name}: {exc}")


def test_born_kernel_separates():
    frame = constellation_world()
    emb = _embed(frame)
    b = BornRuleIndex(top_k=2)
    assert b.fit(frame), b._fit_info
    assert all(len(b.amp_of[n]) == len(b.members[n]) for n in b.members)
    q = np.asarray(emb.encode([TOPICS[1] + " note 0"])[0])
    ids, w = b.query_membership(q)
    same = b.node_score_born("n1_0", ids, _amplitudes(w))
    far = b.node_score_born("n3_2", ids, _amplitudes(w))
    assert 0 <= far < same <= 1, (same, far)


def test_interference_boost_is_quantified():
    """H6's mechanism, as a unit: agreement on TWO constellations must beat
    single-constellation agreement by MORE under the Born kernel than under
    the linear kernel — that is what 'interference-like' means here."""
    b = BornRuleIndex(top_k=2)
    b.members = {"multi": ("c000", "c001"), "single": ("c000",)}
    b.amp_of = {"multi": (0.7, 0.7), "single": (0.99,)}
    q_ids, q_w = ("c000", "c001"), (0.5, 0.5)
    born_multi = b.node_score("multi", q_ids, q_w)
    born_single = b.node_score("single", q_ids, q_w)
    # linear kernel on the same memberships
    lin_multi = 0.5 * 0.7 + 0.5 * 0.7
    lin_single = 0.5 * 0.99
    assert born_multi / born_single > lin_multi / lin_single, \
        (born_multi, born_single, lin_multi, lin_single)


def test_born_matches_linear_for_single_constellation():
    """With membership on one constellation the kernels agree — the
    hypothesis is specifically about spread memberships."""
    b = BornRuleIndex(top_k=1)
    b.members = {"x": ("c000",)}
    b.amp_of = {"x": (1.0,)}
    q_w = (1.0,)
    assert abs(b.node_score("x", ("c000",), q_w) - 1.0) < 1e-9


def test_default_off():
    from core.ncm import ncm_config
    from core.quantum_inspired import make_membership_index
    frame = constellation_world()
    _embed(frame)
    assert make_membership_index(frame, {}) is None            # disabled
    assert ncm_config({})["enabled"] is False
    idx = make_membership_index(frame, {"ncm": {"enabled": True}})
    assert idx is not None and not isinstance(idx, BornRuleIndex)  # linear default
    idx_b = make_membership_index(frame, {"ncm": {"enabled": True,
                                                  "backend": "born"}})
    assert isinstance(idx_b, BornRuleIndex)


def test_sdk_explanations_carry_all_components():
    """SDK retrieve() on the real store: read-only, skips when store empty."""
    try:
        from mira_sdk import MemoryEngine
        eng = MemoryEngine()
    except Exception as exc:  # noqa: BLE001 — env without a store
        print(f"  (skipped: no local store — {exc})")
        return
    try:
        s = eng.stats()
        if s.get("nodes", 0) == 0:
            print("  (skipped: empty store)")
            return
        r = eng.retrieve("memory consolidation", k=3)
        assert r["results"], "expected hits from a populated store"
        exp = r["results"][0]["explanation"]
        assert "constellation" in exp and len(exp) == 11
        assert set(r["results"][0]) >= {"id", "text", "score", "explanation"}
    finally:
        eng.close()


def test_sdk_audit_record():
    """S6: audit() returns the full justification record (read-only)."""
    try:
        from mira_sdk import MemoryEngine
        eng = MemoryEngine(load_llm=False)
    except Exception as exc:  # noqa: BLE001 — env without a store
        print(f"  (skipped: no local store — {exc})")
        return
    try:
        s = eng.stats()
        if s.get("nodes", 0) == 0:
            print("  (skipped: empty store)")
            return
        a = eng.audit("memory consolidation", k=3)
        audit = a["audit"]
        assert a["results"], "audit must include the retrieval results"
        assert len(audit["weights_in_effect"]) == 11
        assert audit["store"].get("nodes", 0) > 0
        assert audit["at"] and "T" in audit["at"]          # ISO timestamp
        # version_store may be None (chain not yet created) or a real summary
        vs = audit["version_store"]
        assert vs is None or (vs["append_only"] is True
                              and vs["records"] >= vs["subjects"] >= 0)
    finally:
        eng.close()


def run_all() -> int:
    check("born_kernel_separates", test_born_kernel_separates)
    check("interference_boost_is_quantified", test_interference_boost_is_quantified)
    check("born_matches_linear_for_single_constellation",
          test_born_matches_linear_for_single_constellation)
    check("default_off", test_default_off)
    check("sdk_explanations_carry_all_components",
          test_sdk_explanations_carry_all_components)
    check("sdk_audit_record", test_sdk_audit_record)
    print(f"{6 - len(FAILURES)}/6 passed")
    if FAILURES:
        for name, exc in FAILURES:
            print(f"  {name}: {exc}")
        return 1
    print("QUANTUM+SDK TESTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_all())
