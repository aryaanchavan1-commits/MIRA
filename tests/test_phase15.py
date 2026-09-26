"""Phase 15 tests: consent-gated web search + Hebbian consolidation.

Network calls are never made — only pure parsing/logic functions run here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.hebbian import W_MAX, W_MIN, reinforce, stdp_update
from core.memory import MemoryEdge, MemoryFrame, MemoryNode, MemoryType


# ---- websearch (pure parts, no network) ----
def test_websearch_consent_gate():
    from tools.websearch import search
    out = search("anything", config={}, allow_web=False)
    assert out["results"] == [] and "disabled" in out["note"]
    # enabled via explicit consent flag must at least attempt (offline CI:
    # backends fail gracefully, note explains) — we only assert shape
    out2 = search("anything",
                  config={"offline": False, "web_search": {"enabled": True}},
                  allow_web=False, backend="duckduckgo")
    assert set(out2.keys()) == {"backend", "results", "note"}


def test_websearch_requires_literal_consent_and_offline_default():
    from tools.websearch import search

    with patch("tools.websearch._duckduckgo",
               return_value=[{"url": "https://example.test"}]) as backend:
        assert search("q", config={}, allow_web=True,
                      backend="duckduckgo")["results"] == []
        assert backend.call_count == 0
        assert search("q", config={"offline": True,
                                    "web_search": {"enabled": True}},
                      allow_web=True, backend="duckduckgo")["results"] == []
        assert backend.call_count == 0

    for malformed in ("true", 1, []):
        with patch("tools.websearch._duckduckgo") as backend:
            try:
                search("q", config={"offline": False}, allow_web=malformed,
                       backend="duckduckgo")
            except ValueError:
                pass
            else:
                raise AssertionError("malformed consent must be rejected")
            assert backend.call_count == 0

    with patch("tools.websearch._duckduckgo",
               return_value=[{"url": "https://example.test"}]) as backend:
        out = search("q", config={"offline": False}, allow_web=True,
                     backend="duckduckgo")
        assert out["results"] and backend.call_count == 1


def test_websearch_parse_jsonish():
    from tools.websearch import _parse_jsonish
    raw = '{"url": "https://x.com/a", "title": "A", "snippet": "s"}\n' \
          '{"url": "https://x.com/b", "title": "B"}\nnoise line\n'
    res = _parse_jsonish(raw)
    assert len(res) == 2 and res[0]["url"].startswith("https://x.com/a")
    res2 = _parse_jsonish("see https://example.com/x and https://y.io/p.")
    assert [r["url"] for r in res2] == ["https://example.com/x", "https://y.io/p"]


# ---- Hebbian consolidation ----
def _frame() -> MemoryFrame:
    f = MemoryFrame()
    for nid in ("a", "b", "c"):
        f.add_node(MemoryNode(id=nid, concept=nid, memory_type=MemoryType.FACT,
                              summary=f"{nid} content", source_ids=["cx"]))
    f.add_edge(MemoryEdge(source_id="a", target_id="b", relation_type="rel"))
    f.add_edge(MemoryEdge(source_id="b", target_id="c", relation_type="rel"))
    f.add_edge(MemoryEdge(source_id="a", target_id="c", relation_type="other"))
    return f


def test_hebbian_strenghtens_used_paths():
    f = _frame()
    changed = reinforce(f, [["a", "b", "c"]], lr=0.2, decay=0.02)
    ab = f.get_edge("a", "b")
    ac = f.get_edge("a", "c")
    assert ab.weight > 1.0, "used edge must strengthen"
    assert ac.weight < 1.0, "unused edge must decay (homeostasis)"
    assert ("a", "b") in [(a, b) for a, b, _ in changed]


def test_hebbian_bounded_and_persisted():
    f = _frame()
    for _ in range(30):
        reinforce(f, [["a", "b"]], lr=0.5, decay=0.0)
    ab = f.get_edge("a", "b")
    assert ab.weight <= 2.0, "weights must stay bounded"

    import tempfile
    from storage.sqlite_store import SQLiteStore
    td = tempfile.mkdtemp()
    st = SQLiteStore(os.path.join(td, "t.db"))
    f2 = _frame()
    st.upsert_node(f2.nodes["a"].to_row())
    st.upsert_node(f2.nodes["b"].to_row())
    e = f2.get_edge("a", "b")
    st.add_edge(e.source_id, e.target_id, e.relation_type, weight=e.weight,
                confidence=e.confidence)
    reinforce(f2, [["a", "b"]], lr=0.2, store=st)
    rows = st.all_edges()
    assert any(abs(r["weight"] - f2.get_edge("a", "b").weight) < 1e-6
               for r in rows), "edge weight not persisted"
    st.close()  # before dir cleanup: WAL sidecars lock the dir on Windows
    import shutil
    shutil.rmtree(td, ignore_errors=True)


def test_stdp_trace_respects_order_window_and_bounds():
    forward = _frame()
    before = forward.get_edge("a", "b").weight
    changed = stdp_update(
        forward,
        [{"node": "a", "tick": 0, "spike": True},
         {"node": "b", "tick": 1, "spike": True}],
        lr=0.2, depression=0.1, window=1, decay=0.0,
    )
    assert changed and forward.get_edge("a", "b").weight > before
    assert W_MIN <= forward.get_edge("a", "b").weight <= W_MAX

    reverse = _frame()
    before = reverse.get_edge("a", "b").weight
    stdp_update(
        reverse,
        [{"node": "b", "tick": 0, "spike": True},
         {"node": "a", "tick": 1, "spike": True}],
        lr=0.2, depression=0.1, window=1, decay=0.0,
    )
    assert reverse.get_edge("a", "b").weight < before

    outside = _frame()
    weights = [e.weight for e in outside.edges]
    assert not stdp_update(
        outside,
        [{"node": "a", "tick": 0, "spike": True},
         {"node": "b", "tick": 3, "spike": True}],
        window=1, decay=0.0,
    )
    assert [e.weight for e in outside.edges] == weights

    capped = _frame()
    capped.get_edge("a", "b").weight = W_MAX
    stdp_update(
        capped,
        [{"node": "a", "tick": 0, "spike": True},
         {"node": "b", "tick": 1, "spike": True}],
        lr=0.5, window=1,
    )
    assert capped.get_edge("a", "b").weight == W_MAX

    floor = _frame()
    floor.get_edge("a", "b").weight = W_MIN
    stdp_update(
        floor,
        [{"node": "b", "tick": 0, "spike": True},
         {"node": "a", "tick": 1, "spike": True}],
        depression=0.5, window=1,
    )
    assert floor.get_edge("a", "b").weight == W_MIN


def test_stdp_is_deterministic_for_the_same_trace():
    trace = [{"node": "a", "tick": 0, "spike": True},
             {"node": "b", "tick": 1, "spike": True}]
    first, second = _frame(), _frame()
    changes_first = stdp_update(first, trace, window=1, decay=0.01)
    changes_second = stdp_update(second, trace, window=1, decay=0.01)
    assert changes_first == changes_second
    assert [e.weight for e in first.edges] == [e.weight for e in second.edges]


def test_stdp_persists_strengthening_and_homeostatic_decay():
    class Store:
        def __init__(self):
            self.updates = []

        def update_edge_weight(self, source, target, weight):
            self.updates.append((source, target, weight))

    frame = _frame()
    frame.add_node(MemoryNode(id="d", concept="d", memory_type=MemoryType.FACT,
                              summary="d content", source_ids=["cx"]))
    frame.add_edge(MemoryEdge("c", "d", relation_type="rel"))
    store = Store()
    changes = stdp_update(
        frame,
        [{"node": "a", "tick": 0, "spike": True},
         {"node": "b", "tick": 1, "spike": True}],
        lr=0.1, decay=0.1, window=1, store=store,
    )
    changed_pairs = {(a, b) for a, b, _ in changes}
    assert ("a", "b") in changed_pairs
    assert ("c", "d") in changed_pairs
    assert {(a, b) for a, b, _ in store.updates} >= changed_pairs


def test_legacy_reinforce_persists_decayed_edges_without_changing_return_api():
    class Store:
        def __init__(self):
            self.updates = []

        def update_edge_weight(self, source, target, weight):
            self.updates.append((source, target, weight))

    frame = _frame()
    store = Store()
    reported = reinforce(frame, [["a", "b"]], lr=0.1, decay=0.1, store=store)
    assert all(a == "a" and b == "b" for a, b, _ in reported)
    assert any((a, b) == ("a", "c") for a, b, _ in store.updates)
    assert any((a, b) == ("a", "b") for a, b, _ in store.updates)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
