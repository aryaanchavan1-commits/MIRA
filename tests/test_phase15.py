"""Phase 15 tests: consent-gated web search + Hebbian consolidation.

Network calls are never made — only pure parsing/logic functions run here.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.hebbian import reinforce
from core.memory import MemoryEdge, MemoryFrame, MemoryNode, MemoryType


# ---- websearch (pure parts, no network) ----
def test_websearch_consent_gate():
    from tools.websearch import search
    out = search("anything", config={}, allow_web=False)
    assert out["results"] == [] and "disabled" in out["note"]
    # enabled via explicit consent flag must at least attempt (offline CI:
    # backends fail gracefully, note explains) — we only assert shape
    out2 = search("anything", config={"web_search": {"enabled": True}},
                  allow_web=False, backend="duckduckgo")
    assert set(out2.keys()) == {"backend", "results", "note"}


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
