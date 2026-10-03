"""Self-check: the BioMIRA HTTP surface (spec §18-19, §1).

FastAPI TestClient against a stubbed workspace, so no model loads. Verifies
(1) /api/biomira/state reports the layer and its aggregates, (2) the memory
table exposes the biology columns, (3) /api/node/{id} carries a "why" block,
(4) a step is a DRY run by default and refuses to write, (5) the step is a
409 while the layer is off, (6) the toggle rebuilds retrieval weights so the
stability component actually takes effect, (7) /api/lab/forgetting says
"not measured" rather than inventing numbers.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402
from core import biomira as B  # noqa: E402
from core.memory import MemoryEdge, MemoryFrame, MemoryNode, MemoryType  # noqa: E402


class FakeLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeStore:
    """Records every write so 'dry run wrote nothing' is checkable."""

    def __init__(self):
        self.rows = []

    def upsert_node(self, row):
        self.rows.append(row)

    def add_edge(self, edge):
        self.rows.append(edge)

    def get_node(self, nid):
        return self.row_for(nid)

    def get_chunk(self, cid):
        return None

    def get_document(self, did):
        return None

    def row_for(self, nid):
        return self.rows_map.get(nid)

    rows_map: dict = {}


class FakeWS:
    def __init__(self, enabled: bool = True):
        self.config = {"biomira": {"enabled": enabled},
                        "retrieval_score": {}}
        self.lock = FakeLock()
        self.store = FakeStore()
        self.embeddings = None
        self.answer_pipeline = None
        now = datetime.utcnow().isoformat(timespec="seconds")
        old = (datetime.utcnow() - timedelta(days=90)).isoformat(timespec="seconds")
        self.frame = MemoryFrame()
        for i in range(4):
            nid = f"n{i}"
            node = MemoryNode(
                id=nid, concept=f"concept {i}", memory_type=MemoryType.FACT,
                summary=f"summary {i}", ring=2, sector="alpha", depth=1,
                importance=0.4 + 0.1 * i, confidence=0.7,
                created_at=old, updated_at=old,
                embedding=np.asarray([1.0, float(i) * 0.01], dtype=np.float32),
            )
            node.metadata["lab_task"] = 0
            self.frame.add_node(node)
            FakeStore.rows_map[nid] = {**node.to_row(), "_action": "update"}
        self.frame.add_edge(MemoryEdge("n0", "n1"))
        self.frame.add_edge(MemoryEdge("n1", "n2"))
        self.frame.add_edge(MemoryEdge("n2", "n3"))
        self.vs = None
        from storage.graph_store import GraphStore
        self.gs = GraphStore()
        self.gs.build_from([n.to_row() for n in self.frame.nodes.values()],
                           [e.to_row() for e in self.frame.edges])

    def reload(self):
        return None


fake = FakeWS(enabled=True)
server._ws = fake
client = TestClient(server.app)

# --- 1) state --------------------------------------------------------------
state = client.get("/api/biomira/state")
assert state.status_code == 200, state.text
body = state.json()
assert body["enabled"] is True
assert body["nodes"] == 4, body
assert set(("states", "replay_buffer_size", "mean_activation")) <= set(body), sorted(body)
assert sum(body["states"].values()) == 4, body["states"]

# --- 2) memory table carries the biology columns --------------------------
mem = client.get("/api/biomira/memories?sort=retention&limit=10")
assert mem.status_code == 200, mem.text
rows = mem.json()["memories"]
assert len(rows) == 4
for key in ("state", "activation", "stability", "retention", "access_count", "version"):
    assert key in rows[0], sorted(rows[0])
# sort honoured: retention, most-retained first
rets = [r["retention"] for r in rows]
assert rets == sorted(rets, reverse=True), rets

# a state filter is honoured rather than ignored
only_new = client.get("/api/biomira/memories?state=new").json()
assert all(r["state"] == "new" for r in only_new["memories"]), only_new

# --- 3) 'why was this retrieved' block on the node endpoint ---------------
node = client.get("/api/node/n0")
assert node.status_code == 200, node.text
why = node.json()["biomira"]
assert why["enabled"] is True
for key in ("why_decayed", "why_consolidated", "consolidation_state", "stability"):
    assert key in why, sorted(why)
assert "never retrieved" in why["why_decayed"], why["why_decayed"]

# --- 4) step is a dry run unless apply=true -------------------------------
# NOTE: each claim gets a fresh workspace. Replay marks memories as recently
# used, which legitimately lifts their retention to ~1.0 and makes decay a
# no-op — so reusing one frame would test nothing.
def _fresh(enabled: bool = True):
    global fake
    fake = FakeWS(enabled=enabled)
    server._ws = fake
    return fake

ws0 = _fresh()
dry = client.post("/api/biomira/step", json={"actions": ["decay", "replay"]})
assert dry.status_code == 200, dry.text
report = dry.json()
assert report["applied"] is False
assert report["decay"]["nodes"] == 4, report["decay"]
assert report["decay"]["dry_run"] is True
assert report["replay"]["replayed"] >= 1, report["replay"]
assert ws0.store.rows == [], "dry run must not write"

ws1 = _fresh()
before = {n.id: n.importance for n in ws1.frame.nodes.values()}
applied = client.post("/api/biomira/step",
                      json={"actions": ["decay"], "apply": True})
assert applied.status_code == 200, applied.text
assert applied.json()["applied"] is True
assert applied.json()["decay"]["dry_run"] is False
assert ws1.store.rows, "apply=true must persist the decay pass"
assert all(n.importance < before[n.id]
           for n in ws1.frame.nodes.values()), "decay must actually lower importance"

# --- 5) the layer is a hard gate ------------------------------------------
off = client.post("/api/biomira/step", json={"actions": ["decay"]})
assert off.status_code == 200
fake.config["biomira"]["enabled"] = False
blocked = client.post("/api/biomira/step", json={"actions": ["decay"]})
assert blocked.status_code == 409, blocked.text
assert client.get("/api/biomira/state").json()["enabled"] is False
# with the layer off the node endpoint must say so instead of inventing stats
assert client.get("/api/node/n0").json()["biomira"] == {"enabled": False}

# --- 6) toggling rebuilds the retriever so weights take effect ------------
toggle = client.post("/api/biomira/enabled",
                     json={"enabled": True, "kappa_stability": 0.2})
assert toggle.status_code == 200, toggle.text
assert toggle.json()["enabled"] is True
assert toggle.json()["kappa_stability"] == 0.2
assert fake.config["retrieval_score"]["kappa_stability"] == 0.2
assert client.get("/api/biomira/state").json()["enabled"] is True

# --- 7) the lab never invents numbers -------------------------------------
lab = client.get("/api/lab/forgetting")
assert lab.status_code == 200, lab.text
if not lab.json()["available"]:
    assert lab.json()["reason"], "unavailable must say what to run"
    assert lab.json().get("variants", {}) == {}

print("PASS biomira endpoints: state, memory columns, why-block, dry-run "
      "writes nothing, apply persists, 409 when off, toggle rebuilds "
      "weights, lab reports absence honestly")