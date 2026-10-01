"""Self-check: POST /api/memory/consolidate (sleep pass over the live ws).

Uses FastAPI TestClient against a stubbed workspace so no model loads.
Verifies: (1) default is a DRY probe — the real replay/decay mechanisms run
against the in-memory frame but every mutated scalar is restored and nothing
is persisted, (2) apply=true persists decayed nodes, gist nodes + gist_of
edges, takes a backup and reloads the index, (3) invalid params are 400,
(4) the ring-0/1 placement guard refuses a collapsed workspace with 409.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402
from core.memory import MemoryEdge, MemoryFrame, MemoryNode, MemoryType  # noqa: E402


class FakeLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Cursor:
    """Mimics sqlite3's execute()-returns-a-cursor contract (replay_paths
    chains .fetchall() on the cursor, not the connection)."""

    def __init__(self, store, sql):
        self.store = store
        self.sql = sql

    def fetchall(self):
        # replay_paths reads retrieval_logs; nothing else selects
        return (self.store.log_rows
                if "FROM retrieval_logs" in self.sql else [])


class _Tx:
    def __init__(self, store):
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=()):
        self.store.last_sql = sql
        return _Cursor(self.store, sql)


class FakeStore:
    def __init__(self, log_rows=()):
        self.log_rows = list(log_rows)
        self.last_sql = ""
        self.upserts = []
        self.edges = []
        self.weight_updates = []

    def tx(self):
        return _Tx(self)

    def list_documents(self):
        return []

    def upsert_node(self, row):
        self.upserts.append(row)

    def add_edge(self, s, t, r="gist_of", w=0.8, c=0.5):
        self.edges.append((s, t, r, w, c))

    def update_edge_weight(self, s, t, w):
        self.weight_updates.append((s, t, w))


class FakeEmb:
    def encode(self, texts, **kw):
        return [np.full(4, 0.5) for _ in texts]


class FakeWS:
    def __init__(self, frame, log_rows=()):
        self.frame = frame
        self.store = FakeStore(log_rows)
        self.embeddings = FakeEmb()
        self.lock = FakeLock()
        self.reloads = 0

    def reload(self):
        self.reloads += 1


def node(nid, importance=0.5, ring=0, sector="alpha",
         mtype=MemoryType.FACT, emb=None):
    n = MemoryNode(concept=nid, memory_type=mtype, id=nid, ring=ring,
                   sector=sector, importance=importance, confidence=0.8,
                   created_at="2026-10-01T00:00:00",
                   updated_at="2026-10-01T00:00:00")
    if emb is not None:
        n.embedding = emb
    return n


def frame_of(nodes, edges=()):
    f = MemoryFrame()
    for n in nodes:
        f.add_node(n)
    for e in edges:
        f.add_edge(e)
    return f


LOG_ROWS = [("2026-10-01T09:00:00", "h1",
             '{"query":"q1","node_ids":["n1","n2"]}')]
URL = "/api/memory/consolidate"
client = TestClient(server.app)

# 1) dry run by default: mechanisms run, then every scalar is restored,
#    nothing persisted, no backup
f = frame_of(
    [node("n1"), node("n2")],
    [MemoryEdge(source_id="n1", target_id="n2", relation_type="related")])
fake = FakeWS(f, LOG_ROWS)
server._ws = fake

r = client.post(URL)
assert r.status_code == 200, r.text
body = r.json()
assert body["apply"] is False
assert body["note"] and "apply" in body["note"]
assert body["replay"]["seed_nodes"] == 2, body["replay"]
assert body["replay"]["paths_fired"] == 1
assert body["decay"]["nodes"] == 2
assert body["decay"]["dry_run"] is True
assert body["backup"] is None
assert fake.store.upserts == [] and fake.store.edges == []
assert fake.store.weight_updates == []  # hebbian ran dry (store=None)
assert fake.reloads == 0
assert f.nodes["n1"].importance == 0.5          # restored
assert f.nodes["n1"].updated_at == "2026-10-01T00:00:00"
assert f.edges[0].weight == 1.0                 # restored
print("PASS dry: mechanisms ran in-memory, scalars restored, nothing persisted")

# 2) apply=true: backup, persisted decay writes, gists + gist_of edges, reload
fact_emb = np.array([1.0, 0.1, 0.0, 0.0])
f2 = frame_of(
    [node(f"f{i}", emb=fact_emb * (1 - 0.01 * i)) for i in range(4)],
    [MemoryEdge(source_id="f1", target_id="f2", relation_type="related")])
fake2 = FakeWS(f2, [("2026-10-01T09:00:00", "h2",
                     '{"query":"q2","node_ids":["f1","f2"]}')])
server._ws = fake2

r2 = client.post(URL, json={"apply": True, "max_gists": 4})
assert r2.status_code == 200, r2.text
body2 = r2.json()
assert body2["apply"] is True
assert body2["decay"]["dry_run"] is False
assert body2["gists"]["gists"] == 1, body2["gists"]
assert body2["backup"].endswith("consolidate_backup_server.db")
assert len(fake2.store.upserts) >= 4            # decay write-back + gist row
assert len(fake2.store.edges) == 3              # gist_of edges (min_members=3)
assert fake2.reloads == 1
gist_nodes = [n for n in f2.nodes.values() if n.metadata.get("gist")]
assert len(gist_nodes) == 1 and gist_nodes[0].embedding is not None
assert gist_nodes[0].ring == 1                  # gists born one ring under core
print("PASS apply: backup taken, decay+gists persisted, gist embedded, "
      "index reloaded")

# 3) invalid params -> 400
server._ws = fake
for bad in ({"apply": "yes"}, {"half_life": "x"}, {"half_life": 0.5},
            {"replay_days": 0}, {"max_gists": -1}):
    rb = client.post(URL, json=bad)
    assert rb.status_code == 400, (bad, rb.status_code, rb.text)
print("PASS validation: bad apply/half_life/replay_days/max_gists rejected")

# 4) collapsed placement guard -> 409 (never sleep a mis-shaped workspace)
big = frame_of([node(f"g{i}", ring=4, mtype=MemoryType.FACT)
                for i in range(2500)])
fake3 = FakeWS(big, [])
server._ws = fake3
r3 = client.post(URL)
assert r3.status_code == 409, r3.status_code
print("PASS guard: ring-0/1 rate < 2% refused with 409")

server._ws = None
print("PASS consolidate endpoint: dry restores, apply persists, params "
      "validated, placement guarded")
