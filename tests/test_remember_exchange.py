"""Self-check: chat 'remember' writes exchanges into mandala memory.

Uses FastAPI TestClient against a stubbed workspace so no model loads.
Verifies: (1) remember=true ingests a '(conversation)' document with a
dedup-stable title, (2) remember=false skips it, (3) the response carries
the 'remembered' receipt.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402


class FakeAnswer:
    text = "Mandalas organize memory in concentric rings around a core."
    mode = "llm"
    agent_mode = "memory"
    memories = [{"id": "m1"}]
    selected_evidence_ids = ["m1"]
    path_labels = []
    sources = ["some-doc.pdf p.3"]
    source_refs = []
    metrics = {"latency_ms": 5.0, "n_memories": 1, "context_tokens": 40,
               "compression_ratio": 0.0}
    confidence_note = ""
    affect_snapshot = {}


class FakeLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeWS:
    def __init__(self):
        self.ingested = []
        self.lock = FakeLock()

    def ask(self, question, **kwargs):
        return FakeAnswer()

    def ingest_text(self, text, title="pasted text", source_path="(inline)"):
        self.ingested.append({"text": text, "title": title,
                              "source_path": source_path})
        return {"document_id": "doc_test123", "n_nodes": 4, "n_edges": 3}


fake = FakeWS()
server._ws = fake

client = TestClient(server.app)

# 1) remember=true -> ingested with provenance + stable title
r1 = client.post("/api/chat", json={
    "question": "What is the mandala memory architecture?",
    "components": "all", "remember": True})
assert r1.status_code == 200, r1.text
body1 = r1.json()
assert body1["remembered"]["document_id"] == "doc_test123"
assert len(fake.ingested) == 1
assert fake.ingested[0]["source_path"] == "(conversation)"
assert fake.ingested[0]["title"].startswith("Conversation: What is the mandala")
assert "Question: What is the mandala memory architecture?" in fake.ingested[0]["text"]
assert "Answer (memory):" in fake.ingested[0]["text"]

# 2) same question again -> same dedup title, still saves (replace semantics
#    are handled by the ingestion pipeline's title-keyed storage)
r2 = client.post("/api/chat", json={
    "question": "What is the mandala memory architecture?",
    "components": "all", "remember": True})
assert r2.json()["remembered"]["n_nodes"] == 4
assert len(fake.ingested) == 2
assert fake.ingested[1]["title"] == fake.ingested[0]["title"]

# 3) remember=false -> no ingest
r3 = client.post("/api/chat", json={
    "question": "Another question entirely", "components": "all",
    "remember": False})
assert r3.json()["remembered"] is None
assert len(fake.ingested) == 2

print("PASS remember: ingest on true, skipped on false, dedup-stable title, "
      "receipt in response")
