"""SQLite persistence (spec §15, §17).

Durable source of truth. FAISS owns vectors; NetworkX owns graph topology;
SQLite owns everything else: nodes, edges, documents, chunks, provenance,
experiments, retrieval logs. WAL mode for smoother concurrent UI reads.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional

from core.types import iso_now, new_id

logger = logging.getLogger("mira.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source_path TEXT,
    file_type TEXT,
    sha256 TEXT,
    n_chars INTEGER,
    n_chunks INTEGER DEFAULT 0,
    ingested_at TEXT,
    meta TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
    seq INTEGER,
    text TEXT NOT NULL,
    n_chars INTEGER,
    page INTEGER,
    meta TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    concept TEXT NOT NULL,
    summary TEXT DEFAULT '',
    raw_text TEXT DEFAULT '',
    memory_type TEXT NOT NULL,
    parent_id TEXT,
    ring INTEGER,
    sector TEXT,
    depth INTEGER,
    radial_distance REAL,
    importance REAL DEFAULT 0.5,
    confidence REAL DEFAULT 0.5,
    created_at TEXT,
    updated_at TEXT,
    valid_from TEXT,
    valid_until TEXT,
    embedding_id INTEGER,
    source_ids TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    status TEXT DEFAULT 'active'
);
CREATE TABLE IF NOT EXISTS node_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT,
    action TEXT,
    at TEXT,
    snapshot TEXT
);
CREATE TABLE IF NOT EXISTS edges (
    source_id TEXT,
    target_id TEXT,
    relation_type TEXT,
    weight REAL DEFAULT 1.0,
    confidence REAL DEFAULT 0.5,
    provenance TEXT DEFAULT '[]',
    created_at TEXT,
    PRIMARY KEY (source_id, target_id, relation_type)
);
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    created_at TEXT,
    name TEXT,
    config TEXT,
    results TEXT,
    git_commit TEXT,
    hardware TEXT,
    seed INTEGER
);
CREATE TABLE IF NOT EXISTS retrieval_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT,
    query_hash TEXT,
    system TEXT,
    n_candidates INTEGER,
    n_final INTEGER,
    latency_ms REAL,
    context_tokens INTEGER,
    meta TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(memory_type);
CREATE INDEX IF NOT EXISTS idx_nodes_sector ON nodes(sector);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
"""

_NODE_COLS = [
    "id", "concept", "summary", "raw_text", "memory_type", "parent_id",
    "ring", "sector", "depth", "radial_distance", "importance", "confidence",
    "created_at", "updated_at", "valid_from", "valid_until", "embedding_id",
    "source_ids", "metadata", "status",
]


class SQLiteStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        logger.info("sqlite ready", extra={"path": db_path})

    @contextmanager
    def tx(self):
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def close(self) -> None:
        self._conn.close()

    # ---- documents / chunks ----
    def add_document(self, title: str, source_path: str, file_type: str,
                     sha256: str, n_chars: int, meta: Dict | None = None) -> str:
        doc_id = new_id("doc")
        with self.tx() as c:
            c.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?,0,?,?)",
                (doc_id, title, source_path, file_type, sha256, n_chars, iso_now(),
                 json.dumps(meta or {}, ensure_ascii=False)),
            )
        return doc_id

    def add_chunk(self, document_id: str, seq: int, text: str,
                  page: Optional[int] = None, meta: Dict | None = None) -> str:
        chunk_id = new_id("chk")
        with self.tx() as c:
            c.execute(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
                (chunk_id, document_id, seq, text, len(text), page,
                 json.dumps(meta or {}, ensure_ascii=False)),
            )
        return chunk_id

    def set_document_chunk_count(self, document_id: str, n: int) -> None:
        with self.tx() as c:
            c.execute("UPDATE documents SET n_chunks=? WHERE id=?", (n, document_id))

    def get_document(self, document_id: str) -> Optional[Dict]:
        row = self._conn.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
        return dict(row) if row else None

    def list_documents(self) -> List[Dict]:
        rows = self._conn.execute("SELECT * FROM documents ORDER BY ingested_at DESC").fetchall()
        return [dict(r) for r in rows]

    def get_chunk(self, chunk_id: str) -> Optional[Dict]:
        row = self._conn.execute("SELECT * FROM chunks WHERE id=?", (chunk_id,)).fetchone()
        return dict(row) if row else None

    def document_chunks(self, document_id: str) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE document_id=? ORDER BY seq", (document_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def find_document_by_sha(self, sha256: str) -> Optional[Dict]:
        row = self._conn.execute("SELECT * FROM documents WHERE sha256=?", (sha256,)).fetchone()
        return dict(row) if row else None

    # ---- nodes ----
    def upsert_node(self, node: Dict) -> None:
        vals = [node.get(col) for col in _NODE_COLS]
        if not vals[_NODE_COLS.index("status")]:
            vals[_NODE_COLS.index("status")] = "active"
        if node.get("created_at") is None:
            vals[_NODE_COLS.index("created_at")] = iso_now()
        vals[_NODE_COLS.index("updated_at")] = iso_now()
        for jcol in ("source_ids", "metadata"):
            v = vals[_NODE_COLS.index(jcol)]
            if not isinstance(v, str):
                vals[_NODE_COLS.index(jcol)] = json.dumps(v or {}, ensure_ascii=False) if jcol == "metadata" \
                    else json.dumps(v or [], ensure_ascii=False)
        placeholders = ",".join("?" * len(_NODE_COLS))
        with self.tx() as c:
            c.execute(
                f"INSERT INTO nodes ({','.join(_NODE_COLS)}) VALUES ({placeholders}) "
                f"ON CONFLICT(id) DO UPDATE SET {','.join(f'{col}=excluded.{col}' for col in _NODE_COLS[1:])}",
                vals,
            )
            c.execute(
                "INSERT INTO node_history(node_id, action, at, snapshot) VALUES (?,?,?,?)",
                (node["id"], node.get("_action", "update"), iso_now(),
                 json.dumps({k: node.get(k) for k in _NODE_COLS}, ensure_ascii=False, default=str)),
            )

    def get_node(self, node_id: str) -> Optional[Dict]:
        row = self._conn.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["source_ids"] = json.loads(d.get("source_ids") or "[]")
        d["metadata"] = json.loads(d.get("metadata") or "{}")
        return d

    def all_nodes(self, include_inactive: bool = False) -> List[Dict]:
        q = "SELECT * FROM nodes" if include_inactive else "SELECT * FROM nodes WHERE status='active'"
        rows = self._conn.execute(q).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["source_ids"] = json.loads(d.get("source_ids") or "[]")
            d["metadata"] = json.loads(d.get("metadata") or "{}")
            out.append(d)
        return out

    def delete_node(self, node_id: str) -> None:
        with self.tx() as c:
            c.execute("UPDATE nodes SET status='deleted' WHERE id=?", (node_id,))
            c.execute("DELETE FROM edges WHERE source_id=? OR target_id=?", (node_id, node_id))

    def count_nodes(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM nodes WHERE status='active'").fetchone()[0]

    # ---- edges ----
    def add_edge(self, source_id: str, target_id: str, relation_type: str,
                 weight: float = 1.0, confidence: float = 0.5,
                 provenance: List[str] | None = None) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO edges VALUES (?,?,?,?,?,?,?)",
                (source_id, target_id, relation_type, weight, confidence,
                 json.dumps(provenance or [], ensure_ascii=False), iso_now()),
            )

    def all_edges(self) -> List[Dict]:
        rows = self._conn.execute("SELECT * FROM edges").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["provenance"] = json.loads(d.get("provenance") or "[]")
            out.append(d)
        return out

    def edges_of(self, node_id: str) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM edges WHERE source_id=? OR target_id=?", (node_id, node_id)
        ).fetchall()
        return [dict(r) for r in rows]

    def update_edge_weight(self, source_id: str, target_id: str, weight: float) -> None:
        """Persist a Hebbian weight change to every edge between the pair.
        A missing row is a no-op (the live edge lives only in the frame)."""
        with self.tx() as c:
            c.execute(
                "UPDATE edges SET weight=? WHERE source_id=? AND target_id=?",
                (float(weight), source_id, target_id))
            c.execute(
                "UPDATE edges SET weight=? WHERE source_id=? AND target_id=?",
                (float(weight), target_id, source_id))

    # ---- experiments / logs ----
    def save_experiment(self, name: str, config: Dict, results: Dict,
                        git_commit: str = "", hardware: Dict | None = None,
                        seed: int = 42) -> str:
        exp_id = new_id("exp")
        with self.tx() as c:
            c.execute(
                "INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?)",
                (exp_id, iso_now(), name, json.dumps(config, ensure_ascii=False, default=str),
                 json.dumps(results, ensure_ascii=False, default=str), git_commit,
                 json.dumps(hardware or {}, ensure_ascii=False), seed),
            )
        return exp_id

    def list_experiments(self) -> List[Dict]:
        rows = self._conn.execute("SELECT * FROM experiments ORDER BY created_at DESC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for jcol in ("config", "results", "hardware"):
                d[jcol] = json.loads(d.get(jcol) or "{}")
            out.append(d)
        return out

    def log_retrieval(self, query_hash: str, system: str, n_candidates: int,
                      n_final: int, latency_ms: float, context_tokens: int,
                      meta: Dict | None = None) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO retrieval_logs(at,query_hash,system,n_candidates,n_final,"
                "latency_ms,context_tokens,meta) VALUES (?,?,?,?,?,?,?,?)",
                (iso_now(), query_hash, system, n_candidates, n_final, latency_ms,
                 context_tokens, json.dumps(meta or {}, ensure_ascii=False)),
            )
