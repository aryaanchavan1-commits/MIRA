"""Memory update orchestrator (spec §19).

Bridges core.memory.apply_update (pure logic) with SQLiteStore (history).
Every action is recorded in node_history; merges are conflict-checked
against existing facts first (§18). No blind overwrites.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.conflicts import detect_conflict
from core.memory import MemoryNode, apply_update
from storage.sqlite_store import SQLiteStore

logger = logging.getLogger("mira.updater")


class MemoryUpdater:
    def __init__(self, store: SQLiteStore):
        self.store = store

    def _load(self, node_id: str) -> Optional[MemoryNode]:
        row = self.store.get_node(node_id)
        return MemoryNode.from_row(row) if row else None

    def create(self, node: MemoryNode) -> str:
        self.store.upsert_node({**node.to_row(), "_action": "create"})
        return node.id

    def update(self, node_id: str, patch: Dict[str, Any]) -> Optional[str]:
        existing = self._load(node_id)
        if existing is None:
            return None
        updated = apply_update(existing, "update", patch)
        self.store.upsert_node({**updated.to_row(), "_action": "update"})
        return node_id

    def merge(self, node_id: str, incoming: MemoryNode) -> Optional[str]:
        existing = self._load(node_id)
        if existing is None:
            return None
        # §18: if the incoming fact contradicts the target, do not merge — keep both
        if detect_conflict(existing, incoming):
            logger.info("merge blocked: conflicting fact kept separate",
                        extra={"target": node_id, "incoming": incoming.id})
            self.create(incoming)
            self.store.add_edge(node_id, incoming.id, "contradicts",
                                weight=0.9, confidence=incoming.confidence)
            return incoming.id
        merged = apply_update(existing, "merge", {
            "raw_text": incoming.raw_text,
            "summary": incoming.summary or existing.summary,
            "source_ids": incoming.source_ids,
            "importance": incoming.importance,
            "confidence": incoming.confidence,
            "metadata": incoming.metadata,
        })
        self.store.upsert_node({**merged.to_row(), "_action": "merge"})
        return node_id

    def split(self, node_id: str, parts: List[MemoryNode],
              reason: str = "") -> List[str]:
        original = self._load(node_id)
        if original is None:
            return []
        marked = apply_update(original, "split", {"reason": reason})
        self.store.upsert_node({**marked.to_row(), "_action": "split"})
        created = [self.create(p) for p in parts]
        for pid in created:
            self.store.add_edge(node_id, pid, "split_from", weight=1.0,
                                confidence=original.confidence)
        return created

    def invalidate(self, node_id: str, reason: str = "") -> Optional[str]:
        existing = self._load(node_id)
        if existing is None:
            return None
        updated = apply_update(existing, "invalidate", {"reason": reason})
        self.store.upsert_node({**updated.to_row(), "_action": "invalidate"})
        return node_id

    def archive(self, node_id: str) -> Optional[str]:
        existing = self._load(node_id)
        if existing is None:
            return None
        updated = apply_update(existing, "archive")
        self.store.upsert_node({**updated.to_row(), "_action": "archive"})
        return node_id

    def restore(self, node_id: str) -> Optional[str]:
        existing = self._load(node_id)
        if existing is None:
            return None
        updated = apply_update(existing, "restore")
        self.store.upsert_node({**updated.to_row(), "_action": "restore"})
        return node_id

    def history(self, node_id: str) -> List[Dict]:
        rows = self.store._conn.execute(
            "SELECT action, at, snapshot FROM node_history WHERE node_id=? ORDER BY at",
            (node_id,),
        ).fetchall()
        return [dict(r) for r in rows]
