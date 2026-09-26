"""FAISS vector store (spec §13).

Flat inner-product index over normalized vectors (equivalent to cosine).
O(n) rebuild on delete — fine at laptop scale; IVF with remove_ids is the
upgrade path if the corpus grows large.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("mira.vector")

try:
    import faiss
except Exception as exc:  # pragma: no cover
    faiss = None
    logger.warning("faiss unavailable: %s — vector search disabled", exc)


class VectorStore:
    def __init__(self, dim: int, persist_path: Optional[str] = None):
        if faiss is None:
            raise RuntimeError("faiss is not installed")
        self.dim = dim
        self.persist_path = persist_path
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self.id_meta: Dict[int, Dict] = {}   # faiss int id -> {node_id, chunk_id, ...}
        self._next = 0

    # ---- core ops ----
    def add(self, vectors: np.ndarray, metas: List[Dict]) -> List[int]:
        """vectors: (n, dim) float32, unnormalized — normalized here."""
        if len(metas) != vectors.shape[0]:
            raise ValueError("vectors/metas length mismatch")
        vecs = np.ascontiguousarray(vectors, dtype="float32")
        faiss.normalize_L2(vecs)
        ids = list(range(self._next, self._next + len(metas)))
        self.index.add_with_ids(vecs, np.asarray(ids, dtype="int64"))
        for i, meta in zip(ids, metas):
            self.id_meta[i] = meta
        self._next += len(metas)
        return ids

    def delete(self, faiss_ids: List[int]) -> int:
        before = self.index.ntotal
        self.index.remove_ids(np.asarray(faiss_ids, dtype="int64"))
        removed = before - self.index.ntotal
        for i in faiss_ids:
            self.id_meta.pop(i, None)
        return removed

    def update(self, faiss_id: int, vector: np.ndarray, meta: Dict) -> None:
        self.delete([faiss_id])
        self.add(vector.reshape(1, -1), [meta])

    def search(self, query: np.ndarray, k: int = 8) -> List[Tuple[float, Dict]]:
        if self.index.ntotal == 0:
            return []
        q = np.ascontiguousarray(query.reshape(1, -1).astype("float32"))
        faiss.normalize_L2(q)
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(q, k)
        out = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            out.append((float(score), self.id_meta.get(int(idx), {})))
        return out

    def batch_search(self, queries: np.ndarray, k: int = 8) -> List[List[Tuple[float, Dict]]]:
        if self.index.ntotal == 0:
            return [[] for _ in range(queries.shape[0])]
        q = np.ascontiguousarray(queries.astype("float32"))
        faiss.normalize_L2(q)
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(q, k)
        results = []
        for row_s, row_i in zip(scores, ids):
            results.append([
                (float(s), self.id_meta.get(int(i), {}))
                for s, i in zip(row_s, row_i) if i != -1
            ])
        return results

    def size(self) -> int:
        return int(self.index.ntotal)

    # ---- persistence ----
    def save(self) -> None:
        if not self.persist_path:
            return
        os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
        faiss.write_index(self.index, self.persist_path)
        with open(self.persist_path + ".meta.json", "w", encoding="utf-8") as fh:
            json.dump({"dim": self.dim, "_next": self._next,
                       "id_meta": {str(k): v for k, v in self.id_meta.items()}}, fh)

    def save_atomic(self) -> None:
        if not self.persist_path:
            return
        path = self.persist_path
        meta_path = path + ".meta.json"
        token = uuid.uuid4().hex
        temp_path = f"{path}.{token}.tmp"
        temp_meta = temp_path + ".meta.json"
        backup_path = f"{path}.{token}.bak"
        backup_meta = backup_path + ".meta.json"
        had_index = os.path.exists(path)
        had_meta = os.path.exists(meta_path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        try:
            if had_index:
                shutil.copy2(path, backup_path)
            if had_meta:
                shutil.copy2(meta_path, backup_meta)
            faiss.write_index(self.index, temp_path)
            with open(temp_meta, "w", encoding="utf-8") as fh:
                json.dump({"dim": self.dim, "_next": self._next,
                           "id_meta": {str(k): v for k, v in self.id_meta.items()}}, fh)
            os.replace(temp_path, path)
            os.replace(temp_meta, meta_path)
        except Exception:
            try:
                if had_index:
                    os.replace(backup_path, path)
                elif os.path.exists(path):
                    os.remove(path)
                if had_meta:
                    os.replace(backup_meta, meta_path)
                elif os.path.exists(meta_path):
                    os.remove(meta_path)
            except Exception:
                logger.exception("failed to restore vector index")
            raise
        finally:
            for temp_file in (temp_path, temp_meta, backup_path, backup_meta):
                try:
                    os.remove(temp_file)
                except FileNotFoundError:
                    pass

    @classmethod
    def load(cls, persist_path: str) -> "VectorStore":
        if not os.path.exists(persist_path):
            raise FileNotFoundError(persist_path)
        index = faiss.read_index(persist_path)
        with open(persist_path + ".meta.json", "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        store = cls.__new__(cls)
        store.dim = meta["dim"]
        store.persist_path = persist_path
        store.index = index
        store._next = meta.get("_next", index.ntotal)
        store.id_meta = {int(k): v for k, v in meta.get("id_meta", {}).items()}
        return store
