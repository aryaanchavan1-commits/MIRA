"""MIRA retrieval core (spec §20-24).

Multi-component scoring with per-component ablation: any component can be
switched off via `active_components`. The full system = all 9 components on.

score = α·semantic + β·structural + γ·radial + δ·graph
      + ε·importance + ζ·confidence + η·recency + θ·path + ι·activation
                                                                 (§21, experimental)
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.activation import SpreadingActivation
from core.memory import MemoryFrame, MemoryNode
from core.types import parse_float, stable_hash, utcnow, datetime
from storage.graph_store import GraphStore
from storage.vector_store import VectorStore

logger = logging.getLogger("mira.retrieval")

ALL_COMPONENTS = ("semantic", "structural", "radial", "graph",
                  "importance", "confidence", "recency", "path", "activation")


@dataclass
class RetrievedItem:
    node: MemoryNode
    score: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    path: List[str] = field(default_factory=list)   # node ids from seed to evidence


@dataclass
class RetrievalResult:
    items: List[RetrievedItem] = field(default_factory=list)
    paths: List[List[str]] = field(default_factory=list)
    latency_ms: float = 0.0
    n_candidates: int = 0
    query: str = ""
    # Present for the optional LIF-inspired activation path; empty for the
    # backward-compatible continuous path or when activation is ablated.
    activation_trace: List[Dict[str, Any]] = field(default_factory=list)

    def path_labels(self) -> List[str]:
        return [" -> ".join(p) for p in self.paths]


class MIRARetriever:
    def __init__(self, frame: MemoryFrame, vector_store: VectorStore,
                 graph_store: GraphStore, config: Dict[str, Any]):
        self.frame = frame
        self.vs = vector_store
        self.gs = graph_store
        self.config = config
        rc = config.get("retrieval", {})
        self.candidate_k = int(rc.get("candidate_k", 24))
        self.final_k = int(rc.get("final_k", 8))
        self.max_hops = int(rc.get("max_hops", 3))
        w = config.get("retrieval_score", {})
        self.weights = {
            "semantic": parse_float(w.get("alpha_semantic", 0.35), 0.35),
            "structural": parse_float(w.get("beta_structural", 0.15), 0.15),
            "radial": parse_float(w.get("gamma_radial", 0.15), 0.15),
            "graph": parse_float(w.get("delta_graph", 0.15), 0.15),
            "importance": parse_float(w.get("epsilon_importance", 0.08), 0.08),
            "confidence": parse_float(w.get("zeta_confidence", 0.07), 0.07),
            "recency": parse_float(w.get("eta_recency", 0.03), 0.03),
            "path": parse_float(w.get("theta_path", 0.02), 0.02),
            "activation": parse_float(w.get("iota_activation", 0.12), 0.12),
        }
        self.activation = SpreadingActivation(frame, config)
        # optional learned weights (§new neural unit) — off by default; run
        # scripts/train_neural.py to fit and set retrieval_score.learned: true
        if w.get("learned"):
            from config.auto_config import DATA_DIR
            from core.neural import DeltaRuleScorer
            scorer = DeltaRuleScorer.load(os.path.join(DATA_DIR, "neural_scorer.json"))
            if scorer is not None:
                self.weights.update(scorer.weights())
                logger.info("using learned retrieval weights: %s", self.weights)
            else:
                logger.warning("retrieval_score.learned=true but no trained "
                               "scorer found — using hand weights")
        self.centrality = graph_store.centrality()
        self._node_index = {n.id: n for n in frame.nodes.values()}

    # ------------------------------------------------------------------
    # component scores, each normalized to [0, 1]
    # ------------------------------------------------------------------
    def _semantic(self, qvec: np.ndarray, node: MemoryNode) -> float:
        """Cosine similarity — dot products are only meaningful for
        normalized embeddings; the hashing fallback's are not."""
        if self.vs is None or node.embedding is None or qvec is None:
            return 0.0
        qn = np.linalg.norm(qvec)
        nn = np.linalg.norm(node.embedding)
        if qn < 1e-9 or nn < 1e-9:
            return 0.0
        return float(np.clip(np.dot(qvec, node.embedding) / (qn * nn), 0.0, 1.0))

    def _structural(self, node: MemoryNode) -> float:
        """Ring proximity: inner rings are structurally central (§9)."""
        if node.ring is None:
            return 0.0
        max_ring = max(1, self.config.get("topology", {}).get("max_rings", 5) - 1)
        return float(np.clip(1.0 - node.ring / max_ring, 0.0, 1.0))

    def _radial(self, node: MemoryNode) -> float:
        """Inverted radial distance: close to core scores high (§12)."""
        if node.radial_distance is None:
            return 0.5
        return float(np.clip(1.0 - node.radial_distance, 0.0, 1.0))

    def _graph(self, node: MemoryNode) -> float:
        return float(np.clip(self.centrality.get(node.id, 0.0) * 5.0, 0.0, 1.0))

    def _recency(self, node: MemoryNode) -> float:
        try:
            age_h = (utcnow() - datetime.fromisoformat(node.updated_at)).total_seconds() / 3600
            return float(np.clip(1.0 - age_h / (24 * 30), 0.0, 1.0))
        except Exception:
            return 0.5

    def _path_score(self, node: MemoryNode, path: List[str]) -> float:
        """Higher when reached through a short, confident chain."""
        if not path:
            return 0.0
        conf = 1.0
        for a, b in zip(path, path[1:]):
            ed = self.gs.edge_data(a, b)
            conf *= float(ed.get("confidence", 0.5))
        length_penalty = 1.0 / len(path)
        return float(np.clip(conf * length_penalty * 2.0, 0.0, 1.0))

    # ------------------------------------------------------------------
    # retrieval pipeline (§20)
    # ------------------------------------------------------------------
    def retrieve(self, query: str, query_vec: np.ndarray,
                 active_components: Optional[Tuple[str, ...]] = None,
                 final_k: Optional[int] = None) -> RetrievalResult:
        t0 = time.perf_counter()
        if isinstance(active_components, str):
            # API callers may pass "all" or a comma-separated component list
            names = [c.strip() for c in active_components.split(",") if c.strip()]
            active_components = None if not names or names == ["all"] else tuple(names)
        unknown = set(active_components or ()) - set(ALL_COMPONENTS)
        if unknown:
            raise ValueError(f"unknown retrieval components: {sorted(unknown)}")
        active = set(active_components) if active_components else set(ALL_COMPONENTS)
        active_names = tuple(sorted(active))
        k = final_k or self.final_k

        # 1-2. semantic candidates from FAISS (§20 steps 3-4)
        cands: Dict[str, RetrievedItem] = {}
        if self.vs is not None:
            for score, meta in self.vs.search(query_vec, k=self.candidate_k):
                nid = meta.get("node_id")
                if nid in self._node_index:
                    cands[nid] = RetrievedItem(node=self._node_index[nid], path=[nid])

        # 3. graph expansion: one hop from top semantic hits, keep multi-hop paths (§22)
        # (semantic scores are computed here, before scoring, because seed
        # selection drives both graph expansion and activation seeding)
        seeds = [nid for nid, _ in sorted(
            ((nid, self._semantic(query_vec, self._node_index[nid])) for nid in cands),
            key=lambda kv: (-round(float(kv[1]), 5), kv[0]))][:5]
        for seed in seeds:
            for nid in sorted(self.gs.neighborhood(seed, radius=1)):
                if nid not in cands and nid in self._node_index:
                    path = self.gs.weighted_path(seed, nid)[: self.max_hops + 1]
                    cands[nid] = RetrievedItem(node=self._node_index[nid], path=path)
        # 3b. Bio-NN-inspired activation: inject nodes reached through
        # bounded multi-hop propagation from the semantic seeds —
        # surfaces associates that neither vector nor 1-hop expansion finds.
        act_map: Dict[str, float] = {}
        activation_trace: List[Dict[str, Any]] = []
        if "activation" in active:
            seed_scores = {nid: self._semantic(query_vec, self._node_index[nid])
                           for nid in seeds}
            act_map = dict(self.activation.activate(
                seed_scores, trace=activation_trace)[:16])
            for nid in sorted(act_map):
                if nid not in cands and nid in self._node_index:
                    cands[nid] = RetrievedItem(node=self._node_index[nid], path=[nid])
        # 4. hierarchical candidates: ring 0-1 concepts matching query tokens
        qtokens = {t for t in re.findall(r"[a-z0-9]{3,}", query.lower())}
        for node_id in sorted(self.frame.nodes):
            n = self.frame.nodes[node_id]
            if n.ring in (0, 1) and n.id not in cands:
                toks = set(re.findall(
                    r"[a-z0-9]{3,}",
                    (n.concept + " " + (n.summary or "")).lower(),
                ))
                if qtokens & toks:
                    cands[n.id] = RetrievedItem(node=n, path=[n.id])

        result = RetrievalResult(query=query, activation_trace=activation_trace)
        result.n_candidates = len(cands)

        # 5. score with active components only (§21)
        wsum = sum(self.weights[c] for c in active_names) or 1.0
        for node_id in sorted(cands):
            item = cands[node_id]
            n = item.node
            comp_scores = {
                "semantic": self._semantic(query_vec, n),
                "structural": self._structural(n),
                "radial": self._radial(n),
                "graph": self._graph(n),
                "importance": float(np.clip(n.importance, 0, 1)),
                "confidence": float(np.clip(n.confidence, 0, 1)),
                "recency": self._recency(n),
                "path": self._path_score(n, item.path) if "path" in active_names else 0.0,
                "activation": float(np.clip(act_map.get(n.id, 0.0), 0.0, 1.0)),
            }
            item.components = comp_scores
            item.score = sum(
                self.weights[c] * comp_scores[c] for c in active_names
            ) / wsum

        # 6. rerank (§21/§20 step 11)
        ranked = sorted(cands.values(), key=lambda it: (-it.score, it.node.id))[:k]
        result.items = ranked
        result.paths = [it.path for it in ranked if len(it.path) > 1]
        result.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return result

    def log_to(self, sqlite_store, result: RetrievalResult, context_tokens: int,
               system: str = "mira") -> None:
        sqlite_store.log_retrieval(
            stable_hash(result.query), system, result.n_candidates,
            len(result.items), result.latency_ms, context_tokens,
            {"active_components": ",".join(sorted(result.items[0].components))
             if result.items else ""},
        )


def ablation_configs() -> Dict[str, Optional[Tuple[str, ...]]]:
    """Named ablation sets (§29): None = full MIRA."""
    return {
        "full_mira": None,
        "vector_only": ("semantic",),
        "graph_only": ("graph", "path"),
        "hierarchy_only": ("structural",),
        "radial_only": ("radial",),
        "vector_graph": ("semantic", "graph", "path"),
        "vector_hierarchy": ("semantic", "structural"),
        "vector_radial": ("semantic", "radial"),
        "graph_hierarchy": ("graph", "structural", "path"),
        "graph_radial": ("graph", "radial", "path"),
        "activation_only": ("activation",),
        "vector_activation": ("semantic", "activation"),
    }
