"""Deterministic in-memory symbolic retrieval/topology invariance benchmark.

The benchmark rotates a shared coordinate frame and applies the same
orthogonal transformation to every embedding and query vector. It measures
symbolic retrieval and graph-topology invariance with fixed synthetic data; it
is not a biological neural simulation.
"""
from __future__ import annotations

import copy
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass, replace
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from core.memory import MemoryEdge, MemoryFrame, MemoryNode, MemoryType
from core.retrieval import MIRARetriever
from evaluation.metrics import aggregate, mrr, recall_at_k
from models.embeddings import EmbeddingBackend
from storage.graph_store import GraphStore
from storage.vector_store import VectorStore


STUDY_NAME = "symbolic retrieval/topology invariance"
FIXED_TIMESTAMP = "2020-01-01T00:00:00+00:00"
CONDITION_ALIASES = {
    "regular_polar_lattice": "square_lattice",
}
CANONICAL_CONDITIONS = (
    "structured_full",
    "degree_preserving_shuffled",
    "square_lattice",
)
CONDITIONS = CANONICAL_CONDITIONS


def _stub_embeddings() -> EmbeddingBackend:
    backend = EmbeddingBackend("stub")
    backend._load_attempted = True
    return backend


METRIC_DEFINITIONS = {
    "paired_top_k_jaccard": "For each paired query, |reference ∩ rotated| / |reference ∪ rotated|; 1.0 when both result sets are empty; averaged over queries and rotations.",
    "top1_agreement": "Fraction of paired queries whose first retrieved node id is the same in the reference and rotated runs.",
    "recall_at_k": "Mean of evaluation.metrics.recall_at_k for the reference and rotated runs against the deterministic query target node.",
    "mrr": "Mean of evaluation.metrics.mrr for the reference and rotated runs against the deterministic query target node.",
    "latency_ms": "Mean MIRARetriever RetrievalResult latency for reference and rotated calls; this is a wall-clock diagnostic and is hardware-dependent.",
}
LIMITATIONS = [
    "This is a symbolic retrieval/topology invariance study, not a biological neural simulation.",
    "Embeddings use the deterministic local stub hashing backend, not a trained semantic model.",
    "The target node is the synthetic relevance label; this is not an open-domain or biological benchmark.",
    "Degree-preserving rewiring uses deterministic double-edge swaps and preserves the degree sequence when a valid swap sequence exists; it is not a sample from the full configuration ensemble.",
    "A common orthogonal embedding rotation preserves pairwise dot products by construction, so this experiment tests the paired MIRA retrieval pipeline rather than general geometric robustness.",
    "Latency depends on the host machine and is not a cross-machine performance claim.",
    "The square_lattice condition is a regular geometric topology and is not a claim about polar or biological organization.",
    "The square_lattice comparator is descriptive and is not degree- or edge-matched; condition-level differences cannot isolate radial organization.",
]


@dataclass(frozen=True)
class RotationBenchmarkConfig:
    """Validated, resolved inputs for the rotational benchmark."""

    seed: int = 42
    replicates: int = 3
    rotations: int = 4
    k: int = 5
    n_nodes: int = 24
    n_queries: int = 12
    conditions: Tuple[str, ...] = CANONICAL_CONDITIONS

    def __post_init__(self) -> None:
        for name in ("seed", "replicates", "rotations", "k", "n_nodes", "n_queries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        if isinstance(self.conditions, str):
            raise TypeError("conditions must be a sequence of condition names")
        if self.replicates < 1:
            raise ValueError("replicates must be at least 1")
        if self.rotations < 1:
            raise ValueError("rotations must be at least 1")
        if self.k < 1 or self.k > self.n_nodes:
            raise ValueError("k must be between 1 and n_nodes")
        if self.n_nodes < 4:
            raise ValueError("n_nodes must be at least 4")
        if self.n_queries < 1:
            raise ValueError("n_queries must be at least 1")
        normalized: List[str] = []
        for condition in self.conditions:
            canonical = CONDITION_ALIASES.get(condition, condition)
            if canonical not in CANONICAL_CONDITIONS:
                raise ValueError(f"unknown condition: {condition!r}")
            if canonical not in normalized:
                normalized.append(canonical)
        if not normalized:
            raise ValueError("at least one condition is required")
        object.__setattr__(self, "conditions", tuple(normalized))


@dataclass
class SyntheticWorld:
    """In-memory frame, stores, and paired query data for one condition."""

    frame: MemoryFrame
    vector_store: VectorStore
    graph_store: GraphStore
    queries: List[Dict[str, Any]]
    query_vectors: np.ndarray
    condition: str
    replicate: int


def _edge_endpoints(edge: Any) -> Tuple[str, str]:
    if isinstance(edge, MemoryEdge):
        return str(edge.source_id), str(edge.target_id)
    if isinstance(edge, Mapping):
        return str(edge["source_id"]), str(edge["target_id"])
    if isinstance(edge, (tuple, list)) and len(edge) >= 2:
        return str(edge[0]), str(edge[1])
    raise TypeError("edges must be MemoryEdge, mapping, or (source, target)")


def _canonical_pair(source: str, target: str) -> Tuple[str, str]:
    return (source, target) if source <= target else (target, source)


def _copy_edge(edge: Any, source: str, target: str,
               fixed_timestamp: str = FIXED_TIMESTAMP) -> Any:
    source, target = _canonical_pair(str(source), str(target))
    if isinstance(edge, MemoryEdge):
        return MemoryEdge(
            source_id=source,
            target_id=target,
            relation_type=edge.relation_type,
            weight=edge.weight,
            confidence=edge.confidence,
            provenance=list(edge.provenance or []),
            created_at=edge.created_at or fixed_timestamp,
        )
    if isinstance(edge, Mapping):
        copied = dict(edge)
        copied["source_id"] = source
        copied["target_id"] = target
        copied.setdefault("created_at", fixed_timestamp)
        return copied
    return (source, target)


def _edge_sort_key(edge: Any) -> Tuple[str, str]:
    return _edge_endpoints(edge)


def _unique_edges(edges: Sequence[Any],
                  fixed_timestamp: str = FIXED_TIMESTAMP) -> List[Any]:
    seen = set()
    result = []
    for edge in edges:
        source, target = _edge_endpoints(edge)
        if source == target:
            continue
        pair = _canonical_pair(source, target)
        if pair in seen:
            continue
        seen.add(pair)
        result.append(_copy_edge(edge, pair[0], pair[1], fixed_timestamp))
    return sorted(result, key=_edge_sort_key)


def _new_edge(source: int, target: int, n_nodes: int,
              fixed_timestamp: str = FIXED_TIMESTAMP) -> MemoryEdge:
    if source == target:
        raise ValueError("self edges are not valid")
    a = f"n{source:03d}"
    b = f"n{target:03d}"
    return MemoryEdge(
        source_id=a,
        target_id=b,
        relation_type="related",
        weight=1.0,
        confidence=0.5,
        provenance=[f"geometry:{a}", f"geometry:{b}"],
        created_at=fixed_timestamp,
    )


def structured_full_edges(n_nodes: int,
                          fixed_timestamp: str = FIXED_TIMESTAMP) -> List[MemoryEdge]:
    """Build a deterministic structured circulant-plus-hierarchy graph."""
    if n_nodes < 4:
        raise ValueError("n_nodes must be at least 4")
    pairs = set()
    half = max(1, n_nodes // 2)
    for i in range(n_nodes):
        for target in ((i + 1) % n_nodes, (i + 2) % n_nodes, (i + half) % n_nodes):
            if i != target:
                pairs.add(_canonical_pair(f"n{i:03d}", f"n{target:03d}"))
    for i in range(1, n_nodes, 4):
        pairs.add(_canonical_pair("n000", f"n{i:03d}"))
    return sorted(
        (_new_edge(int(a[1:]), int(b[1:]), n_nodes, fixed_timestamp)
         for a, b in pairs),
        key=_edge_sort_key,
    )


def square_lattice_edges(n_nodes: int,
                         fixed_timestamp: str = FIXED_TIMESTAMP) -> List[MemoryEdge]:
    """Build a deterministic rectangular square-lattice graph."""
    if n_nodes < 4:
        raise ValueError("n_nodes must be at least 4")
    rows = max(1, int(math.sqrt(n_nodes)))
    columns = (n_nodes + rows - 1) // rows
    pairs = set()
    for row in range(rows):
        for column in range(columns):
            index = row * columns + column
            if index >= n_nodes:
                continue
            if column + 1 < columns and index + 1 < n_nodes:
                pairs.add(_canonical_pair(f"n{index:03d}", f"n{index + 1:03d}"))
            if row + 1 < rows and index + columns < n_nodes:
                pairs.add(_canonical_pair(f"n{index:03d}", f"n{index + columns:03d}"))
    return sorted(
        (_new_edge(int(a[1:]), int(b[1:]), n_nodes, fixed_timestamp)
         for a, b in pairs),
        key=_edge_sort_key,
    )


regular_polar_lattice_edges = square_lattice_edges


def degree_preserving_shuffle(edges: Sequence[Any], seed: int) -> List[Any]:
    """Deterministically rewire a simple graph with double-edge swaps.

    Each accepted swap replaces two disjoint edges with two cross-edges, so
    the edge count and every node degree are preserved. The input is returned
    unchanged if the graph admits no valid swap.
    """
    current = _unique_edges(edges)
    if len(current) < 2:
        return current
    rng = random.Random(int(seed))
    current_set = {_edge_endpoints(edge) for edge in current}
    target_swaps = max(1, min(len(current) // 3, 25))
    swaps = 0
    attempts = 0
    max_attempts = max(200, len(current) * 60)
    while swaps < target_swaps and attempts < max_attempts:
        attempts += 1
        first_index = rng.randrange(len(current))
        second_index = rng.randrange(len(current) - 1)
        if second_index >= first_index:
            second_index += 1
        first = current[first_index]
        second = current[second_index]
        a, b = _edge_endpoints(first)
        c, d = _edge_endpoints(second)
        if len({a, b, c, d}) < 4:
            continue
        candidates = [((a, c), (b, d)), ((a, d), (b, c))]
        if rng.randrange(2):
            candidates.reverse()
        for first_pair, second_pair in candidates:
            first_pair = _canonical_pair(*first_pair)
            second_pair = _canonical_pair(*second_pair)
            if first_pair in current_set or second_pair in current_set:
                continue
            replacement = list(current)
            replacement[first_index] = _copy_edge(first, *first_pair)
            replacement[second_index] = _copy_edge(second, *second_pair)
            new_first = _edge_endpoints(replacement[first_index])
            new_second = _edge_endpoints(replacement[second_index])
            replacement = sorted(replacement, key=_edge_sort_key)
            current = replacement
            current_set.remove(_edge_endpoints(first))
            current_set.remove(_edge_endpoints(second))
            current_set.add(new_first)
            current_set.add(new_second)
            swaps += 1
            break
    return current


degree_preserving_shuffled = degree_preserving_shuffle


def _lattice_coordinates(n_nodes: int) -> List[List[float]]:
    rows = max(1, int(math.sqrt(n_nodes)))
    columns = (n_nodes + rows - 1) // rows
    center_x = (columns - 1) / 2.0
    center_y = (rows - 1) / 2.0
    coordinates = []
    for index in range(n_nodes):
        row, column = divmod(index, columns)
        coordinates.append([float(column - center_x), float(row - center_y)])
    return coordinates


def _structured_coordinates(n_nodes: int) -> List[List[float]]:
    layers = min(4, max(1, n_nodes))
    coordinates = []
    for index in range(n_nodes):
        layer = index % layers
        position = index // layers
        count = max(1, int(math.ceil((n_nodes - layer) / layers)))
        angle = 2.0 * math.pi * (position + 0.17 * layer) / count
        radius = 0.25 + 0.75 * layer / max(1, layers - 1) if layers > 1 else 0.25
        coordinates.append([radius * math.cos(angle), radius * math.sin(angle)])
    return coordinates


def _apply_geometry(frame: MemoryFrame, max_rings: int = 5) -> None:
    values = []
    for node_id in sorted(frame.nodes):
        coordinate = frame.nodes[node_id].metadata.get("coordinate", [0.0, 0.0])
        values.append(np.asarray(coordinate, dtype="float64"))
    matrix = np.asarray(values, dtype="float64")
    if matrix.ndim != 2 or matrix.shape[1] != 2:
        raise ValueError("benchmark coordinates must have shape (n, 2)")
    maximum = float(np.max(np.linalg.norm(matrix, axis=1))) if len(matrix) else 0.0
    for node_id, coordinate in zip(sorted(frame.nodes), matrix):
        radius = float(np.linalg.norm(coordinate) / maximum) if maximum else 0.0
        angle = math.atan2(float(coordinate[1]), float(coordinate[0]))
        sector = int(((angle + math.pi) / (2.0 * math.pi)) * 8.0) % 8
        node = frame.nodes[node_id]
        node.ring = min(max_rings - 1, int(radius * max_rings))
        node.depth = node.ring
        node.sector = f"sector_{sector:02d}"
        node.radial_distance = round(radius, 8)
        node.metadata["radius"] = round(radius, 8)
        node.metadata["angle_degrees"] = round(math.degrees(angle), 6)
    return None


def rotate_coordinates(points: Sequence[Sequence[float]], angle_degrees: float) -> np.ndarray:
    """Rotate 2-D benchmark coordinates by a global angle in degrees."""
    matrix = np.asarray(points, dtype="float64")
    if matrix.ndim != 2 or matrix.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    radians = math.radians(float(angle_degrees))
    cosine, sine = math.cos(radians), math.sin(radians)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype="float64")
    return np.asarray(matrix @ rotation.T, dtype="float64")


def global_rotation_matrix(dimension: int, seed: int) -> np.ndarray:
    """Return a deterministic common orthogonal matrix for embedding coordinates."""
    if dimension < 1:
        raise ValueError("dimension must be at least 1")
    rng = np.random.default_rng(int(seed) & ((1 << 32) - 1))
    matrix, triangular = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    signs = np.sign(np.diag(triangular))
    signs[signs == 0] = 1.0
    return np.ascontiguousarray(matrix * signs, dtype="float64")


def rotate_vectors(vectors: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """Apply a common orthogonal transformation to a row vector matrix."""
    matrix = np.asarray(vectors)
    transform = np.asarray(rotation_matrix)
    if matrix.ndim != 2 or transform.shape != (matrix.shape[1], matrix.shape[1]):
        raise ValueError("rotation_matrix shape must match the vector dimension")
    return np.ascontiguousarray(matrix @ transform.T, dtype="float32")


def rotate_frame(frame: MemoryFrame, angle_degrees: float,
                 vector_rotation: Optional[np.ndarray] = None) -> MemoryFrame:
    """Return a copied frame with global 2-D and optional embedding rotation."""
    rotated = copy.deepcopy(frame)
    for node in rotated.nodes.values():
        coordinate = node.metadata.get("coordinate")
        if coordinate is not None:
            node.metadata["coordinate"] = rotate_coordinates(
                [coordinate], angle_degrees)[0].tolist()
        node.metadata["rotation_degrees"] = round(float(angle_degrees), 6)
    if vector_rotation is not None:
        embedded_ids = [
            node_id for node_id in sorted(rotated.nodes)
            if rotated.nodes[node_id].embedding is not None
        ]
        if embedded_ids:
            matrix = np.stack([
                np.asarray(rotated.nodes[node_id].embedding)
                for node_id in embedded_ids
            ])
            transformed = rotate_vectors(matrix, vector_rotation)
            for node_id, vector in zip(embedded_ids, transformed):
                rotated.nodes[node_id].embedding = vector.copy()
    _apply_geometry(rotated)
    return rotated


def _replicate_seed(seed: int, replicate: int) -> int:
    return (int(seed) + 1000003 * int(replicate)) & ((1 << 63) - 1)


def _node_text(index: int, n_nodes: int, replicate: int) -> str:
    topic = index % min(6, max(1, n_nodes))
    return (
        f"synthetic retrieval topic {topic:02d} node {index:03d} "
        f"replicate {replicate:03d} fixed geometry marker"
    )


def _build_payload(config: RotationBenchmarkConfig, replicate: int,
                   embeddings: Any) -> Dict[str, Any]:
    rng = random.Random(_replicate_seed(config.seed, replicate))
    texts = [_node_text(i, config.n_nodes, replicate) for i in range(config.n_nodes)]
    vectors = np.asarray(embeddings.encode(texts), dtype="float32")
    if vectors.ndim != 2 or vectors.shape[0] != config.n_nodes:
        raise ValueError("embedding backend returned an unexpected shape")
    order = list(range(config.n_nodes))
    rng.shuffle(order)
    queries = []
    query_texts = []
    for index in range(config.n_queries):
        target = order[index % len(order)]
        text = _node_text(target, config.n_nodes, replicate)
        queries.append({
            "id": f"q{index:03d}",
            "text": text,
            "target_id": f"n{target:03d}",
            "relevant_ids": [f"n{target:03d}"],
        })
        query_texts.append(text)
    query_vectors = np.asarray(embeddings.encode(query_texts), dtype="float32")
    if query_vectors.shape[0] != config.n_queries:
        raise ValueError("embedding backend returned an unexpected query shape")
    return {
        "vectors": vectors,
        "query_vectors": query_vectors,
        "queries": queries,
        "replicate": replicate,
        "structured_coordinates": _structured_coordinates(config.n_nodes),
        "lattice_coordinates": _lattice_coordinates(config.n_nodes),
    }


def _edges_for_condition(condition: str, n_nodes: int, seed: int) -> Tuple[List[Any], Dict[str, Any]]:
    if condition == "structured_full":
        edges = structured_full_edges(n_nodes)
        return edges, {
            "method": "deterministic structured circulant plus hierarchy",
            "changed": False,
        }
    if condition == "degree_preserving_shuffled":
        original = structured_full_edges(n_nodes)
        edges = degree_preserving_shuffle(original, seed)
        original_pairs = {_edge_endpoints(edge) for edge in original}
        shuffled_pairs = {_edge_endpoints(edge) for edge in edges}
        original_degrees = Counter(endpoint for edge in original for endpoint in _edge_endpoints(edge))
        shuffled_degrees = Counter(endpoint for edge in edges for endpoint in _edge_endpoints(edge))
        return edges, {
            "method": "deterministic double-edge swaps",
            "seed": int(seed),
            "changed": original_pairs != shuffled_pairs,
            "edge_count_preserved": len(original) == len(edges),
            "degree_sequence_preserved": original_degrees == shuffled_degrees,
            "original_edge_count": len(original),
            "rewired_edge_count": len(edges),
        }
    if condition == "square_lattice":
        return square_lattice_edges(n_nodes), {
            "method": "deterministic rectangular square lattice",
            "changed": False,
        }
    raise ValueError(f"unknown condition: {condition}")


def _build_frame(config: RotationBenchmarkConfig, payload: Dict[str, Any],
                 condition: str, edges: Sequence[Any],
                 vector_rotation: Optional[np.ndarray] = None,
                 angle_degrees: float = 0.0) -> MemoryFrame:
    coordinates = (payload["lattice_coordinates"] if condition == "square_lattice"
                   else payload["structured_coordinates"])
    vectors = rotate_vectors(payload["vectors"], vector_rotation) \
        if vector_rotation is not None else payload["vectors"]
    coordinates = rotate_coordinates(coordinates, angle_degrees) \
        if angle_degrees else np.asarray(coordinates)
    frame = MemoryFrame()
    for index, node_id in enumerate(sorted({f"n{i:03d}" for i in range(config.n_nodes)})):
        text = _node_text(index, config.n_nodes, int(payload["replicate"]))
        node = MemoryNode(
            id=node_id,
            concept=f"synthetic_topic_{index % min(6, max(1, config.n_nodes)):02d}_{index:03d}",
            memory_type=MemoryType.CONCEPT,
            summary=text,
            raw_text=text,
            embedding=np.asarray(vectors[index], dtype="float32").copy(),
            importance=round(0.45 + ((index * 17) % 11) / 100.0, 6),
            confidence=round(0.55 + ((index * 13) % 9) / 100.0, 6),
            created_at=FIXED_TIMESTAMP,
            updated_at=FIXED_TIMESTAMP,
            valid_from=FIXED_TIMESTAMP,
            source_ids=[f"synthetic:{index:03d}"],
            metadata={
                "benchmark_condition": condition,
                "benchmark_node_index": index,
                "coordinate_space": "2d",
                "coordinate": [round(float(value), 8) for value in coordinates[index]],
            },
        )
        frame.add_node(node)
    for edge in _unique_edges(edges):
        frame.add_edge(edge)
    _apply_geometry(frame)
    return frame


def _stores_for_frame(frame: MemoryFrame) -> Tuple[VectorStore, GraphStore]:
    nodes = [frame.nodes[node_id] for node_id in sorted(frame.nodes)]
    matrix = np.stack([np.asarray(node.embedding, dtype="float32") for node in nodes])
    vector_store = VectorStore(dim=int(matrix.shape[1]))
    vector_store.add(matrix, [{"node_id": node.id} for node in nodes])
    graph_store = GraphStore()
    graph_store.build_from(
        [node.to_row() for node in nodes],
        [edge.to_row() for edge in frame.edges],
    )
    return vector_store, graph_store


def build_synthetic_world(config: Optional[RotationBenchmarkConfig] = None,
                          condition: str = "structured_full",
                          replicate: int = 0,
                          angle_degrees: float = 0.0,
                          vector_rotation: Optional[np.ndarray] = None,
                          embeddings: Any = None) -> SyntheticWorld:
    """Build one deterministic in-memory synthetic world."""
    config = config or RotationBenchmarkConfig()
    canonical = CONDITION_ALIASES.get(condition, condition)
    if canonical not in CANONICAL_CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    embeddings = embeddings if embeddings is not None else _stub_embeddings()
    payload = _build_payload(config, replicate, embeddings)
    seed = _replicate_seed(config.seed, replicate)
    edges, _ = _edges_for_condition(canonical, config.n_nodes, seed)
    frame = _build_frame(config, payload, canonical, edges,
                         vector_rotation=vector_rotation,
                         angle_degrees=angle_degrees)
    vector_store, graph_store = _stores_for_frame(frame)
    query_vectors = (rotate_vectors(payload["query_vectors"], vector_rotation)
                     if vector_rotation is not None else payload["query_vectors"])
    return SyntheticWorld(
        frame=frame,
        vector_store=vector_store,
        graph_store=graph_store,
        queries=payload["queries"],
        query_vectors=query_vectors,
        condition=canonical,
        replicate=replicate,
    )


def graph_degree_stats(graph: Any, node_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Return JSON-safe degree statistics for a GraphStore or edge sequence."""
    if hasattr(graph, "g"):
        ids = list(node_ids) if node_ids is not None else list(graph.g.nodes)
        degrees = [int(graph.degree(node_id)) for node_id in ids]
    else:
        edges = _unique_edges(list(graph))
        ids = list(node_ids) if node_ids is not None else sorted({
            endpoint for edge in edges for endpoint in _edge_endpoints(edge)
        })
        counts = Counter(endpoint for edge in edges for endpoint in _edge_endpoints(edge))
        degrees = [int(counts.get(node_id, 0)) for node_id in ids]
    return _degree_payload(sorted(degrees))


def _degree_payload(degrees: Sequence[int]) -> Dict[str, Any]:
    values = [int(value) for value in degrees]
    histogram = Counter(values)
    if values:
        array = np.asarray(values, dtype="float64")
        mean = float(array.mean())
        median = float(np.median(array))
        std = float(array.std())
    else:
        mean = median = std = 0.0
    return {
        "min": min(values) if values else 0,
        "max": max(values) if values else 0,
        "mean": round(mean, 6),
        "median": round(median, 6),
        "std": round(std, 6),
        "histogram": {str(key): int(histogram[key]) for key in sorted(histogram)},
        "sequence": values,
    }


def _frame_graph_stats(frame: MemoryFrame, graph_store: GraphStore) -> Dict[str, Any]:
    node_ids = sorted(frame.nodes)
    degrees = [int(graph_store.degree(node_id)) for node_id in node_ids]
    frame_stats = frame.stats()
    return {
        "nodes": int(frame_stats["nodes"]),
        "edges": int(graph_store.g.number_of_edges()),
        "frame_edges": int(frame_stats["edges"]),
        "degree": _degree_payload(degrees),
    }


def _retrieval_config(config: RotationBenchmarkConfig) -> Dict[str, Any]:
    return {
        "topology": {"max_rings": 5},
        "retrieval": {
            "candidate_k": max(config.k, config.n_nodes),
            "final_k": config.k,
            "max_hops": 3,
        },
        "retrieval_score": {
            "alpha_semantic": 0.35,
            "beta_structural": 0.15,
            "gamma_radial": 0.15,
            "delta_graph": 0.15,
            "epsilon_importance": 0.08,
            "zeta_confidence": 0.07,
            "eta_recency": 0.03,
            "theta_path": 0.02,
            "iota_activation": 0.12,
            "learned": False,
        },
        "activation": {
            "hops": 2,
            "decay": 0.5,
            "fanout_normalize": True,
            "seed_semantic_weight": 1.0,
            "seed_importance_weight": 0.3,
        },
    }


def _retrieve(frame: MemoryFrame, vector_store: VectorStore,
              graph_store: GraphStore, queries: Sequence[Dict[str, Any]],
              query_vectors: np.ndarray, config: Dict[str, Any],
              k: int) -> List[Tuple[List[str], float]]:
    retriever = MIRARetriever(frame, vector_store, graph_store, config)
    results = []
    for query, query_vector in zip(queries, query_vectors):
        result = retriever.retrieve(query["text"], query_vector, final_k=k)
        results.append(([item.node.id for item in result.items],
                        float(result.latency_ms)))
    return results


def _paired_jaccard(reference: Sequence[str], rotated: Sequence[str]) -> float:
    left, right = set(reference), set(rotated)
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def _top1_agreement(reference: Sequence[str], rotated: Sequence[str]) -> float:
    if not reference and not rotated:
        return 1.0
    if not reference or not rotated:
        return 0.0
    return float(reference[0] == rotated[0])


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    selected = [{
        "top_k_jaccard": row["top_k_jaccard"],
        "top1_agreement": row["top1_agreement"],
        "recall_at_k": row["recall_at_k"],
        "mrr": row["mrr"],
        "latency_ms": row["latency_ms"],
    } for row in rows]
    summary = aggregate(selected)
    summary["paired_top_k_jaccard"] = summary.get("top_k_jaccard", 0.0)
    summary["recall"] = summary.get("recall_at_k", 0.0)
    summary["n_queries"] = len(rows)
    return summary


def _resolved_config(config: RotationBenchmarkConfig,
                     angles: Sequence[float]) -> Dict[str, Any]:
    resolved = asdict(config)
    resolved["conditions"] = list(config.conditions)
    resolved["fixed_timestamp"] = FIXED_TIMESTAMP
    resolved["study"] = STUDY_NAME
    resolved["embedding_backend"] = "stub"
    resolved["learned_scorer"] = False
    resolved["offline"] = True
    resolved["live_database"] = False
    resolved["live_llm"] = False
    resolved["rotation_angles_degrees"] = list(angles)
    resolved["retrieval"] = _retrieval_config(config)
    resolved["condition_aliases"] = dict(CONDITION_ALIASES)
    return resolved


def _rewiring_report(original: Sequence[Any], rewired: Sequence[Any],
                     seed: int) -> Dict[str, Any]:
    original_pairs = {_edge_endpoints(edge) for edge in original}
    rewired_pairs = {_edge_endpoints(edge) for edge in rewired}
    original_degrees = Counter(endpoint for edge in original for endpoint in _edge_endpoints(edge))
    rewired_degrees = Counter(endpoint for edge in rewired for endpoint in _edge_endpoints(edge))
    return {
        "method": "deterministic double-edge swaps",
        "seed": int(seed),
        "changed": original_pairs != rewired_pairs,
        "edge_count_preserved": len(original) == len(rewired),
        "degree_sequence_preserved": original_degrees == rewired_degrees,
        "original_edge_count": len(original),
        "rewired_edge_count": len(rewired),
    }


def _coerce_config(config: Optional[RotationBenchmarkConfig | Mapping[str, Any]],
                   overrides: Mapping[str, Any]) -> RotationBenchmarkConfig:
    if config is None:
        return RotationBenchmarkConfig(**dict(overrides))
    if isinstance(config, RotationBenchmarkConfig):
        return replace(config, **dict(overrides)) if overrides else config
    if isinstance(config, Mapping):
        values = dict(config)
        values.update(overrides)
        return RotationBenchmarkConfig(**values)
    raise TypeError("config must be RotationBenchmarkConfig, mapping, or None")


def run_benchmark(config: Optional[RotationBenchmarkConfig | Mapping[str, Any]] = None,
                  **overrides: Any) -> Dict[str, Any]:
    """Run paired reference/rotated retrieval and return JSON-ready results."""
    config = _coerce_config(config, overrides)
    embeddings = _stub_embeddings()
    dimension = int(getattr(embeddings, "dim"))
    rotation_matrix = global_rotation_matrix(dimension, config.seed + 104729)
    angles = [round(360.0 * (index + 1) / (config.rotations + 1), 6)
              for index in range(config.rotations)]
    retrieval_config = _retrieval_config(config)
    graph_counts: List[Dict[str, Any]] = []
    rewiring_reports: List[Dict[str, Any]] = []
    condition_rows: Dict[str, List[Dict[str, Any]]] = {c: [] for c in config.conditions}
    pair_summaries: List[Dict[str, Any]] = []
    for replicate in range(config.replicates):
        payload = _build_payload(config, replicate, embeddings)
        rotated_query_vectors = rotate_vectors(payload["query_vectors"], rotation_matrix)
        seed = _replicate_seed(config.seed, replicate)
        for condition in config.conditions:
            original_edges = structured_full_edges(config.n_nodes)
            edges, _ = _edges_for_condition(condition, config.n_nodes, seed)
            if condition == "degree_preserving_shuffled":
                edge_info = _rewiring_report(original_edges, edges, seed)
                rewiring_reports.append({"condition": condition, "replicate": replicate, **edge_info})
            base_frame = _build_frame(config, payload, condition, edges)
            base_vs, base_gs = _stores_for_frame(base_frame)
            reference = _retrieve(
                base_frame, base_vs, base_gs, payload["queries"],
                payload["query_vectors"], retrieval_config, config.k)
            stats = _frame_graph_stats(base_frame, base_gs)
            graph_counts.append({
                "condition": condition,
                "replicate": replicate,
                **stats,
            })
            for rotation_index, angle in enumerate(angles):
                rotated_frame = rotate_frame(base_frame, angle, rotation_matrix)
                rotated_vs, rotated_gs = _stores_for_frame(rotated_frame)
                rotated = _retrieve(
                    rotated_frame, rotated_vs, rotated_gs, payload["queries"],
                    rotated_query_vectors, retrieval_config, config.k)
                rows = []
                for query, reference_result, rotated_result in zip(
                        payload["queries"], reference, rotated):
                    reference_ids, reference_latency = reference_result
                    rotated_ids, rotated_latency = rotated_result
                    relevant = set(query["relevant_ids"])
                    reference_recall = recall_at_k(reference_ids, relevant, config.k)
                    rotated_recall = recall_at_k(rotated_ids, relevant, config.k)
                    reference_mrr = mrr(reference_ids, relevant)
                    rotated_mrr = mrr(rotated_ids, relevant)
                    jaccard = _paired_jaccard(reference_ids, rotated_ids)
                    row = {
                        "condition": condition,
                        "replicate": replicate,
                        "rotation_index": rotation_index,
                        "rotation_degrees": angle,
                        "query_id": query["id"],
                        "reference_top_k": reference_ids,
                        "rotated_top_k": rotated_ids,
                        "top_k_jaccard": round(jaccard, 6),
                        "paired_top_k_jaccard": round(jaccard, 6),
                        "top1_agreement": _top1_agreement(reference_ids, rotated_ids),
                        "reference_recall_at_k": round(reference_recall, 6),
                        "rotated_recall_at_k": round(rotated_recall, 6),
                        "recall_at_k": round((reference_recall + rotated_recall) / 2.0, 6),
                        "recall": round((reference_recall + rotated_recall) / 2.0, 6),
                        "reference_mrr": round(reference_mrr, 6),
                        "rotated_mrr": round(rotated_mrr, 6),
                        "mrr": round((reference_mrr + rotated_mrr) / 2.0, 6),
                        "reference_latency_ms": round(reference_latency, 4),
                        "rotated_latency_ms": round(rotated_latency, 4),
                        "latency_ms": round((reference_latency + rotated_latency) / 2.0, 4),
                    }
                    rows.append(row)
                    condition_rows[condition].append(row)
                pair_summary = {
                    "condition": condition,
                    "replicate": replicate,
                    "rotation_index": rotation_index,
                    "rotation_degrees": angle,
                    "metrics": _metric_summary(rows),
                }
                pair_summaries.append(pair_summary)
    all_rows = [row for rows in condition_rows.values() for row in rows]
    by_condition = {
        condition: _metric_summary(rows)
        for condition, rows in condition_rows.items()
    }
    results = {
        "schema_version": 1,
        "study": STUDY_NAME,
        "aggregate": _metric_summary(all_rows),
        "by_condition": by_condition,
        "pair_summaries": pair_summaries,
        "rows": all_rows,
    }
    manifest = {
        "schema_version": 1,
        "study": STUDY_NAME,
        "not_biological_neural_simulation": True,
        "seed": config.seed,
        "resolved_config": _resolved_config(config, angles),
        "graph_counts": graph_counts,
        "node_count": config.n_nodes,
        "edge_counts": {
            condition: sorted({row["edges"] for row in graph_counts
                               if row["condition"] == condition})
            for condition in config.conditions
        },
        "degree_stats": [
            {"condition": row["condition"], "replicate": row["replicate"],
             **row["degree"]}
            for row in graph_counts
        ],
        "rewiring": rewiring_reports,
        "metric_definitions": dict(METRIC_DEFINITIONS),
        "runtime": {
            "embedding_backend": "stub",
            "learned_scorer": False,
            "database": "none; in-memory MemoryFrame, VectorStore, and GraphStore",
            "llm": "disabled",
            "network": "disabled",
            "fixed_timestamps": True,
        },
        "limitations": list(LIMITATIONS),
    }
    return {"manifest": manifest, "results": results}


def run_rotational_benchmark(config: Optional[RotationBenchmarkConfig | Mapping[str, Any]] = None,
                             **overrides: Any) -> Dict[str, Any]:
    """Compatibility-named entry point for the rotational benchmark."""
    return run_benchmark(config, **overrides)


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)


def write_benchmark_output(report: Mapping[str, Any], output: str | Path) -> Optional[Path]:
    """Write a combined JSON file or a results.json/manifest.json directory."""
    target = str(output)
    if target == "-":
        print(_json_text(report))
        return None
    path = Path(target)
    if path.suffix.lower() == ".json":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json_text(report), encoding="utf-8")
        return path
    path.mkdir(parents=True, exist_ok=True)
    (path / "results.json").write_text(_json_text(report["results"]), encoding="utf-8")
    (path / "manifest.json").write_text(_json_text(report["manifest"]), encoding="utf-8")
    return path
