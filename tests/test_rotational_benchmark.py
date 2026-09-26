"""Assert-based tests for the symbolic retrieval/topology benchmark."""
from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.rotation import (
    FIXED_TIMESTAMP,
    RotationBenchmarkConfig,
    build_synthetic_world,
    degree_preserving_shuffle,
    global_rotation_matrix,
    graph_degree_stats,
    rotate_coordinates,
    rotate_frame,
    rotate_vectors,
    run_benchmark,
    structured_full_edges,
    write_benchmark_output,
)


def _degree_map(edges):
    return Counter(
        endpoint
        for edge in edges
        for endpoint in (edge.source_id, edge.target_id)
    )


def _stable_report(report):
    manifest = json.loads(json.dumps(report["manifest"]))
    rows = []
    for row in report["results"]["rows"]:
        rows.append({key: value for key, value in row.items() if "latency" not in key})
    return manifest, rows


def test_degree_preserving_rewiring_is_deterministic():
    original = structured_full_edges(24)
    first = degree_preserving_shuffle(original, 19)
    second = degree_preserving_shuffle(original, 19)
    assert first == second
    assert len(first) == len(original)
    assert first != original
    assert _degree_map(first) == _degree_map(original)
    assert graph_degree_stats(first) == graph_degree_stats(original)


def test_global_rotation_preserves_geometry_and_inner_products():
    config = RotationBenchmarkConfig(seed=5, replicates=1, rotations=1,
                                     k=3, n_nodes=8, n_queries=2)
    world = build_synthetic_world(config)
    rotation = global_rotation_matrix(world.query_vectors.shape[1], 11)
    rotated = rotate_frame(world.frame, 90.0, rotation)
    original_points = np.asarray([
        node.metadata["coordinate"] for node in world.frame.nodes.values()
    ])
    rotated_points = np.asarray([
        node.metadata["coordinate"] for node in rotated.nodes.values()
    ])
    assert not np.allclose(original_points, rotated_points)
    assert np.allclose(
        np.linalg.norm(original_points, axis=1),
        np.linalg.norm(rotated_points, axis=1),
    )
    assert np.allclose(
        rotate_coordinates(original_points, 90.0), rotated_points
    )
    original_vectors = np.stack([
        node.embedding for node in world.frame.nodes.values()
    ])
    rotated_vectors = np.stack([
        node.embedding for node in rotated.nodes.values()
    ])
    assert np.allclose(
        original_vectors @ original_vectors.T,
        rotated_vectors @ rotated_vectors.T,
        atol=2e-5,
    )
    rotated_queries = rotate_vectors(world.query_vectors, rotation)
    assert np.allclose(
        world.query_vectors @ world.query_vectors.T,
        rotated_queries @ rotated_queries.T,
        atol=2e-5,
    )


def test_fixed_runtime_and_manifest():
    config = RotationBenchmarkConfig(seed=3, replicates=1, rotations=2,
                                     k=3, n_nodes=8, n_queries=3)
    world = build_synthetic_world(config)
    assert all(node.created_at == FIXED_TIMESTAMP for node in world.frame.nodes.values())
    assert all(node.updated_at == FIXED_TIMESTAMP for node in world.frame.nodes.values())
    assert all(edge.created_at == FIXED_TIMESTAMP for edge in world.frame.edges)
    report = run_benchmark(config)
    manifest = report["manifest"]
    assert manifest["study"] == "symbolic retrieval/topology invariance"
    assert manifest["not_biological_neural_simulation"] is True
    assert manifest["resolved_config"]["learned_scorer"] is False
    assert manifest["runtime"] == {
        "embedding_backend": "stub",
        "learned_scorer": False,
        "database": "none; in-memory MemoryFrame, VectorStore, and GraphStore",
        "llm": "disabled",
        "network": "disabled",
        "fixed_timestamps": True,
    }
    assert set(manifest["edge_counts"]) == {
        "structured_full", "degree_preserving_shuffled", "square_lattice"
    }
    assert manifest["rewiring"][0]["degree_sequence_preserved"] is True
    assert manifest["rewiring"][0]["edge_count_preserved"] is True
    assert "paired_top_k_jaccard" in manifest["metric_definitions"]
    assert "mrr" in manifest["metric_definitions"]


def test_benchmark_is_reproducible_and_metrics_are_bounded():
    config = RotationBenchmarkConfig(seed=17, replicates=1, rotations=2,
                                     k=4, n_nodes=10, n_queries=4)
    first = run_benchmark(config)
    second = run_benchmark(config)
    first_manifest, first_rows = _stable_report(first)
    second_manifest, second_rows = _stable_report(second)
    assert first_manifest == second_manifest
    assert first_rows == second_rows
    for row in first["results"]["rows"]:
        assert 0.0 <= row["paired_top_k_jaccard"] <= 1.0
        assert row["top1_agreement"] in (0.0, 1.0)
        assert 0.0 <= row["recall_at_k"] <= 1.0
        assert 0.0 <= row["mrr"] <= 1.0
        assert row["reference_latency_ms"] >= 0.0
        assert row["rotated_latency_ms"] >= 0.0
    aggregate = first["results"]["aggregate"]
    assert aggregate["paired_top_k_jaccard"] >= 0.999
    assert aggregate["top1_agreement"] >= 0.999
    for name in ("paired_top_k_jaccard", "top1_agreement", "recall", "mrr"):
        assert 0.0 <= aggregate[name] <= 1.0


def test_json_output_has_results_and_manifest():
    config = RotationBenchmarkConfig(seed=2, replicates=1, rotations=1,
                                     k=2, n_nodes=6, n_queries=2)
    report = run_benchmark(config)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "benchmark.json"
        assert write_benchmark_output(report, path) == path
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["manifest"]["seed"] == 2
        assert loaded["results"]["rows"]


def main():
    tests = [
        test_degree_preserving_rewiring_is_deterministic,
        test_global_rotation_preserves_geometry_and_inner_products,
        test_fixed_runtime_and_manifest,
        test_benchmark_is_reproducible_and_metrics_are_bounded,
        test_json_output_has_results_and_manifest,
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    if failed:
        raise SystemExit(1)
    print("ROTATIONAL BENCHMARK TESTS PASS")


if __name__ == "__main__":
    main()
