"""Run the deterministic symbolic retrieval/topology invariance benchmark."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def _parser() -> argparse.ArgumentParser:
    from evaluation.rotation import CANONICAL_CONDITIONS

    parser = argparse.ArgumentParser(
        description="Run the symbolic retrieval/topology invariance benchmark"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--rotations", type=int, default=4)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--nodes", type=int, default=24)
    parser.add_argument("--queries", type=int, default=12)
    parser.add_argument(
        "--output",
        default="experiments/rotational_benchmark",
        help="JSON file, output directory, or - for stdout",
    )
    parser.add_argument(
        "--condition",
        action="append",
        choices=CANONICAL_CONDITIONS,
        help="run one or more conditions; repeat the option for multiple arms",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from evaluation.rotation import (
        CANONICAL_CONDITIONS,
        RotationBenchmarkConfig,
        run_benchmark,
        write_benchmark_output,
    )

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = RotationBenchmarkConfig(
            seed=args.seed,
            replicates=args.replicates,
            rotations=args.rotations,
            k=args.k,
            n_nodes=args.nodes,
            n_queries=args.queries,
            conditions=tuple(args.condition or CANONICAL_CONDITIONS),
        )
        report = run_benchmark(config)
        destination = write_benchmark_output(report, args.output)
    except (TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    aggregate = report["results"]["aggregate"]
    print(
        "[mira] symbolic retrieval/topology invariance "
        f"jaccard={aggregate['paired_top_k_jaccard']} "
        f"top1={aggregate['top1_agreement']} "
        f"recall={aggregate['recall']} mrr={aggregate['mrr']}"
    )
    if destination is not None:
        print(f"[mira] results: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
