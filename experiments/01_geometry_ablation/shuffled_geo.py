from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def run():
    from scripts.run_rotational_benchmark import main

    return main([
        "--condition", "degree_preserving_shuffled",
        "--output", "experiments/01_geometry_ablation/shuffled_geo",
    ])


if __name__ == "__main__":
    raise SystemExit(run())
