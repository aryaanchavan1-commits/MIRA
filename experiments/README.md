# MIRA Experiments

This directory separates reproducible research artifacts from the live application.
`EXP-*` directories are saved runs. Their `results.json` and `results.csv` files are the
source of truth for their reported numbers; `summary.md` is generated from those files.
Do not hand-edit generated tables.

## Artifact inventory

| Artifact | Status | Purpose |
|---|---|---|
| `EXP-0001` | historical smoke | Small synthetic acceptance run; not publication evidence |
| `EXP-0002` | historical smoke | Small synthetic acceptance run; not publication evidence |
| `EXP-0003` | historical smoke | Small synthetic acceptance run; not publication evidence |
| `EXP-0004` | historical web run | Corpus-generated retrieval run; not publication evidence |
| `01_geometry_ablation/` | reproducible wrappers | Structured, degree-preserving shuffled, and square-lattice conditions |
| `rotational_benchmark/` | generated output | Invariance benchmark output; ignored by Git because it can be regenerated |
| `../data_bench/bench_real_results.json` | real-data research | MuSiQue 300q × 3 seeds retrieval benchmark (mira vs flat vs hierarchical, bootstrap significance); exported to `paper/results_real.md` via `paper/export_results.py --real` |
| `../data_bench/ablation_real_results.json` | real-data research | Leave-one-out ablation over all 9 retrieval components on the same questions |
| `../data_bench/scale_sweep_results.json` | real-data research | Aware-vs-blind placement strategy gap across a corpus size ladder (0.05×→1×) |
| `../experiments/neural_validation.json` | real-data research | Doc-grouped holdout verdict for the learned scorer; config never auto-flips |

The older `EXP-*` manifests predate the current schema/provenance fields. They remain
useful as transparent run records, but they must not be presented as held-out research
results. In particular, their token-F1 values may be unavailable and their `git_commit`
fields may be empty.

## Running experiments

Use the project virtual environment and a real dataset path. A new run writes a fresh
`EXP-XXXX` directory:

```bat
.venv\Scripts\python.exe scripts\run_experiment.py --dataset data\datasets\custom.json --format custom --limit 50 --k 8 --study-type research --name custom-validation
.venv\Scripts\python.exe paper\export_results.py --exp EXP-0005
```

Smoke runs are allowed for plumbing checks, but the exporter rejects them unless
`--allow-smoke` is supplied. A research run must record dataset hash, resolved
configuration, model/runtime metadata, hardware, software versions, seed, git state,
and declared provenance.

## Geometry and invariance checks

The geometry wrappers call the deterministic headless benchmark. They use fixed
embeddings, fixed timestamps, in-memory stores, no LLM, and no network:

```bat
.venv\Scripts\python.exe experiments\01_geometry_ablation\full_mira.py
.venv\Scripts\python.exe experiments\01_geometry_ablation\shuffled_geo.py
.venv\Scripts\python.exe experiments\01_geometry_ablation\lattice_only.py
.venv\Scripts\python.exe scripts\run_rotational_benchmark.py --replicates 3 --rotations 4 --output experiments\rotational_benchmark
```

The square lattice is a descriptive comparator and is not degree- or edge-matched.
Rotation tests measure implementation invariance; they do not establish a biological
rotational representation.

## Affect-state validation

The transparent simulated affect layer is intentionally separate from retrieval
benchmarks. It never changes the mandala, embeddings, citations, or ranking. Its
bounded state, neutral decay, route rules, citation-error handling, explicit feedback,
identity route, and immutability checks are in `tests/test_affect.py`. The runtime
state is in-memory and resets on process restart.

## Safety and reproducibility

Do not commit private documents, database files, model weights, API keys, or live web
content. Generated benchmark directories are reproducible and are excluded from Git.
Use `git status` and the saved manifest before publishing or exporting a result.
