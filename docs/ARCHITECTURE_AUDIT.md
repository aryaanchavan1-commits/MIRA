# MIRA Architecture Audit (pre-NCM)

Written before any NCM code, per the research-program spec (§1). Everything here
was verified by reading the repository and the committed run artifacts on the
audit date. **No code was modified to produce this document.**

## 1. What MIRA is

A local-first retrieval memory system for RAG on a single consumer laptop
(RTX 3050 4 GB, 16 GB RAM, zero cloud calls). Evidence is stored as memory
nodes organized radially (rings × sectors), hierarchically (parents/children),
and as a graph; retrieval fuses an explicit, fully ablatable weighted function
(`core/retrieval.py`); biologically *inspired* dynamics (decay, bounded replay,
gists) run as an optional layer (`core/biomira.py`). The store grows
monotonically — nothing is deleted — so all interference results measure
*ranking*, not data loss.

## 2. Modules (verified inventory)

| Module | Role |
|---|---|
| `core/types.py` | helpers: `utcnow`, `iso_now`, `new_id`, `stable_hash` (sha256, 16-hex), `parse_float`, `clamp` |
| `core/memory.py` | `MemoryNode` (concept, parent_id/children, ring/sector/depth/radial_distance, importance/confidence, valid_from/until, source_ids, metadata; embedding held in memory only), `MemoryEdge`, `MemoryCluster` (sector), `MemoryRing`, `MemoryFrame` (in-memory nodes+edges, `children_of`, `neighbors`) |
| `core/placement.py` | `place(frame, mode)` — assigns ring/sector/depth/radial_distance ("hybrid_mira" etc.) |
| `core/retrieval.py` | `MIRARetriever`; `ALL_COMPONENTS = (semantic, structural, radial, graph, importance, confidence, recency, path, activation, stability)`; per-component scores normalized [0,1]; pipeline: FAISS candidates → graph expansion → spreading activation → ring-0/1 token match → weighted score → top-k; `ablation_configs()` = 12 named sets |
| `core/ranking.py` | benchmark-side scoring re-export of components |
| `core/activation.py` | `SpreadingActivation` — bounded multi-hop propagation |
| `core/biomira.py` | optional bio layer; `bio_config` §1 gate: off unless `biomira.enabled`, weight `kappa_stability` = 0 otherwise |
| `core/memory_dynamics.py` | decay/replay/consolidation passes (sleep) |
| `core/answer.py` | compress-then-cite answer stage; explicit no-evidence state; two-stage widened-window fallback |
| `core/agent.py` | memory gate (`memory_gate_semantic` 0.45), escalation labeling |
| `core/conflicts.py`, `core/updater.py`, `core/affect.py`, `core/hebbian.py`, `core/neural.py` | conflict detection, updates, affect, Hebbian/STDP-inspired traces, optional learned weights |
| `models/llm.py` | `LLMBackend` — GGUF ladder with cross-model fallback, budgeted on the commit limit (fixed this session) |
| `models/embeddings.py` | `init_embeddings("stub"|"minilm"|multilingual)` — pluggable |
| `storage/` | SQLite persistence, `VectorStore` (FAISS-style), `GraphStore` (`build_from`, `centrality`, `neighborhood`, `weighted_path`, `edge_data`) |
| `evaluation/` | `benchmark.run_system`, `metrics` (mrr, recall@k, token_f1), `report` |
| `scripts/` | bench builders (MuSiQue, HotpotQA, Indic), `eval_*` runners, forgetting lab (`build_forgetting_lab.py` + `eval_forgetting.py`), significance, ablation |
| `web/` | FastAPI console (`server.py` exposes `/api/chat`, `/api/lab`, `/api/biomira/*`, …), landing page + i18n (en/hi/mr) |
| `paper/` | `export_results.py --real` → `results_real.md`; `build_preprint.py` (pandoc→Chrome→pypdf); `ncm_program.md/pdf` research program |
| `tests/` | 29 suites, assert-based, plain scripts; `tests/test_acceptance.py` is the final gate |

## 3. Research hypotheses and results already in the repo

Six hypotheses (H1–H6) are pre-registered in `paper/ncm_program.md` with
thresholds and rejection conditions. Demonstrated so far [DEMO]:
MuSiQue MRR 0.7016 vs flat 0.4347 (p≈0.0001); HotpotQA 0.9175 vs 0.6774;
component attribution (structural 0.0442, activation 0.0179; graph/recency
inert on this corpus); forgetting lab variants A–K (radial variant best joint
variant; bio dynamics halve forgetting at MRR-neutral p=0.378; homeostasis
hurts); IndicQA honest boundary (BM25 strongest at full scale).

## 4. Existing baselines, metrics, experiments

Baselines: flat vector, BM25 (unicode-aware), hierarchical RAG, plus 12
component-ablation sets and 11 forgetting-lab variants (A–K). Metrics: MRR,
recall@8, token-F1, answer coverage/found, forgetting, retention, latency,
memory growth. Experiments emit JSON artifacts under `data_bench/`,
`data_lab/`, `experiments/`; every number in the paper regenerates from them.

## 5. Existing limitations (measured, not excuses)

- recall@8 ties flat vector on MuSiQue — the lead is ranking, not coverage.
- `graph` and `recency` components are inert on this corpus (delta exactly 0).
- Dense retrieval collapses on Devanagari with English-centric embedder;
  even multilingual backend trails BM25 at full IndicQA scale.
- Answer-stage F1 delta is statistically consistent but tiny in absolute terms.
- The forgetting lab measures retrieval interference with the LLM frozen —
  it does not measure parametric forgetting.

## 6. Extension points for NCM

1. **Membership term in retrieval** — follow the `stability` precedent: add
   `constellation` to `ALL_COMPONENTS`, weight 0 unless NCM enabled → all
   existing behavior unchanged (verified: no test asserts on the component
   list; `eval_ablation_real.py` derives minus-set from `ALL_COMPONENTS`
   dynamically, so a new component self-registers in ablations).
2. **Membership data** — `MemoryNode.metadata["constellations"]` (spec §4
   maps the schema; `metadata` already persists in SQLite via `to_row`).
   No new columns, no storage migration.
3. **Centroids** — k-means over node embeddings (`scikit-learn>=1.4` already
   a requirement); cached per store.
4. **Versioning** — new `core/ncm.py` `VersionChain` keyed by stable content
   hash; provenance via `MemoryEdge.provenance` and node `source_ids`.
5. **Mode switch** — config flags only (`ncm.enabled`, `ncm.top_k`, …);
   `MODE=classic_mira` remains the default behavior bit-for-bit.

## 7. Files to be created / modified (planned)

Create: `core/ncm.py`, `tests/test_ncm.py`, `docs/EXPERIMENT_PROTOCOL.md`,
`docs/RELATED_WORK.md`, this audit. Modify: `core/retrieval.py` (one optional
component, ~15 lines), `README.md` (what-NCM-is/isn't + enable instructions).
Nothing else. `scripts/build_forgetting_lab.py` and all committed artifacts
must remain untouched.

## 8. Files that must remain untouched

`data_lab/forgetting_results.json`, `data_bench/*`, committed experiment
manifests, `scripts/build_forgetting_lab.py`, `core/answer.py` (just hardened),
`tests/test_acceptance.py` semantics, existing baseline behaviors. Any change
here would invalidate the measured foundation the NCM program builds on.
