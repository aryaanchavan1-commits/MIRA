# ◎ MIRA — Mandala-Inspired Memory for AI Retrieval

**Arynox · by Aryan Chavan.** A **local-first research platform** for studying whether
a *radial, hierarchical, graph-based memory topology* — inspired by the organizational
principles of mandalas/yantras — improves AI memory **retrieval and storing**,
multi-hop reasoning, and grounded answering compared with conventional Vector RAG,
Graph RAG, and hierarchical memory.

**Measured headline (all artifacts in-repo, nothing hand-typed):** MIRA ranks
supporting evidence substantially higher than flat-vector and BM25 retrieval on two
real multi-hop datasets (MuSiQue MRR 0.70 vs 0.43/0.31; HotpotQA MRR 0.92 vs 0.68;
both p<0.001), produces significantly better end answers under an identical local-LLM
answer stage, and holds retrieval recall stable as the corpus grows. Full ablation,
scale sweep, and an Indic (Hindi/Marathi) evaluation included — with the language-
barrier diagnosis (embedding backend, not architecture) demonstrated and fixed.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.placeholder.svg)](https://doi.org/placeholder)
[![License](https://img.shields.io/badge/license-MIT-gold)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11-blue) ![FAISS](https://img.shields.io/badge/vector--index-FAISS-blueviolet)
![Benchmarks](https://img.shields.io/badge/benchmarks-MuSiQue%20·%20HotpotQA%20·%20IndicQA-gold)

> **What MIRA is NOT.** MIRA is an experimental architecture inspired by the
> organizational and visual principles of mandalas/yantras. It does **not**
> claim that historical mandalas were artificial neural networks or that
> ancient traditions contained modern AI technology.
>
> **The usefulness of radial organization is an empirical research question.**
> The system is built to prove the hypothesis wrong as easily as to support it.
> A negative result is a valid result.

---

## 1. Why this matters for the Indian AI revolution

India's AI buildout needs systems that work **offline-first, at near-zero cloud cost,
and in Indian languages** — a 4 GB GPU laptop should be enough to run a private,
citable research assistant. MIRA is built as evidence for that thesis:

- **Local-first by architecture**: retrieval, memory, and answering run fully on
  device; zero cloud calls by default (DPDP-aligned data sovereignty).
- **Frugal**: a 1.5B quantized model + MiniLM-class embedders deliver measured
  multi-hop gains; the memory layer — not model size — does the heavy lifting.
- **Indic-honest**: the IndicQA evaluation quantifies exactly where local-first
  Indic RAG breaks (the English-centric embedding backend), shows the ~10× fix from
  swapping to a multilingual embedder, and keeps unicode-aware BM25 as the measured
  default for Devanagari today.
- **Open and reproducible**: every number regenerates from committed scripts and
  artifacts — the same standard Indian academia and industry need to build on.

## 2. Research question

> Can a radial hierarchical memory topology provide measurable advantages for
> AI retrieval and reasoning compared with flat vector retrieval and ordinary
> graph/hierarchical retrieval when model, context budget, dataset, and
> hardware are held constant?

Primary hypothesis: combining semantic similarity, hierarchical organization,
radial proximity, graph relationships, importance, confidence, provenance, and
temporal information can improve evidence retrieval and multi-hop reasoning
while reducing irrelevant context. **This is tested, never assumed** — every
default weight in the config is marked *experimental*.

## 3. Research hypotheses and scientific scope

MIRA is currently a **symbolic memory and retrieval architecture**: a weighted graph
of memory records with semantic, radial, hierarchical, temporal, path, and activation
features. The mandala is an explicit organizational prior, not a claim that historical
mandalas were neural networks or that MIRA is a biological brain simulation.

### Transparent simulated affect

MIRA also keeps a small, deterministic **algorithmic affect state** attached to each
workspace. It records bounded `valence`, `arousal`, `confidence`, and `stress` values
from observable route, evidence, citation, and explicit-feedback signals. Every answer
includes an immutable snapshot and an explicit `simulated: true` disclosure. The state
is in-memory, resets when the process restarts, and never writes to or overrides the
mandala memory. It is not biological emotion, sentience, consciousness, or a learned
model of human feelings.

The state is deliberately inspectable rather than anthropomorphic. Grounded memory/web
answers increase confidence and reduce stress; parametric, no-evidence, and citation-
error signals lower confidence or raise stress; identity answers remain neutral because
they are deterministic metadata rather than retrieved evidence. `/api/affect` exposes
the current snapshot and `/api/affect/feedback` accepts only bounded, explicit deltas.

### H1 — radial organization

> With semantic embeddings, graph topology, context budget, and query set held fixed,
> adding radial placement and radial scoring improves retrieval or multi-hop evidence
> selection over vector-only and vector-plus-graph controls.

This is falsifiable: if the radial term is removed without a measurable loss, the
mandala-specific contribution is not supported.

### H2 — structured topology

> A structured graph with controlled radial/hierarchical relations improves paired
> retrieval over a deterministic degree-preserving rewiring and a descriptive
> regular square lattice, when node count and reported degree/edge statistics
> are disclosed. The lattice is not degree- or edge-matched.

The comparison is between **symbolic graph conditions**. It is not evidence of
head-direction cells, grid cells, or biological neural computation.

### H3 — coordinate-rotation null control

> Applying the same orthogonal transformation to every node embedding and query vector
> should preserve retrieval rankings for the current implementation, because angular
> coordinates are presentation metadata and are not consumed by the retriever.

A score of approximately 1.0 for paired top-k Jaccard is an implementation-invariance
result, not a learned rotational representation. A future biological-claims experiment
must use a separately specified recurrent model and preregister its metrics.

### Controlled benchmark

The headless benchmark is implemented in `evaluation/rotation.py` and exposed through
`scripts/run_rotational_benchmark.py`. It uses fixed timestamps, deterministic stub
embeddings, in-memory stores, no LLM, no network, and no live database:

```bat
.venv\Scripts\python.exe scripts\run_rotational_benchmark.py --replicates 3 --rotations 4 --output experiments\rotational_benchmark
```

Conditions:

| Condition | Meaning |
|---|---|
| `structured_full` | Deterministic structured graph with radial/hierarchical coordinates |
| `degree_preserving_shuffled` | Deterministic double-edge rewiring preserving edge count and degree sequence |
| `square_lattice` | Descriptive regular two-dimensional lattice comparator; not degree-matched |

Primary outputs are paired top-k Jaccard, top-1 agreement, recall@k, MRR, latency,
degree statistics, rewiring invariants, and a JSON manifest. The output explicitly
records that the study is **not a biological neural simulation**.

### Related neuroscience, used as context rather than equivalence

- Doeller, Barry, and Burgess (2010), *Nature*: evidence for grid-like signals in a
  human memory network.
- Nau et al. (2018), *Nature Neuroscience*: hexadirectional coding of visual space in
  human entorhinal cortex.
- Banino et al. (2018), *Nature*: grid-like representations emerging in artificial agents.
- Bronstein et al. (2021), *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics,
  and Gauges*.
- Cohen and Welling (2016), *Group Equivariant Convolutional Networks*.

These works motivate geometric and neuroscience-inspired hypotheses; they do not
establish that MIRA implements the same circuits.

### Indic-pattern extension

A future `data/kolam_patterns/` study should use images with explicit provenance,
artist/tradition attribution, and a documented license. Synthetic patterns may be used
for smoke tests, but must not be presented as authentic traditional kolam data. The
initial benchmark therefore does not fabricate cultural artifacts.

## 4. Architecture

```text
                  MIRA
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
    Semantic    Structural   Temporal
     Memory       Memory      Memory
       │           │           │
       └───────────┼───────────┘
                   ▼
           MANDALA TOPOLOGY   (rings 0-4+, sectors, radial distance)
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
     Rings       Sectors      Paths
                   │
                   ▼
              RADIAL SEARCH     (9-component score, per-component ablation)
                   │
                   ▼
             EVIDENCE GRAPH    (smallest high-confidence paths)
                   │
                   ▼
             CONTEXT COMPRESSION (dedup → rank → compress → provenance)
                   │
                   ▼
              SMALL SLM        (local GGUF via llama.cpp; extractive fallback)
                   │
                   ▼
                ANSWER         (memories + path + sources + metrics)
```

## 5. Memory representation

- **MemoryNode** — `id, concept, summary, raw_text, memory_type, parent_id,
  children, ring, sector, depth, radial_distance, embedding, importance,
  confidence, created_at, updated_at, valid_from, valid_until, source_ids,
  metadata`
- **MemoryEdge** — typed, weighted, confidence-scored, with provenance
- **MemoryFrame / MemoryRing / MemorySector / MemoryCluster** — topology views
- Ten memory types (semantic, episodic, procedural, working, fact, entity,
  event, document, concept, relation); the enum is extensible.

## 6. Mandala topology

- **Rings** — ring 0 = core/root concept, ring 1 = major concepts, ring 2 =
  subconcepts/entities, ring 3 = facts/evidence, ring 4+ = raw document
  evidence. Ring count is configurable (`topology.max_rings`).
- **Sectors** — *discovered*, not hard-coded: embedding clusters are named from
  member tokens; new domains generate new sector names automatically.
- **Radial distance** — configurable weighted mix:
  `radial = α·semantic + β·hierarchy + γ·graph + δ·temporal` (§12, experimental).
- **Placement strategies** (switchable, comparable in the UI):
  `embedding_clusters`, `graph_centrality`, `hierarchy`, `temporal`, `hybrid_mira`.

## 7. Retrieval algorithm

Score (all components individually switchable for ablation):

```text
score = α·semantic + β·structural + γ·radial + δ·graph
      + ε·importance + ζ·confidence + η·recency + θ·path + ι·activation
```

**Bio-NN-inspired activation (ι)** — production uses a bounded, deterministic
`lif_like` mode: seed energy leaks through a membrane accumulator, crosses a
threshold, resets, enters a refractory period, and propagates through a capped
sparse neighbor frontier. Each query exposes a bounded voltage/spike trace.
The older continuous mode remains available for comparison. This is an
algorithmic mechanism inspired by neural dynamics, not a biological brain
simulation.

**Hebbian/STDP-like consolidation (slow timescale).** Accepted grounded
answers can strengthen path edges, while a bounded timing rule uses the LIF
spike trace for STDP-like potentiation/depression and homeostatic decay.
Updates are clamped, persisted, and invalidate live graph/activation caches.
This is an engineering heuristic inspired by synaptic timing, not biochemical
STDP or proof that the update improved answer quality. A verified user
accept/reject or benchmark error signal is still required for predictive
plasticity.

**Learned weights (optional)** — one neural unit (9 sigmoid inputs, delta-rule
training, Widrow-Hoff 1960) can replace the hand-tuned coefficients. It is
disabled by default; enable it only after a held-out, document-grouped validation
run records the split, scorer hash, and effective weights. The learned-vs-hand
setting is itself ablatable.

**Sleep consolidation (offline, `scripts/consolidate.py`).** Three
biologically-motivated dynamics applied as an explicit, measurable pass —
never hidden magic: Ebbinghaus-style soft forgetting (importance fades with
time-since-use, slower for well-wired memories; never deletes), replay-based
reinforcement (recent retrievals re-fire through the Hebbian rule; used nodes
get an importance lift and a decay-clock reset), and gist abstraction
(clustered summaries stored one ring under the core, wired to member facts
with `gist_of` edges). Dry-run by default with a pre-write SQLite snapshot;
a ring-rate guard aborts if the workspace's ring-0/1 fraction is collapsed
(a real failure mode this repo hit and fixed). Measured on the 82,783-node
bench workspace: retention 0.9789, 507 gists → 1,521 member edges, retrieval
preserved (MRR 0.7003 → 0.6986, p=0.257) — harmless on a fresh corpus by
design; the benefit targets aged, repeatedly-used workspaces, which
`scripts/eval_aging.py` demonstrates directly: after simulating 60 days of
disuse on 80% of nodes, replayed memories *rise* (MRR 0.7014 → 0.7321) while
unreplayed ones fade (0.6972 → 0.6319) — sleep consolidation protects what
was used and lets the rest decay. The same pass is exposed as
`POST /api/memory/consolidate` (dry-run default; `{"apply": true}` persists
after a snapshot backup).

**BioMIRA — the optional adaptive layer (`core/biomira.py`).** Behind
`BIOMIRA_ENABLED` (config `biomira.enabled`): adaptive stability, LIF-inspired
sparse activation, Hebbian and STDP-inspired association on *existing* graph
edges, homeostatic normalization, adaptive decay, the
`NEW → CANDIDATE → STABLE → CONSOLIDATED` state ladder, a bounded replay
buffer (at-risk, important and representative memories only), and
activation-driven ring migration. Off means off — no writes, and
`kappa_stability` defaults to 0 so the baseline ranking is unchanged, which
is asserted in `tests/test_biomira.py`.

The Catastrophic Forgetting Lab (`scripts/build_forgetting_lab.py` +
`scripts/eval_forgetting.py`) ingests four sequential corpora built from real
MuSiQue paragraphs (899 docs, 18,632 nodes, 300 questions) and re-tests every
task after every arrival. Result on that corpus — stated plainly because it is
not the flattering one:

| variant | final MRR | avg forgetting | Δ MRR vs MIRA (95% CI, p) |
|---|---|---|---|
| A vector RAG | 0.4921 | 0.0409 | −0.2940 [−0.350, −0.237], p<0.001 |
| **B MIRA** | **0.7861** | **0.0356** | baseline |
| C MIRA + radial | 0.8069 | 0.0400 | +0.0209 [+0.004, +0.041], p=0.015 |
| D MIRA + graph | 0.4971 | 0.0412 | −0.2890 [−0.345, −0.232], p<0.001 |
| E MIRA + hierarchy | 0.8013 | 0.0332 | +0.0152 [−0.004, +0.036], p=0.129 |
| F MIRA + decay | 0.7861 | 0.0356 | +0.0000, p=1.0 |
| G MIRA + consolidation | 0.7861 | 0.0356 | +0.0000, p=1.0 |
| H MIRA + replay | 0.7876 | 0.0351 | +0.0016 [−0.002, +0.006], p=0.466 |
| I MIRA + bio dynamics | 0.7739 | 0.0126 | −0.0121 [−0.035, +0.010], p=0.278 |
| J full BioMIRA | 0.7634 | 0.0231 | **−0.0227 [−0.046, −0.003], p=0.028** |

So: **the full adaptive layer is significantly worse than plain MIRA on
retrieval**, and no individual mechanism helps at all. What it does buy is
retention — average forgetting drops 0.0356 → 0.0126 (−65%) with the dynamics
variant, mean retention 0.958 → 0.984. The layer trades peak retrieval
precision for resistance to interference; that trade only pays under a
staleness objective. The zero rows for decay and consolidation are a property
of this corpus (every node shares one ingest time, so decay rescales
importance almost uniformly and cannot reorder anything) — the same decay
clearly discriminates in the aged-workspace experiment above. Exposed in the
console as **Biological Memory** and **Catastrophic Forgetting Lab**, with
per-memory "why was this retrieved / why did it decay / why is it
consolidated" readouts on every node.

Pipeline: query analysis → embedding → semantic (FAISS) → seed selection →
graph expansion with available path provenance → **spreading activation** →
hierarchical candidates → merge → score → rerank → path selection → evidence
selection → compression → SLM → answer. Answers show only the auditable
retrieval path and cited evidence — never a hidden chain-of-thought.

### Agent routing (core/agent.py)

Every query is routed by evidence strength, and the route is labeled on the
answer:

```text
query ──▶ identity route? ── yes ──▶ deterministic project identity
            │ no
            ▼
         retrieve from the mandala
            │ strong evidence (cosine ≥ gate, real context)
            ▼                      
        MEMORY ANSWER (cited)          ← default path
            │ weak / no evidence
            ▼
   web consented? ── no ──▶ PARAMETRIC ANSWER (model knowledge, labeled ungrounded)
            │ yes
            ▼
   live search → ingest top pages → re-retrieve → WEB ANSWER (cited, now
   permanent memories — the agent literally grows its mandala to learn)
```

Creator questions are answered deterministically as: MIRA was made by Aryan
Chavan, a Bio-NN-inspired mandala-based symbolic memory and retrieval research
system. This is project metadata, not retrieved evidence.

**Affect integration.** After routing, `AgentPipeline` updates the workspace-owned
state and freezes an immutable snapshot on the returned `Answer`. The update uses only
observable answer signals, applies a fixed neutral-baseline decay, and is advisory: it
cannot change retrieval, memory placement, citations, or the mandala. The web console
shows the snapshot in Overview and each answer; the legacy Streamlit chat shows the same
bounded values and disclosure.

## 8. Baselines and ablations

| System | What it is |
|---|---|
| Baseline A `vector_rag` | naive FAISS cosine retrieval |
| Baseline B `graph_rag` | vector seeds + degree-weighted graph expansion |
| Baseline C `hierarchical_rag` | coarse-to-fine tree descent |
| Baseline D `full_mira` | all 9 components |

Ablation sets: `vector_only`, `graph_only`, `hierarchy_only`, `radial_only`,
`vector_graph`, `vector_hierarchy`, `vector_radial`, `graph_hierarchy`,
`graph_radial`, `activation_only`, `vector_activation`, `full_mira` — runnable
per-query (Research Lab) or over a dataset (Benchmarks). All systems share the
same frame, stores, embedding model, dataset, context budget, and hardware. The
topology benchmark additionally reports degree and edge-count invariants for every
condition.

## 9. Metrics (computed, never fabricated)

Retrieval recall@k, precision@k, MRR, answer token-F1 (lexical proxy — stated
as such, not an LLM judge), context tokens, compression ratio, latency,
candidates count, throughput. Resource usage (RAM/VRAM) is measured live on
the Hardware page.

## 10. Installation

```bat
git clone <repo> MIRA
cd MIRA
setup.bat
run.bat
```

`setup.bat` creates `.venv`, installs pinned dependencies (temp/cache on the
project drive), initializes SQLite, and validates imports. It never modifies
your system Python. No Docker required.

`run.bat` starts the **web console** (FastAPI + a dependency-free HTML/CSS/JS
frontend) at `http://127.0.0.1:8000` and opens your browser. The frontend is
hand-built vanilla JS with a canvas mandala — no node_modules, no build step,
fully offline. (`run.bat streamlit` starts the legacy Streamlit UI instead.)
The HTTP API is documented at `/api/docs` when the server is running. The affect
state is available at `GET /api/affect`; bounded explicit feedback is accepted at
`POST /api/affect/feedback` (or the short `POST /api/affect` form). These endpoints
update only the simulated state, never the mandala.

Command-line experiments (reproducible, headless):

```bat
.venv\Scripts\python.exe scripts/run_experiment.py --dataset data/datasets/custom.json --limit 50 --judge
.venv\Scripts\python.exe paper/export_results.py --exp EXP-0005
.venv\Scripts\python.exe paper/export_results.py --allow-smoke --exp EXP-0005
.venv\Scripts\python.exe scripts/run_rotational_benchmark.py --replicates 3 --rotations 4 --output experiments/rotational_benchmark
```

The paper draft lives in `paper/mira_paper.md`; its result tables are
generated **only** from saved experiment directories — never written by hand.

## 11. Hardware requirements & auto-configuration

Works from CPU-only laptops upward. At startup MIRA detects CPU/RAM/GPU/VRAM/
CUDA/disk and selects a safe runtime: smallest-fitting GGUF (0.5B–3B, Q4/Q5),
context length, GPU layers, threads, batch size — always with headroom
(`usable = total × fraction`, never 100%). If something doesn't fit it steps
down; it never crashes merely because the GPU has 4 GB VRAM. Performance
modes: `safe | balanced | fast | research`.

The reference development machine is an RTX 3050 Laptop (4 GB) with 16 GB RAM;
nothing is hard-coded to it.

## 12. Offline mode

`offline: true` (default) makes MIRA strictly local: no external APIs, no
telemetry, no downloads, no network calls. Downloads additionally require
`models.allow_download: true` plus explicit UI confirmation with a precheck of
disk and memory budgets.

**Live web search (opt-in, consent-gated).** Web access requires
`offline: false`, `web_search.enabled: true` (or an explicit per-request
boolean consent), and a public http(s) fetch target. The Web Search view can
query the live web through pluggable backends, tried in priority order:

1. **agent-reach** (`Panniantong/Agent-Reach`) — unified read/search across
   Twitter/X, Reddit, YouTube, GitHub and more, zero API keys, if installed on PATH
2. **opencli** (`jackwener/OpenCLI`) — sites-as-CLI through your logged-in
   Chrome, if installed on PATH
3. **DuckDuckGo HTML** — stdlib fallback, always available

Search results are plain data (never executed). One click ingests a page into
the mandala: chunked → embedded → memory nodes/edges → placed on rings and
sectors with URL provenance. While offline mode stays on, no search is possible;
the gate is deliberate. Fetch URLs are checked against public DNS/IP ranges to
reduce SSRF risk.

## 13. Training (optional, safety-gated)

Fine-tuning is **never** automatic. `training/lora.py` runs a precheck
(VRAM ≥ 5 GB usable, RAM, disk, dataset ≥ 50 samples, GGUF base) and prints
`SKIP TRAINING` with the reason when unmet — on the 4 GB reference machine it
will skip, and the system keeps working with external memory. The dataset
builder emits only provenance-grounded samples (`query, memory_context,
retrieval_path, evidence, answer`), each traced to real ingested chunks.

## 14. Benchmarking

1. Ingest documents (Documents page).
2. Create a dataset — local JSON/JSONL of
   `{"question", "answer", "supporting_ids?"}` (Benchmarks page can generate
   a clearly-labeled synthetic one from your chunks).
3. Run benchmarks — all baselines + all ablations, one table. The web API accepts
   dataset files only from `data/datasets/`; the CLI remains a local-file tool.
4. Save — each experiment lands in `experiments/EXP-XXXX/` with `config.json`
   (study type, available runtime metadata, seed, and declared provenance),
   `results.json`, `results.csv`, `summary.md`, plus a row in SQLite.
5. Compare/export on the Experiments page (JSON/CSV/Markdown). The paper exporter
   rejects smoke runs unless `--allow-smoke` is explicitly supplied.

See `experiments/README.md` for the artifact inventory, historical-run labels,
geometry-ablation commands, and the rule that generated output is never
hand-edited. Simulated affect is advisory and is not part of retrieval metrics;
its deterministic checks live in `tests/test_affect.py`.

Compatible external datasets (LoCoMo, LongMemEval, HotpotQA, 2WikiMultiHopQA,
MuSiQue) can be converted to the record format locally — nothing is
downloaded without confirmation.

### 14.1 Real-data benchmark (measured)

`scripts/build_musique_bench.py` builds a 300-question answerable 2-hop MuSiQue set;
`scripts/build_bench_workspace.py` ingests its 4,043 paragraphs into an isolated
`data_bench/` workspace (82,783 nodes, bulk FAISS build); `scripts/eval_benchmark_real.py`
runs mira_full vs flat_vector vs hierarchical_rag (3 seeds, k=8) with paired-bootstrap
significance (`scripts/eval_significance.py`). Latest measured outcome
(`paper/results_real.md`, auto-generated — never hand-edited):

| system | MRR | recall@8 |
|---|---|---|
| mira_full | 0.7016 | 0.2991 |
| flat_vector | 0.4347 | 0.3066 |
| hierarchical_rag | 0.2367 | 0.0806 |

MRR difference significant (Δ +0.267, 95% CI [0.224, 0.311], p<0.001); recall@8
tied (p≈0.52) — MIRA ranks supporting evidence higher without surfacing more of it.
Retrieval is deterministic given the seed (all 3 seeds identical). Additional measured
evidence, all exported to `paper/results_real.md`:

- **HotpotQA replication** (`--dataset hotpotqa`, 17,962-node `data_hotpot/`
  workspace, 300 bridge questions): MRR 0.9175 vs 0.6774 flat (Δ+0.240, p<0.001);
  recall@8 slightly favors flat (p=0.007) — the MuSiQue shape replicates.
- **BM25 control** (`scripts/add_bm25_baseline.py`, pure-Python Okapi): MRR 0.3066 —
  below flat-vector, so MIRA's lead over it is Δ+0.395 (p<0.001).
- **Answer stage** (`scripts/eval_answers_standalone.py`, Qwen2.5-1.5B forced,
  identical compression + prompt): MIRA token-F1 0.076 vs flat 0.039
  (Δ+0.066, CI [0.013, 0.123], p≈0.009) — the retrieval advantage survives
  into end answers under a fixed local LLM.
- **IndicQA hi/mr** (`scripts/build_indic_bench.py` + `scripts/eval_indic_retrieval.py`):
  constructive negative result — the English-centric MiniLM embedder collapses on
  Devanagari (dense MRR 0.02-0.04) while unicode-aware BM25 stays robust (0.39-0.45).
  Local-first Indic RAG needs a multilingual embedding backend; BM25 is the default
  there today.
- **Ablation, scale sweep, neural validation**: `scripts/eval_ablation_real.py`
  (leave-one-out over all 9 components), `scripts/eval_scale_strategies.py`
  (aware-vs-blind placement across a size ladder), `scripts/eval_neural_validated.py`
  (doc-grouped holdout for the learned scorer; config never auto-flips).

## 15. Reproducibility

CLI research runs record the command, dataset hash, resolved configuration,
embedding/LLM metadata, hardware, software versions, seed, git commit, dirty-tree
state, study type, and full results. Older or UI-created smoke manifests may not
contain every field and must not be treated as publication-ready evidence.

## 16. Applications across AI fields

The memory layer is model-agnostic: anything that needs an LLM or agent to
**remember, rank, and cite** its own growing corpus — locally — can sit on
top of it. Measured properties that transfer:

- **Higher-precision evidence ranking** (MRR +0.24–0.27 over flat/bm25 on two
  multi-hop datasets) → fewer wrong citations in any grounded-answer system.
- **Recall stability under scale** (hybrid placement held recall while blind
  placement degraded) → corpora that grow (support bots, research notes).
- **Component attribution** (every retrieval signal is independently
  ablatable) → tune the memory to a domain by measurement, not intuition.
- **Frugal/local-first** (4GB GPU, zero cloud calls, DPDP-friendly) →
  deployment where data cannot leave the device.

Concrete fits: **personal knowledge assistants** (long-lived notes with
cited answers), **customer-support copilots** (product docs, ticket history),
**legal/medical reference retrieval** (citation-first answers, on-premise),
**education tech in Indian languages** (with a multilingual embedder; BM25
fallback measured), **agent memory** (episodic + semantic with spreading-
activation recall), **edge/defense/offline** deployments (no-network
capability is architectural, not an afterthought), and **research
reproducibility** itself (the bench/builder/exporter loop).

## 17. Positioning against prior work

- **GraphRAG** (Edge et al., 2024) retrieves over LLM-extracted entity
  communities; **HippoRAG** (Wang et al., 2024) over Personalized PageRank
  associations; **RAPTOR** (Sarthi et al., 2024) over recursive summary trees.
  Each adds one organizational axis to dense retrieval.
- MIRA adds the **radial axis** (rings/sectors + a formal radial distance) as
  one switchable component, and measures whether it carries weight *beyond*
  graph and hierarchy — the ablation design exists to answer exactly that.
- Baseline B is deliberately weaker than GraphRAG (degree expansion, not
  community summaries): it establishes a floor, not a ceiling. Beating it is
  necessary but not sufficient for claims; the paper's threats-to-validity
  section states this plainly.

A full related-work discussion and formal mechanism definitions are in
`paper/mira_paper.md`.

## 18. How to interpret results

- Compare `full_mira` against `vector_only` as a scoring-arm smoke check; it
  does not isolate candidate-generation effects until the candidate pool is frozen.
- Jaccard overlap shows whether extra components change retrieval for a query;
  identical retrieval is evidence of no measured difference, not proof of a win.
- Token-F1 is a proxy. Publishable answer claims require a shared answer/context
  stage, held-out evidence labels, and an independent or human evaluation.
- Report negative results as they are. Do not tune weights on the test set.

## 19. Limitations

- NetworkX centrality is O(n) per compute — fine for laptop-scale corpora,
  swap `storage/graph_store.py` for larger graphs.
- answer_token_f1 is lexical; it rewards surface overlap, not reasoning.
- Sector naming is lexical (top tokens); it can produce imperfect labels.
- The synthetic chunk-derived benchmark measures lexical memorization, not
  multi-hop reasoning — build real QA sets for real claims.
- Simulated affect is a fixed, inspectable state machine. It is workspace-scoped and
  resets on restart; it is not evidence of emotion, sentience, or human-like feeling.
- llama-cpp-python PyPI wheels are CPU-only; GPU offload needs a CUDA wheel.
- The forgetting lab measures *retrieval interference* with the LLM frozen, not
  parametric catastrophic forgetting. Four sequential tasks on one corpus is a
  small sample for a retention claim.
- BioMIRA did not beat plain MIRA on this corpus (§7). It is shipped because a
  documented negative result with an ablation ladder is worth more than an
  unmeasured feature — not because the layer works.

## 20. Project structure

```text
MIRA/
├── run.bat / setup.bat / app.py
├── config/          config.yaml, auto_config.py
├── core/            hardware, memory, mandala, placement, retrieval,
│                    ranking, compression, conflicts, updater, answer,
│                    workspace, affect, memory_dynamics, biomira
├── ingestion/       loaders, chunker, extractors, pipeline
├── models/          model_manager, llm, embeddings (+ models/*.gguf)
├── storage/         sqlite_store, vector_store, graph_store
├── baselines/       vector_rag, graph_rag, hierarchical_rag
├── evaluation/      benchmark, metrics, ablation, report, rotation,
│                    forgetting (retention / forgetting definitions)
├── training/        dataset_builder, lora, evaluator
├── visualization/   mandala_view (plotly polar)
├── ui/pages/        Dashboard, Chat, Mandala, Memory Explorer, Documents,
│                    Research Lab, Benchmarks, Experiments, Models,
│                    Hardware, Settings, Logs
├── web/             console (landing page + operator console) + js/css
├── scripts/         reproducible runners: benchmark, ablation, significance,
│                    aging, consolidate, build_forgetting_lab, eval_forgetting
├── tests/           phase test suites (assert-based, no frameworks)
├── data/            mira.db, indexes/, uploads/, datasets/
├── data_bench/      isolated MuSiQue bench corpus (4,043 paragraphs)
├── data_lab/        sequential forgetting-lab corpus + results
├── experiments/     EXP-0001/ ... · README.md · geometry wrappers
└── logs/            mira.log
```

## 21. License

MIT — see `LICENSE`.
