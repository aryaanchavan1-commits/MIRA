# ◎ MIRA — Mandala-Inspired Memory Architecture

A **local-first research prototype** for studying whether a *radial, hierarchical,
graph-based memory topology* — inspired by the organizational principles of
mandalas/yantras — improves AI memory retrieval and multi-hop reasoning compared
with conventional Vector RAG, Graph RAG, and hierarchical memory.

> **What MIRA is NOT.** MIRA is an experimental architecture inspired by the
> organizational and visual principles of mandalas/yantras. It does **not**
> claim that historical mandalas were artificial neural networks or that
> ancient traditions contained modern AI technology.
>
> **The usefulness of radial organization is an empirical research question.**
> The system is built to prove the hypothesis wrong as easily as to support it.
> A negative result is a valid result.

---

## 1. Research question

> Can a radial hierarchical memory topology provide measurable advantages for
> AI retrieval and reasoning compared with flat vector retrieval and ordinary
> graph/hierarchical retrieval when model, context budget, dataset, and
> hardware are held constant?

Primary hypothesis: combining semantic similarity, hierarchical organization,
radial proximity, graph relationships, importance, confidence, provenance, and
temporal information can improve evidence retrieval and multi-hop reasoning
while reducing irrelevant context. **This is tested, never assumed** — every
default weight in the config is marked *experimental*.

## 2. Architecture

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
             RADIAL SEARCH     (8-component score, per-component ablation)
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

## 3. Memory representation

- **MemoryNode** — `id, concept, summary, raw_text, memory_type, parent_id,
  children, ring, sector, depth, radial_distance, embedding, importance,
  confidence, created_at, updated_at, valid_from, valid_until, source_ids,
  metadata`
- **MemoryEdge** — typed, weighted, confidence-scored, with provenance
- **MemoryFrame / MemoryRing / MemorySector / MemoryCluster** — topology views
- Ten memory types (semantic, episodic, procedural, working, fact, entity,
  event, document, concept, relation); the enum is extensible.

## 4. Mandala topology

- **Rings** — ring 0 = core/root concept, ring 1 = major concepts, ring 2 =
  subconcepts/entities, ring 3 = facts/evidence, ring 4+ = raw document
  evidence. Ring count is configurable (`topology.max_rings`).
- **Sectors** — *discovered*, not hard-coded: embedding clusters are named from
  member tokens; new domains generate new sector names automatically.
- **Radial distance** — configurable weighted mix:
  `radial = α·semantic + β·hierarchy + γ·graph + δ·temporal` (§12, experimental).
- **Placement strategies** (switchable, comparable in the UI):
  `embedding_clusters`, `graph_centrality`, `hierarchy`, `temporal`, `hybrid_mira`.

## 5. Retrieval algorithm

Score (all components individually switchable for ablation):

```text
score = α·semantic + β·structural + γ·radial + δ·graph
      + ε·importance + ζ·confidence + η·recency + θ·path + ι·activation
```

**Spreading activation (ι)** — the bio-inspired component (Collins & Loftus 1975;
ACT-R). Retrieved seeds inject activation energy into the memory graph; energy
spreads along edges with per-hop decay and fan-out normalization (synaptic
scaling: busy hubs don't over-fire). Associates reached through energy
propagation surface as candidates even when vector similarity misses them.
This is a *mechanism inspired by neural activation dynamics*, not a simulation
of a biological brain.

**Hebbian consolidation (slow timescale).** Edges along successful multi-hop
retrieval paths strengthen ("neurons that fire together wire together",
Hebb 1949) while all edges decay slightly each pass (homeostasis) — so the
memory's wiring literally adapts to what it is used for. Bounded, logged,
persisted; toggle with `memory.hebbian`.

**Learned weights (optional)** — one neural unit (9 sigmoid inputs, delta-rule
training, Widrow-Hoff 1960) can replace the hand-tuned coefficients. Train on
corpus QA with `python scripts/train_neural.py`, then set
`retrieval_score.learned: true`. The learned weights are inspectable and the
learned-vs-hand setting is itself ablatable.

Pipeline: query analysis → embedding → semantic (FAISS) → seed selection →
graph expansion with explicit multi-hop paths → **spreading activation** →
hierarchical candidates → merge → score → rerank → path selection → evidence
selection → compression → SLM → answer. Answers show only the auditable
retrieval path and cited evidence — never a hidden chain-of-thought.

### Agent routing (core/agent.py)

Every query is routed by evidence strength, and the route is labeled on the
answer:

```text
query ──▶ retrieve from the mandala
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

## 6. Baselines and ablations

| System | What it is |
|---|---|
| Baseline A `vector_rag` | naive FAISS cosine retrieval |
| Baseline B `graph_rag` | vector seeds + degree-weighted graph expansion |
| Baseline C `hierarchical_rag` | coarse-to-fine tree descent |
| Baseline D `full_mira` | all 8 components |

Ablation sets: `vector_only`, `graph_only`, `hierarchy_only`, `radial_only`,
`vector_graph`, `vector_hierarchy`, `vector_radial`, `graph_hierarchy`,
`graph_radial`, `full_mira` — runnable per-query (Research Lab) or over a
dataset (Benchmarks). All systems share the same frame, stores, embedding
model, dataset, context budget, and hardware.

## 7. Metrics (computed, never fabricated)

Retrieval recall@k, precision@k, MRR, answer token-F1 (lexical proxy — stated
as such, not an LLM judge), context tokens, compression ratio, latency,
candidates count, throughput. Resource usage (RAM/VRAM) is measured live on
the Hardware page.

## 8. Installation

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
The HTTP API is documented at `/api/docs` when the server is running.

Command-line experiments (reproducible, headless):

```bat
.venv\Scripts\python.exe scripts/run_experiment.py --dataset data/datasets/custom.json --limit 50 --judge
.venv\Scripts\python.exe paper/export_results.py   -- rematerialize paper tables
```

The paper draft lives in `paper/mira_paper.md`; its result tables are
generated **only** from saved experiment directories — never written by hand.

## 9. Hardware requirements & auto-configuration

Works from CPU-only laptops upward. At startup MIRA detects CPU/RAM/GPU/VRAM/
CUDA/disk and selects a safe runtime: smallest-fitting GGUF (0.5B–3B, Q4/Q5),
context length, GPU layers, threads, batch size — always with headroom
(`usable = total × fraction`, never 100%). If something doesn't fit it steps
down; it never crashes merely because the GPU has 4 GB VRAM. Performance
modes: `safe | balanced | fast | research`.

The reference development machine is an RTX 3050 Laptop (4 GB) with 16 GB RAM;
nothing is hard-coded to it.

## 10. Offline mode

`offline: true` (default) makes MIRA strictly local: no external APIs, no
telemetry, no downloads, no network calls. Downloads additionally require
`models.allow_download: true` plus explicit UI confirmation with a precheck of
disk and memory budgets.

**Live web search (opt-in, consent-gated).** When `web_search.enabled: true`
(or per-request consent), the Web Search view can query the live web through
pluggable backends, tried in priority order:

1. **agent-reach** (`Panniantong/Agent-Reach`) — unified read/search across
   Twitter/X, Reddit, YouTube, GitHub and more, zero API keys, if installed on PATH
2. **opencli** (`jackwener/OpenCLI`) — sites-as-CLI through your logged-in
   Chrome, if installed on PATH
3. **DuckDuckGo HTML** — stdlib fallback, always available

Search results are plain data (never executed). One click ingests a page into
the mandala: chunked → embedded → memory nodes/edges → placed on rings and
sectors with URL provenance. While offline mode stays on, no search is possible
— the gate is deliberate.

## 11. Training (optional, safety-gated)

Fine-tuning is **never** automatic. `training/lora.py` runs a precheck
(VRAM ≥ 5 GB usable, RAM, disk, dataset ≥ 50 samples, GGUF base) and prints
`SKIP TRAINING` with the reason when unmet — on the 4 GB reference machine it
will skip, and the system keeps working with external memory. The dataset
builder emits only provenance-grounded samples (`query, memory_context,
retrieval_path, evidence, answer`), each traced to real ingested chunks.

## 12. Benchmarking

1. Ingest documents (Documents page).
2. Create a dataset — local JSON/JSONL of
   `{"question", "answer", "supporting_ids?"}` (Benchmarks page can generate
   a clearly-labeled synthetic one from your chunks).
3. Run benchmarks — all baselines + all ablations, one table.
4. Save — each experiment lands in `experiments/EXP-XXXX/` with `config.json`
   (hardware, versions, git commit, seed, weights), `results.json`,
   `results.csv`, `summary.md`, plus a row in SQLite.
5. Compare/export on the Experiments page (JSON/CSV/Markdown).

Compatible external datasets (LoCoMo, LongMemEval, HotpotQA, 2WikiMultiHopQA,
MuSiQue) can be converted to the record format locally — nothing is
downloaded without confirmation.

## 13. Reproducibility

Every experiment stores: experiment_id, timestamp, git commit, model +
quantization, embedding model, dataset, retrieval parameters and weights,
hardware profile, software versions, random seed, and full results.

## 14. Positioning against prior work

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

## 15. How to interpret results

- Compare `full_mira` against `vector_only` first — that isolates the
  contribution of everything beyond pure similarity.
- Jaccard overlap in the Research Lab shows *whether* the extra components
  change retrieval at all; identical retrieval means no benefit for that query.
- Token-F1 is a proxy; for publishable claims add an LLM judge or human
  evaluation on top.
- Report negative results as they are. Do not tune weights on the test set.

## 15. Limitations

- NetworkX centrality is O(n) per compute — fine for laptop-scale corpora,
  swap `storage/graph_store.py` for larger graphs.
- answer_token_f1 is lexical; it rewards surface overlap, not reasoning.
- Sector naming is lexical (top tokens); it can produce imperfect labels.
- The synthetic chunk-derived benchmark measures lexical memorization, not
  multi-hop reasoning — build real QA sets for real claims.
- llama-cpp-python PyPI wheels are CPU-only; GPU offload needs a CUDA wheel.

## 16. Project structure

```text
MIRA/
├── run.bat / setup.bat / app.py
├── config/          config.yaml, auto_config.py
├── core/            hardware, memory, mandala, placement, retrieval,
│                    ranking, compression, conflicts, updater, answer, workspace
├── ingestion/       loaders, chunker, extractors, pipeline
├── models/          model_manager, llm, embeddings (+ models/*.gguf)
├── storage/         sqlite_store, vector_store, graph_store
├── baselines/       vector_rag, graph_rag, hierarchical_rag
├── evaluation/      benchmark, metrics, ablation, report
├── training/        dataset_builder, lora, evaluator
├── visualization/   mandala_view (plotly polar)
├── ui/pages/        Dashboard, Chat, Mandala, Memory Explorer, Documents,
│                    Research Lab, Benchmarks, Experiments, Models,
│                    Hardware, Settings, Logs
├── tests/           phase test suites (assert-based, no frameworks)
├── data/            mira.db, indexes/, uploads/, datasets/
├── experiments/     EXP-0001/ ...
└── logs/            mira.log
```

## 17. License

MIT — see `LICENSE`.
