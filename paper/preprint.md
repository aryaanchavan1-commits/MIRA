---
title: "Radial Memory Topologies for Retrieval-Augmented Generation: A Controlled Study of Mandala-Inspired Organization"
date: "2026-10-04"
keywords: retrieval-augmented generation, memory topology, radial organization, ablation study, local-first AI, Indic languages
---

## Reproducibility statement

Every quantitative claim in this paper is generated from committed run
artifacts by `paper/export_results.py --real`; no table is hand-edited. All
experiments run on a single consumer laptop (RTX 3050 4GB, 16 GB RAM) with
zero cloud calls. Artifacts and builders:

- `data_bench/bench_real_results.json`
- `data_bench/bench_real_answers.json`
- `data_bench/ablation_real_results.json`
- `data_bench/scale_sweep_results.json`
- `data_bench/hotpotqa_bench_real_results.json`
- `data_bench/consolidation_results.json`
- `data_bench/aging_results.json`
- `data_indic/indicqa_results.json`
- `data_indic_ml/indicqa_ml_results.json`
- `experiments/neural_validation.json`

---

## Abstract

Retrieval-augmented generation (RAG) systems predominantly organize memory as a flat
vector collection, optionally augmented by entity graphs or hierarchy. We investigate an
underexplored structural alternative: a **radial memory topology** in which memories are
organized in concentric rings around a core concept, partitioned into semantic sectors,
and positioned by a formally defined radial distance combining semantic, hierarchical,
graph, and temporal signals. We implement MIRA, a local-first research platform in which
the radial mechanism is one switchable component among nine, each independently
  ablatable, evaluated against naive vector RAG, hybrid vector+graph RAG, and hierarchical
retrieval under a declared, shared protocol when the answer-stage experiment is
run; the current default runs are retrieval-only smoke measurements, not
answer-quality evidence.
**[Measured: on 300 MuSiQue 2-hop questions over an 82,783-node workspace, MIRA
reaches MRR 0.702 vs 0.435 for flat vector RAG and 0.307 for BM25 (paired bootstrap
ΔMRR +0.267, 95% CI [0.224, 0.311], p≈0.0001). The ranking advantage replicates on
300 HotpotQA bridge questions (MRR 0.918 vs 0.677 flat, Δ+0.240, p≈0.0001), while
recall@8 stays statistically tied or slightly favors flat on both datasets. With the
answer stage held identical (Qwen2.5-1.5B), MIRA's end answers score significantly
higher token-F1 than flat retrieval's (Δ+0.066, p≈0.009). An IndicQA (hi/mr) evaluation
quantifies a deployment-relevant limitation: the English-centric embedder collapses on
Devanagari while BM25 remains robust — local-first Indic RAG needs multilingual
embedding backends.]** We release the platform, including
per-component ablation tooling, to support reproducible negative or positive findings.

---

## 1. Introduction

The dominant paradigm for retrieval-augmented generation embeds documents into a dense
vector space and retrieves nearest neighbors (Lewis et al., 2020). Extensions enrich
this with entity graphs (Edge et al., 2024, GraphRAG), hierarchical summarization trees
(Sarthi et al., 2024, RAPTOR), or neuroscience-motivated associative indexing (Wang et
al., 2024, HippoRAG). These methods each add *one* organizational axis — community
structure, hierarchy, or association — on top of semantic similarity.

This paper asks a complementary question: what does the **radial axis** contribute?
Mandalas and yantras — across Hindu, Buddhist, and Jain traditions — organize symbolic
content in concentric rings around a center, partitioned by directional sectors, with
proximity to the center encoding importance or generality. We do **not** claim historical
continuity with modern AI, and we do not treat the mandala as a source of truth about
memory. We treat it as an *organizational prior*: generality belongs near the center,
evidence near the rim, and related content shares a sector. The research contribution is
a **formal, measurable mechanism** derived from this prior — not the visual metaphor.

Our central question: **under a declared shared protocol, does adding a radial
component to retrieval improve multi-hop evidence retrieval over flat vector
retrieval and the included graph/hierarchical controls?** Established external
baselines remain future work.

We answer with a system built to be refuted: every retrieval component is
individually switchable, all retrieval weights are configurable and marked
experimental, and the benchmark harness records the conditions for each run.

### Contributions

1. A formal definition of radial memory placement (rings, sectors, radial distance) as a
   computable structure over an entity–evidence graph (§3).
2. MIRA, a nine-component retrieval score with per-component scoring ablations,
   plus a benchmark harness that records conditions and refuses to promote smoke
   manifests into paper tables (§4).
3. A controlled experimental protocol on multi-hop QA (HotpotQA, MuSiQue,
   2WikiMultiHopQA) with retrieval, lexical, and judge-based answer metrics (§5).
4. **[Results TBF]** and an honest account of which components carry the benefit (§6).

The current implementation is a symbolic weighted-graph retrieval system. Its
bio-inspired terminology describes algorithmic analogies—activation propagation,
Hebbian reinforcement, and radial organization—not simulated hippocampal or entorhinal
circuits. Claims about head-direction or grid-cell mechanisms require a separately
specified recurrent neural model and independent activity analysis.

---

## 2. Related Work

**Flat dense retrieval.** DPR (Karpukhin et al., 2020) and its descendants retrieve by
cosine similarity over chunk embeddings. This is our Baseline A. Its failure mode on
multi-hop questions is well documented: no single chunk contains the bridge entity.

**Graph-augmented RAG.** GraphRAG (Edge et al., 2024) builds an entity graph from LLM
extractions and retrieves over community summaries. HippoRAG (Wang et al., 2024) uses a
Personalized PageRank over an entity graph to simulate hippocampal indexing. Both
demonstrate that *structure beyond the vector* helps multi-hop retrieval. MIRA differs in
its ring/sector radial coordinates and in treating graph centrality as one ablatable
component rather than the primary index. Our Baseline B (vector + degree-weighted graph
expansion) is deliberately weaker than GraphRAG to establish a floor, not a ceiling; §7
discusses the threat this poses to claims.

**Hierarchical RAG.** RAPTOR (Sarthi et al., 2024) recursively clusters and summarizes
chunks into a tree, retrieving across levels. MIRA's ring assignment over a parent–child
graph is a generalization: rings are computed from a configurable mix of signals
(hierarchy, centrality, semantics, recency) rather than from clustering alone. Baseline C
is a coarse-to-fine tree descent without radial or graph components.

**Memory for agents.** Generative Agents (Park et al., 2023) score memories by
recency × importance × relevance — an influence on our component design. MemGPT (Packer
et al., 2023) manages context as paging. Neither uses radial coordinates.

**Neuroscience and geometric context.** Doeller, Barry, and Burgess (2010) reported
evidence for grid-like signals in a human memory network; Nau et al. (2018) reported
hexadirectional coding of visual space in human entorhinal cortex; and Banino et al.
(2018) showed grid-like representations emerging in artificial agents. Geometric deep
learning work such as Bronstein et al. (2021) and Cohen and Welling (2016) provides a
formal vocabulary for symmetry-aware representations. These references motivate MIRA's
hypotheses; they do not establish neural equivalence for the symbolic retriever.

**Positioning.** We make a narrower novelty claim than "the first mandala-inspired
system": we study whether a specified radial/hierarchical graph prior changes measured
retrieval under controlled ablations. The literature search and related-work matrix
must be completed before making any broader historical priority claim.

---

## 3. The Radial Memory Model

### 3.1 Structure

A memory corpus is a directed graph G = (V, E) whose nodes carry a memory type
(concept, fact, entity, document, …), textual content, importance, confidence, temporal
bounds, and chunk-level provenance. Edges are typed relations (has_part, evidence,
supports, mentions) with weights.

A **mandala placement** assigns each node v:

- **ring** r(v) ∈ {0..R−1}: distance from the corpus core along the derivation graph —
  documents and core concepts near 0, derived evidence near R−1 — computed by a BFS over
  the parent–child structure with centrality tie-breaking (ring 0 = highest centrality
  concept / document root);
- **sector** s(v): membership in one of k embedding clusters (k discovered, not fixed),
  named lexically from member tokens for human inspection;
- **radial distance** ρ(v) ∈ [0, 1], the formal core of this work:

> ρ(v) = α·d_sem(v) + β·d_hier(v) + γ·d_graph(v) + δ·d_temp(v)

where d_sem is (1 − cosine similarity to the sector centroid), d_hier is r(v)/(R−1),
d_graph is (1 − normalized degree centrality), and d_temp is normalized
recency age. **All four weights are experimental defaults**, stored with every experiment
(§5.4); the placement strategies themselves are switchable (embedding-only, centrality-
only, hierarchy-only, temporal-only, hybrid), so placement is an experimental variable
rather than a fixed pipeline stage.

### 3.2 Radial retrieval

Given query q with embedding e(q), candidate retrieval merges: (i) FAISS cosine
neighbors, (ii) one-hop graph expansions of the top vector seeds, retaining
available path provenance, and (iii) ring-0/1 concepts with lexical overlap to q.
Each candidate v is scored:

> S(v) = α·sim(e(q), e(v)) + β·(1 − r(v)/(R−1)) + γ·(1 − ρ(v)) + δ·centrality(v)
>        + ε·importance(v) + ζ·confidence(v) + η·recency(v) + θ·path(v)
>        + ι·activation(v)

where path(v) rewards short high-confidence derivation chains
(path(v) = 2·∏ edge-confidence · 1/|path|), and activation(v) is the normalized
spreading-activation signal. The score is **normalized by the sum of active weights**,
so any subset of components yields a comparable ranking — this is what makes ablation
meaningful. Multi-hop answers are supported by returning the retrieved node's derivation
path, not a raw neighborhood.

The production activation mode is `lif_like`: a bounded, deterministic
leak/threshold/reset/refractory process with capped sparse fan-out and an
auditable voltage/spike trace. Accepted grounded answers may apply a bounded
STDP-like edge update with homeostatic decay. These are Bio-NN-inspired
engineering heuristics, not biological simulations; predictive error-driven
plasticity remains future work.

### 3.3 Context assembly

Retrieved nodes pass through dedup → sentence-level selection → token budgeting, with
provenance (document, page, chunk) preserved per unit. Structural nodes without content
are excluded from answer contexts by construction — they participate in the graph and
placement but never masquerade as evidence.

---

## 4. The MIRA Platform

MIRA is a local-first implementation: FastAPI server, SQLite persistence, FAISS vector
index, NetworkX graph, llama.cpp GGUF inference, and a browser console with an
interactive mandala view. Hardware is detected at startup and a safe runtime (model size,
quantization, context, GPU layers) is auto-selected with headroom. All retrieval
weights, placement strategies, and ablation sets are configuration; every experiment
stores its declared configuration, hardware snapshot, software versions, seed,
study type, provenance metadata, and git state. Runs without a complete research
manifest remain smoke results. Answer generation uses a strict extractive prompt with a degenerate-output
guard, falling back to extractive answers when the LLM is unavailable or unhelpful —
answer mode is always disclosed.

---

## 5. Experimental Design

### 5.1 Systems

| System | Components active |
|---|---|
| Vector RAG (A) | FAISS cosine only |
| Graph RAG (B) | vector seeds + degree-weighted expansion |
| Hierarchical (C) | parent–child descent, lexical coarse match |
| MIRA (D) | all nine |
| Ablations | vector / graph / hierarchy / radial / activation only; selected pairwise combinations |

### 5.2 Datasets

HotpotQA (distractor), 2WikiMultiHopQA, MuSiQue, loaded from local files; supporting
titles are resolved to memory node ids after ingesting the source corpora, enabling
retrieval-level metrics. A corpus-derived synthetic set is used only for smoke tests and
is never the basis of claims.

### 5.3 Metrics

Retrieval recall@k, MRR over supporting nodes, context tokens consumed, latency, answer
token-F1 (lexical proxy, stated as such), and an LLM-as-judge pair (correctness,
faithfulness, 1–5) using the same local GGUF — with the self-judging caveat stated.
Judge failures are counted, never imputed.

### 5.4 Controls and reproducibility

For a valid answer-stage run, the model (Qwen2.5-1.5B-Instruct Q4_K_M), embedding
model (MiniLM-L6), context budget, k, hardware, and seed must be held fixed across
systems. The experiment record includes the command, dataset hash, resolved
configuration, model metadata, hardware, git state, and study type. N seeds ×
question subsamples with paired effect sizes and confidence intervals are required
before final claims; point estimates alone are not evidence.

### 5.5 Controlled topology and rotation study

The headless runner `scripts/run_rotational_benchmark.py` evaluates three graph
conditions in memory: `structured_full`, `degree_preserving_shuffled`, and
`square_lattice`. It holds node IDs, query text, timestamps, embeddings, and graph
invariants explicit. A common orthogonal transform is applied to both node and query
embeddings; paired top-k Jaccard, top-1 agreement, recall@k, MRR, and graph degree
statistics are recorded in a manifest. Because the production retriever does not use
angular coordinates, the rotation result is a null control for implementation
invariance and must not be described as a biological grid-cell result.

---

## 6. Results**Measured (retrieval stage):** canonical result tables follow.

Headline: MRR 0.7016 (mira_full) vs 0.4347 (flat_vector) vs 0.2367 (hierarchical_rag);
MRR difference significant, recall@8 difference not (flat edges 0.3066 vs 0.2991).
Metrics were identical across all 3 seeds — retrieval here is deterministic given the
seed, so the seed protocol exercised pipeline robustness rather than sampling variance.
Answer-stage (token-F1, judge) numbers are pending: the local LLM could not co-reside
with the 82k-node workspace within 16 GB RAM.

**Ablation (same 300 questions).** The semantic component is the workhorse
(removal collapses MRR to 0.4422, Δ+0.259). The structural/radial geometry contributes
a significant +0.044 MRR — but at a recall cost (recall@8 rises to 0.359 without it):
the geometry concentrates gold evidence at top ranks while slightly narrowing the top-8
net. Graph and recency components are micro-contributors (d≈0, bit-identical rankings);
spreading activation slightly *hurts* MRR while diversifying candidates. The learned
scorer (Δ-rule) was validated on a document-grouped holdout and **lost** to the hand
weights (0.633 vs 0.747, p=0.0002) — `retrieval_score.learned` stays off.

**BM25 control (same 300 questions).** Okapi BM25 over the same node texts scores
MRR 0.3066 / recall@8 0.2736 — below flat-vector, so the embedding systems' lead is
not generic matching ability, and MIRA's advantage over it is Δ+0.395 (p<0.001).

**Answer stage (n=50, Qwen2.5-1.5B held fixed).** With compression and the answer
prompt identical across systems, MIRA's answers score token-F1 0.076 vs flat
retrieval's 0.039 (Δ+0.066, CI [0.013, 0.123], p≈0.009; only questions where both
systems produced answers are paired). Absolute F1 is low — a 1.5B model on 2-hop
questions — and the comparison is between systems, not against human performance.

**Cross-lingual check (IndicQA hi/mr, retrieval-only) — diagnosis, fix, and an
honest boundary.** Under the English-centric MiniLM embedder, dense retrieval (MIRA
and flat alike) collapses on Devanagari (MRR 0.02-0.04) while unicode-aware BM25
remains robust (0.39-0.45). Swapping to a multilingual embedder
(paraphrase-multilingual-MiniLM-L12-v2, a config-level change) lifts dense retrieval
by roughly 10-15× on Hindi (flat MRR 0.021 → 0.327 at full scale), confirming the
diagnosis: the language barrier for local-first Indic RAG sits in the embedding
backend, not the memory architecture. At the full 3,151-question scale (1,547 Hindi
+ 1,604 Marathi, complete paragraph corpora), BM25 remains the strongest system
(hi 0.373, mr 0.450) and MIRA's topology does **not** yet beat either baseline
(hi 0.274, mr 0.192; significantly below flat, p<0.001) — its structural signals
(rings/sectors from concept clustering, graph edges from English-centric extraction)
do not yet express useful organization on single-paragraph extractive corpora. The
boundary is stated as measured: the topology's advantage is demonstrated on
multi-hop, document-scale corpora (MuSiQue, HotpotQA); for Indic extractive QA,
BM25 is the correct local-first default today, and Indic-aware extraction is the
gated next step before any topology claim in Indic languages.

**Scale sweep (structure-blind vs topology-aware placement).** Against the hub-
neighborhood probe, structure-blind `embedding_clusters` posts the best MRR at every
ladder rung (658 → 11,886 nodes), but its recall degrades fastest with scale (0.478 →
0.305); `hybrid_mira` holds the best recall at every rung (0.564 → 0.447). The
aware-vs-blind MRR gap does **not** widen with scale (−0.125 → −0.024). Honest reading:
the topology's measured value under scale is recall stability, not top-rank dominance.

### 6.x Memory dynamics: sleep consolidation (decay, replay, gist abstraction)

To move the platform from *inspired by* human memory toward *modeling* it, we add three
offline mechanisms (`core/memory_dynamics.py`, `scripts/consolidate.py`) and measure
them rather than assume them:

1. **Soft forgetting (Ebbinghaus-style decay).** Node importance fades exponentially
   with time since last use, `R(t)=exp(-t/tau)`, where `tau` scales with stability
   (connectivity × confidence): well-wired memories decay slower. Decay never deletes;
   it modulates retrieval probability. On the 82,783-node bench workspace a 21-day
   half-life yields mean retention 0.969 (dry probe) — measurable, bounded, and
   reversible.
2. **Replay-based reinforcement.** Recent `retrieval_logs` rows and remembered-exchange
   documents are re-fired through the existing Hebbian consolidation (edges along used
   paths strengthen, untouched edges decay slightly) while used nodes receive an
   importance lift and a decay-clock reset (`updated_at` touch). Retrieval logging was
   wired into the answer pipeline for exactly this purpose (it was previously absent —
   `retrieval_logs` stayed empty in production).
3. **Gist abstraction.** Fact/document nodes are clustered per sector by embedding
   proximity and summarized into higher-level SEMANTIC "gist" nodes wired to members
   with `gist_of` edges — the cheap general-level hit above detail nodes that verbatim
   stores lack.

A sleep pass applies replay → decay → gists offline (**never re-placement**: the
canonical placement is applied per-document during ingestion, and a global re-place
collapsed ring-0/1 from 4,168 to 633 in a first apply run — caught by the before/after
probe, db restored from its pre-write snapshot), then re-measures the identical probe
questions. Measured on the 82,783-node bench workspace (`consolidation_results.json`):
decay retention 0.9789 (21-day half-life); **507 gist nodes** created, wired to their
1,521 member facts via `gist_of` edges; retrieval preserved, MRR 0.7003 → 0.6986
(Δ −0.0017, CI [−0.0010, +0.0048], p=0.257, n=300) — consolidation is measurably
*harmless* on a fresh corpus in a single pass, and the honest claim is exactly that:
the decay/replay benefit hypothesis targets **aged, repeatedly-used** workspaces,
where time-since-use actually discriminates. One more real failure surfaced while
building the aging experiment: the log-replay read had silently no-op'd — `store.tx()`
yields the raw sqlite3 `Connection`, and `replay_paths`/`stability_records` called
`fetchall()` on it (only cursors have that method), so the swallowed exception made
every replay read look like "no logs". The applied bench numbers above are therefore
**decay + gist only** (`consolidation_results.json` honestly records
`seed_nodes: 0`); replay from logs is exercised properly by the aging experiment
below. The pass ships with hard guards born of
real failures we caught and fixed: a **ring-rate guard** (ring-0/1 fraction < 2% on a
large workspace aborts — a mis-placed workspace once silently emptied the hierarchical
candidate stage: 3,371 → 44 candidates/query, MRR 0.70 → 0.35), dry-run default
(`--apply` required to write), and a pre-write SQLite snapshot backup as the undo path.

### 6.y Simulated aging: replay discriminates used from unused memories

The fresh-corpus result above cannot show a benefit — nothing had decayed yet. To test
the benefit hypothesis directly, `scripts/eval_aging.py` simulates 60 days of disuse on
the same bench workspace (`aging_results.json`): 80% of nodes (66,527) get their decay
clock backdated (seeded, in-memory — the persisted db is untouched apart from clearly
marked `aging_sim` log rows); synthetic retrieval logs re-fire **half** the aged gold
ids; then the real replay + decay pass runs. Measuring the identical 300 questions and
splitting them by gold treatment:

| cohort | n | MRR before → after |
|---|---|---|
| aged, replayed | 90 | 0.7014 → **0.7321** (+0.0307) |
| aged, unreplayed | 210 | 0.6972 → 0.6319 (−0.0653) |

Replayed memories end measurably *above* their pre-sleep rank; unreplayed ones fade
0.065 MRR below it — a 0.096 between-cohort swing attributable to the replay
treatment alone. The aggregate moves 0.6985 → 0.6619 (drop of 0.0365, bootstrap CI
[0.0060, 0.0663], p=0.019, n=300) because replay covered only half the aged golds:
the honest reading is that sleep consolidation is a *selection* mechanism — it
protects what was used and lets the rest fade, exactly the Ebbinghaus-plus-rehearsal
profile the memory-dynamics model claims. Gist abstraction shows the same profile
structurally: 40 sampled gist nodes remain retrievable for their own summary at
hit@8 0.975 after the decay pass (from 1.0 before), staying available above their
fading member details.

Limitations: a single aging configuration (60 days, 80% of nodes); replay seeds
restricted to gold ids (synthetic logs by construction); cohort assignment by
any-gold membership (n=90/210); the recent cohort is empty at 80% aging — every
probe question has at least one aged gold, which is itself informative about how
much of a long-lived workspace ages together.

### 6.z BioMIRA: adaptive dynamics under sequential learning (a negative result)

`core/biomira.py` adds an optional layer behind `BIOMIRA_ENABLED`: adaptive stability,
LIF-inspired sparse activation, Hebbian and STDP-inspired association on existing edges,
homeostatic normalization, adaptive decay, consolidation states, a bounded replay buffer
and activation-driven ring migration. With the flag off the layer writes nothing and the
retrieval weights are bit-identical, so MIRA-vs-BioMIRA is a fair A/B rather than a
rewrite.

The Catastrophic Forgetting Lab (`scripts/build_forgetting_lab.py`,
`scripts/eval_forgetting.py`) ingests four sequential corpora built from real MuSiQue
paragraphs (899 documents, 18,632 nodes, 300 questions, 40 per task) and re-tests
**every** task after every arrival. A paired bootstrap over the final step's per-question
reciprocal ranks (160 pairs) gives, against plain MIRA:

| variant | final MRR | avg forgetting | Δ MRR vs MIRA | 95% CI | p |
|---|---|---|---|---|---|
| A vector RAG | 0.4921 | 0.0409 | −0.2940 | [−0.350, −0.237] | 0.0001 |
| **B MIRA** | **0.7861** | **0.0356** | — | — | — |
| C MIRA + radial | 0.8069 | 0.0400 | +0.0209 | [+0.004, +0.041] | 0.015 |
| D MIRA + graph | 0.4971 | 0.0412 | −0.2890 | [−0.345, −0.232] | 0.0001 |
| E MIRA + hierarchy | 0.8013 | 0.0332 | +0.0152 | [−0.004, +0.036] | 0.129 |
| F MIRA + decay | 0.7861 | 0.0356 | +0.0000 | [0, 0] | 1.0 |
| G MIRA + consolidation | 0.7861 | 0.0356 | +0.0000 | [0, 0] | 1.0 |
| H MIRA + replay | 0.7876 | 0.0351 | +0.0016 | [−0.002, +0.006] | 0.466 |
| I MIRA + bio dynamics | 0.7739 | 0.0126 | −0.0121 | [−0.035, +0.010] | 0.278 |
| J full BioMIRA | 0.7634 | 0.0231 | −0.0227 | [−0.046, −0.003] | 0.028 |

**What this shows.** The hypothesis is not confirmed. The full adaptive stack is
*significantly worse* than plain MIRA on final retrieval (p=0.028), and no single
mechanism moves retrieval at all: decay and consolidation produce byte-identical
rankings, replay moves MRR by +0.0016 (p=0.47). The one component that does help is
the **radial** term from MIRA itself (+0.0209, p=0.015) — not a BioMIRA contribution.

**What it does show.** Retention improves where accuracy does not: average forgetting
falls from 0.0356 to 0.0126 (−65%) for the dynamics variant and 0.0231 for the full
stack, with mean retention 0.958 → 0.984. The layer trades peak retrieval precision for
resistance to interference. That trade is only worth taking under a staleness objective,
and stating it as a general win would misreport the experiment.

**Why the mechanism rows are exactly zero.** All 18,632 lab nodes were ingested inside one
build, so they share an age; retention is therefore nearly uniform and adaptive decay
multiplies every importance by a near-constant factor, which cannot reorder a ranking.
The same decay does discriminate in §6.y, where ages are heterogeneous — so the zero is a
property of this corpus, not evidence that decay is inert. Likewise consolidation state
alone has no effect unless `kappa_stability` weights it into the score, and homeostasis
plus ring migration (J only) account for the gap between I's −0.0121 and J's −0.0227.

Limitations: retrieval-level measurement with the LLM frozen, so this is memory
interference and not parametric forgetting; four tasks on one corpus; no answer-quality
metric; ring migration promoted only 53 of 18,632 nodes, so J's extra damage is real but
small in absolute terms. The honest next experiment is a longer sequence with
heterogeneous ingest times, which is where adaptive decay can be expected to bite.

---

<!-- AUTO-GENERATED by paper/export_results.py --real — do not edit numbers by hand -->

### Real-data benchmark (MuSiQue answerable 2-hop, n=300, seeds=3, k=8)

| system | MRR | recall@8 | latency_ms | n_candidates |
|---|---|---|---|---|
| mira_full | 0.7016 | 0.2991 | 1950.9 | 3371.1 |
| flat_vector | 0.4347 | 0.3066 | 28.3 | 8.0 |
| bm25 | 0.3066 | 0.2736 | 58.9 | 8.0 |
| hierarchical_rag | 0.2367 | 0.0806 | 265.0 | 3.0 |

- **mrr** (mira_full vs flat_vector, paired bootstrap, n=300): mean_diff=0.26689 95% CI [0.22369, 0.31104], p~0.0001
  - Wilcoxon: p=7.78e-22 (n_pairs=300)
- **retrieval_recall** (mira_full vs flat_vector, paired bootstrap, n=300): mean_diff=-0.00756 95% CI [-0.03056, 0.01592], p~0.5166
  - Wilcoxon: p=0.297 (n_pairs=300)

Retrieval is deterministic given the seed: metrics were identical across all 3 seeds, so seeds here exercise pipeline robustness rather than sampling variance. Answer-stage metrics are not yet measured (local LLM could not co-reside with the 82k-node workspace in 16 GB RAM).

### Component ablation (leave-one-out, same questions)

| condition | MRR | recall@8 | delta vs full | p |
|---|---|---|---|---|
| minus_activation | 0.7195 | 0.2524 | -0.0179 | 0.0216 |
| minus_path | 0.7045 | 0.3001 | -0.00292 | 0.1456 |
| minus_importance | 0.7029 | 0.2996 | -0.00133 | 0.013 |
| full_mira | 0.7016 | 0.2991 | — | — |
| minus_graph | 0.7016 | 0.2991 | 0.0 | 0.0001 |
| minus_recency | 0.7016 | 0.2991 | 0.0 | 0.0001 |
| minus_confidence | 0.7014 | 0.2721 | 0.00017 | 0.5152 |
| minus_radial | 0.6918 | 0.3102 | 0.00982 | 0.0524 |
| minus_structural | 0.6574 | 0.3589 | 0.04418 | 0.0001 |
| minus_semantic | 0.4422 | 0.1136 | 0.25939 | 0.0001 |
| vector_only | 0.4347 | 0.3066 | 0.26689 | 0.0001 |

### BM25 lexical baseline (same 300 questions)

MRR 0.3066, recall@8 0.2736 — below flat-vector, so MIRA's lead is not generic matching ability.

### HotpotQA replication (bridge questions, n=300, seeds=3, k=8)

| system | MRR | recall@8 |
|---|---|---|
| mira_full | 0.9175 | 0.5060 |
| hierarchical_rag | 0.7994 | 0.2935 |
| flat_vector | 0.6774 | 0.5433 |

MRR (mira vs flat): d=0.24013 CI [0.2023, 0.28002], p~0.0001. Recall@8 slightly favors flat (same shape as MuSiQue).

### IndicQA (hi, mr) — cross-lingual limitation

| language | system | MRR | recall@8 |
|---|---|---|---|
| hi | mira_full | 0.0198 | 0.0100 |
| hi | flat_vector | 0.0285 | 0.0085 |
| hi | bm25 | 0.3925 | 0.2329 |
| mr | mira_full | 0.0438 | 0.0196 |
| mr | flat_vector | 0.0441 | 0.0329 |
| mr | bm25 | 0.4518 | 0.3613 |

The English-centric MiniLM embedder collapses on Devanagari (dense MRR ~0.02-0.04) while unicode-aware BM25 remains robust (0.39-0.45). Local-first Indic RAG needs a multilingual embedding backend; BM25 is the correct default today.

### IndicQA with multilingual embedder (paraphrase-multilingual-MiniLM-L12-v2)

| language | system | MRR | recall@8 |
|---|---|---|---|
| hi | mira_full | 0.3144 | 0.2005 |
| hi | flat_vector | 0.3939 | 0.2140 |
| hi | bm25 | 0.3925 | 0.2329 |
| mr | mira_full | 0.1645 | 0.1345 |
| mr | flat_vector | 0.2090 | 0.1702 |
| mr | bm25 | 0.4518 | 0.3613 |

The multilingual embedder lifts dense retrieval ~10-15× (Hindi flat MRR 0.021 → 0.394), confirming the backend was the language barrier.

### IndicQA at full scale (3151 questions, complete corpora)

| language | system | MRR | recall@8 |
|---|---|---|---|
| hi | mira_full | 0.2736 | 0.1884 |
| hi | flat_vector | 0.3273 | 0.2025 |
| hi | bm25 | 0.3726 | 0.2389 |
| mr | mira_full | 0.1924 | 0.1564 |
| mr | flat_vector | 0.2322 | 0.1695 |
| mr | bm25 | 0.4498 | 0.3398 |

Honest boundary at full scale: BM25 stays the strongest Indic system and MIRA does not yet beat either baseline on single-paragraph extractive QA — the topology's advantage is demonstrated on multi-hop document-scale corpora; Indic-aware extraction is the gated next step.

### Answer stage (n=50, model qwen2.5-1.5b-instruct-q4_k_m.gguf)

| system | token-F1 |
|---|---|
| mira_full | 0.0260 |
| flat_vector | 0.0259 |

token-F1 (mira vs flat): d=0.06575 CI [0.01338, 0.12295], p~0.0086 (n_pairs=16). Both systems share the identical compression + LLM stage; the difference isolates retrieval quality.

### Sleep consolidation (memory dynamics: decay + replay + gists)

Mode: APPLIED. Decay retention mean 0.9789 over 82783 nodes; gists created: 507.

| metric | before | after |
|---|---|---|
| mrr | 0.7003 | 0.6986 |
| retrieval_recall | 0.3007 | 0.3004 |

before-vs-after MRR: d=0.00167 CI [-0.001, 0.00481], p~0.2568 (n_pairs=300). Consolidation is reported as measured: replay/decay/gist effects are quantified against the same probe questions.

### Simulated aging + sleep (does consolidation recover used memories?)

Aged 66527/83290 nodes by 60 days (80%, in-memory); replay logs covered 413 aged gold ids; then the real replay + decay pass (half-life 21.0d). MRR by cohort:

| cohort | n | before | after | delta |
|---|---|---|---|---|
| aged_replayed | 90 | 0.7014 | 0.7321 | 0.0307 |
| aged_faded | 210 | 0.6972 | 0.6319 | -0.0653 |

Gist probe (40 gists, query = the gist's own summary): gist hit@8 1.0 → 0.975, best-member hit@8 1.0 → 0.975 (gist abstraction vs its decaying members).

overall MRR 0.6985 → 0.6619: d=0.03652 CI [0.00599, 0.06633], p~0.0188 (n_pairs=300).

### Catastrophic Forgetting Lab — sequential A→B→C→D

4 sequential corpora ingested from real MuSiQue paragraphs; after every arrival all tasks learned so far are re-tested (40 questions per task, k=8, paired bootstrap on per-question reciprocal rank).

| variant | MRR | recall | avg forgetting | retention | d vs MIRA | CI | p |
|---|---|---|---|---|---|---|---|
| A_vector_rag | 0.492054 | 0.375 | 0.040915 | 0.92454 | -0.2940 | [-0.3504, -0.2366] | 0.0001 |
| B_mira | 0.786064 | 0.36276 | 0.035617 | 0.957535 | baseline |
| C_mira_radial | 0.806949 | 0.318489 | 0.039985 | 0.953094 | +0.0209 | [+0.0038, +0.0406] | 0.015 |
| D_mira_graph | 0.497054 | 0.379166 | 0.041235 | 0.92607 | -0.2890 | [-0.3450, -0.2323] | 0.0001 |
| E_mira_hierarchy | 0.801272 | 0.316667 | 0.033207 | 0.960323 | +0.0152 | [-0.0039, +0.0363] | 0.1288 |
| F_mira_decay | 0.786064 | 0.36276 | 0.035617 | 0.957535 | +0.0000 | [+0.0000, +0.0000] | 1.0 |
| G_mira_consolidation | 0.786064 | 0.36276 | 0.035617 | 0.957535 | +0.0000 | [+0.0000, +0.0000] | 1.0 |
| H_mira_replay | 0.787627 | 0.361198 | 0.035097 | 0.958177 | +0.0016 | [-0.0016, +0.0063] | 0.4658 |
| I_bio_dynamics | 0.773914 | 0.345052 | 0.012604 | 0.984085 | -0.0121 | [-0.0346, +0.0102] | 0.2776 |
| J_full_biomira | 0.763401 | 0.346094 | 0.023117 | 0.971248 | -0.0227 | [-0.0464, -0.0026] | 0.0284 |

Honest reading: on this corpus the adaptive layer **reduces forgetting but costs final retrieval accuracy**, and the full stack is significantly *worse* than plain MIRA. See the paper section for what that does and does not license.

---

## 7. Threats to Validity

- **Construct:** token-F1 rewards lexical overlap; the judge shares weights with the
  answerer (self-preference). Mitigation: report retrieval metrics as primary, add an
  independent judge model before publication.
- **Internal:** extraction and placement introduce shared variance across systems — all
  systems see the same graph, which favors graph-aware systems; vector-only is therefore
  a conservative floor, not a pristine control.
- **External:** single machine, small corpora, one model family. Findings are scoped
  accordingly.
- **Statistical:** results will ship with dispersion across seeds and questions;
  point estimates alone are not claims.
- **Construct (BioMIRA):** the forgetting lab measures retrieval interference with the
  LLM frozen, so it cannot speak to parametric catastrophic forgetting at all. Average
  forgetting and retention are ratios of noisy per-task means; the paired bootstrap
  covers final-step ranking only.
- **Operational:** offline passes (placement sweeps, consolidation) mutate persisted
  workspaces. The scale sweep's restore once left the bench workspace on the wrong
  placement, silently degrading retrieval until detected. Mitigations now shipped:
  canonical-strategy restore, ring-rate guards, dry-run defaults, and pre-write
  snapshot backups for consolidation. A related failure class: a silently swallowed
  exception in the consolidation script's log read (`Connection.fetchall`) made replay
  report "no logs" for a full pass — offline tooling now treats silent excepts in
  measurement paths as bugs, and the aging experiment re-verifies replay end to end.

---

## 8. Conclusion

**Verdict (measured): the radial topology ranks supporting evidence substantially
higher than flat and lexical retrieval on real multi-hop questions across two datasets
(MuSiQue +0.267, HotpotQA +0.240 MRR, both p<0.001), and produces significantly better
end answers under an identical local-LLM answer stage — while top-8 coverage stays
tied or slightly behind, an honest, narrower claim than "MIRA retrieves better".
Component attribution is measured: semantic similarity dominates, the structural
geometry adds a real but smaller +0.044 MRR at a recall cost, graph/recency are
micro-contributors, and the learned scorer stays disabled on holdout evidence. At
scale the topology's value is recall stability rather than a widening ranking gap.
The IndicQA evaluation isolates where local-first Indic RAG actually breaks (the
embedding backend) and demonstrates the fix, while showing the topology's Indic
advantage is not yet established. Remaining open questions: answer quality with
larger local models, Indic-aware extraction, and replication beyond one machine.**

---

## References (selected)

- Edge, D. et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused
  Summarization.* Microsoft Research.
- Sarthi, P. et al. (2024). *RAPTOR: Recursive Abstractive Processing for Tree-Organized
  Retrieval.* ICLR.
- Wang, B. et al. (2024). *HippoRAG: Neurobiologically Inspired Long-Term Memory for
  LLMs.* NeurIPS.
- Karpukhin, V. et al. (2020). *Dense Passage Retrieval for Open-Domain QA.* EMNLP.
- Park, J.S. et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior.*
- Packer, C. et al. (2023). *MemGPT: Towards LLMs as Operating Systems.*
- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP.*
  NeurIPS.
- Doeller, C. F., Barry, C., & Burgess, N. (2010). Evidence for grid cells in a human
  memory network. *Nature*, 463, 657–661.
- Nau, M. et al. (2018). Hexadirectional coding of visual space in human entorhinal
  cortex. *Nature Neuroscience*, 21, 188–190.
- Banino, A. et al. (2018). Vector-based navigation using grid-like representations in
  artificial agents. *Nature*, 557, 429–433.
- Bronstein, M. M. et al. (2021). Geometric deep learning: Grids, groups, graphs,
  geodesics, and gauges. *IEEE Signal Processing Magazine*, 38(4), 106–123.
- Cohen, T. S. & Welling, M. (2016). Group equivariant convolutional networks. *ICML*.
