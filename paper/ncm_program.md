# Abstract

MIRA is a local-first retrieval memory system for retrieval-augmented generation (RAG) whose
evidence store is organized radially (rings and sectors), hierarchically, and as a graph, scored
by a ten-term explicit function, and populated by biologically *inspired* consolidation dynamics
(decay, bounded replay, gist summaries). In controlled experiments its retrieval beats strong
baselines on multi-hop benchmarks (MRR 0.7016 vs 0.4347 for flat vectors on MuSiQue, p≈0.0001;
0.9175 vs 0.6774 on HotpotQA) while losing honestly elsewhere (IndicQA, where unicode-aware BM25
remains strongest). This document specifies the next research program: **MIRA-NCM (Nested
Constellation Memory)**. It states the primary research question, six falsifiable hypotheses
(H1–H6) with pre-registered success and rejection criteria, the architecture and formalization
of sparse overlapping constellation membership, nested memory nodes, and persistent versioned
storage, a mechanism-level analysis of *why* each structure should help AI memory and *how* each
effect would be measured, the ablation ladder and fairness controls that isolate every mechanism,
a prior-work comparison that bounds the novelty claims, and the future-scope roadmap. Nothing in
this document is a claimed result: sections that report numbers cite committed run artifacts;
everything else is labeled hypothesis or speculation. The program is designed so that any of its
hypotheses can be rejected by the experiments it prescribes.

# 1. Introduction and audit of the existing system

Long-term memory for AI agents fails in specific, measurable ways: relevant evidence is buried
under newer distractors (interference), multi-hop questions require binding facts that no single
embedding links, facts change over time and stale versions resurface as current, and the system
cannot say *why* it retrieved what it retrieved. MIRA attacks these as retrieval-organization
problems on a single consumer laptop (RTX 3050 4 GB, 16 GB RAM) with a frozen, replaceable local
LLM — no cloud calls, no fine-tuning, so every observed effect is attributable to memory
architecture rather than to model capacity.

**What exists today** (verified against the repository and committed artifacts):

- **Radial organization.** Every memory node carries interpretable coordinates: ring (recency /
  importance band) and sector (topic region), giving a radial-distance term in retrieval scoring.
- **Ten-term explicit scoring.** `score = α·semantic + β·structural + γ·radial + δ·graph +
  ε·importance + ζ·confidence + η·recency + θ·path + ι·activation + κ·stability`. No hidden
  terms; every component is individually ablatable and logged per retrieval.
- **Graph + hierarchy + vector** as parallel retrieval paths over the same evidence nodes.
- **Biologically inspired dynamics (BioMIRA).** Adaptive decay (half-life configurable), bounded
  replay during "sleep" consolidation, gist summaries (507 gists created over an 82,783-node
  workspace), Hebbian-inspired association and STDP-inspired temporal updates, activation and
  stability traces. These are engineering analogies, not claims of biological equivalence.
- **Explainability.** Each retrieval returns per-component scores and the evidence path; the
  answer stage compresses only retrieved evidence and cites it, with an explicit no-evidence
  state that is never silently converted into a grounded answer.
- **Experiment framework.** Committed manifests and result JSONs for: the real-data benchmark
  (n=300 MuSiQue questions, 3 seeds), component ablations, HotpotQA replication, IndicQA
  (Hindi/Marathi), the answer-stage comparison (n=50), sleep consolidation, simulated aging,
  and a sequential catastrophic-forgetting lab over six corpora (A→F, 30 questions per task,
  paired bootstrap), with per-question metrics and confidence intervals.

**Honest current weaknesses** measured on the committed artifacts: retrieval recall@8 ties flat
vectors (0.2991 vs 0.3066, p=0.52) — MIRA's MRR lead is ranking, not coverage; `graph` and
`recency` components are currently inert on the MuSiQue corpus (ablation deltas exactly 0.0);
dense retrieval collapses on Devanagari with an English-centric embedder (MRR 0.02–0.04) and
still trails BM25 at full IndicQA scale even with a multilingual embedder; the 2-hop answer-stage
F1 gain over flat vectors, while statistically consistent, is small in absolute terms (n=50).
These weaknesses are the starting point for the NCM program, not embarrassments to hide: each
one names a mechanism the new architecture must justify or drop.

# 2. Research questions

**Primary.** *Does structured, sparse, overlapping constellation-based memory organization
improve long-term information retrieval, multi-hop retrieval, temporal reasoning, and continual
memory retention compared with conventional vector, graph, hierarchical, and non-overlapping
radial memory organizations?*

**Secondary.** *Does nested memory organization represent relationships at multiple scales
without uncontrolled memory duplication?* **Tertiary.** *Can persistent versioned memory with
adaptive dynamics improve retention while keeping storage and retrieval cost bounded?*
**Optional fourth.** *Can quantum-inspired probabilistic representations improve ambiguity and
overlapping-membership handling on classical hardware?*

None of these is assumed to have a "yes" answer. Section 4 gives each a rejection condition.

# 3. Status taxonomy — what is what

Every claim in the MIRA program is tagged with exactly one status:

| Tag | Meaning | Examples |
|---|---|---|
| [CS] | Established computer science | kNN search, BM25, exponential decay, checksums, versioned records |
| [ENG] | MIRA's original engineering implementation | radial ring/sector layout; ten-term scoring; compress-then-cite |
| [HYP] | Falsifiable hypothesis, not yet tested | H1–H6 (§4); constellation overlap effects |
| [DEMO] | Demonstrated by committed artifacts | the tables in §5; forgetting-lab results |
| [SPEC] | Speculative future direction | quantum-inspired layer; swarm memory (§10) |

"Quantum-inspired" always means a classical probabilistic representation; "persistent" always
means persistent *and versioned*, never "forever memory"; "biologically inspired" never implies
biological fidelity; radial geometry is an organizational coordinate system, not a physical one.

# 4. Hypotheses, success criteria, and rejection conditions

Each hypothesis names its metric, comparison, and threshold. Thresholds are fixed before the
final runs (pre-registration); exploratory deviations will be labeled as such.

**H1 [HYP] — Radial organization improves structured retrieval.** *Supported* if removing the
radial term (leave-one-out) costs ≥ 0.02 MRR on multi-hop corpora at p < 0.05. *Current
evidence [DEMO]*: minus-radial costs 0.0098 MRR (p=0.052) on MuSiQue — borderline, not yet
supported at the pre-registered threshold; in the forgetting lab the radial variant is the best
joint variant (MRR 0.802, forgetting 0.049). *Rejected* if larger corpora or the NCM benchmark
show no significant cost when radial is removed.

**H2 [HYP] — Sparse overlapping constellations improve multi-hop retrieval.** *Supported* if
MIRA-NCM beats the strongest non-overlapping structural variant by ≥ 0.03 MRR on 2-hop and ≥ 0.05
on 3–4-hop synthetic multi-hop sets, without recall@8 loss and within +15 % latency. *Rejected*
if overlap adds cost without a hop-depth-dependent gain. *Mechanistic prior [DEMO]*: the
structural/path term is the second most valuable component (removing it costs 0.0442 MRR,
p=0.0001), suggesting binding structure — not raw similarity — is what multi-hop retrieval uses.

**H3 [HYP] — Nested memory gives multi-scale access within bounded storage.** *Supported* if
hierarchy-sensitive questions gain ≥ 0.03 MRR while total node growth stays < 10 % (summaries
replace duplication; children are references, not copies). *Rejected* if gains appear only with
unbounded depth or duplicated storage. *Prior [DEMO]*: the hierarchy variant already retains
0.946 (best of all forgetting-lab variants); nesting generalizes it.

**H4 [HYP] — Persistent versioning improves temporal/conflict correctness.** *Supported* if the
versioned variant reduces wrong-current-answer errors by ≥ 50 % on the temporal and contradiction
benchmarks (§7) relative to last-write-wins storage. *Rejected* if version lookup alone (without
supersession-aware scoring) performs equally, or if history storage dominates the budget.

**H5 [HYP] — Adaptive dynamics improve continual retention without retrieval cost.** *Supported*
if average forgetting drops by ≥ 30 % with MRR indistinguishable from the static system (paired
CI includes 0) across intervals {7, 30, 90 days}. *Current evidence [DEMO], partial*: the
biologically inspired stack cuts average forgetting 0.1074 → 0.0527 (retention 0.873 → 0.938)
with MRR +0.0115, CI [−0.014, +0.037], p=0.378 — the forgetting reduction replicates across
intervals (0.039/0.053/0.038 at 7/30/90 days) but the MRR CI does not exclude a small loss, and
two mechanisms (homeostatic normalization, ring migration) *hurt* (retention 0.914) and are
therefore disabled in the tuned stack. H5 is half-supported: dynamics buy retention; their
retrieval neutrality is not yet established at n=300 questions.

**H6 [SPEC→HYP] — A quantum-inspired representation improves ambiguity handling.** *Supported*
only if the optional probabilistic backend beats the classical scorer on a pre-defined ambiguous-
membership benchmark at the same compute budget. Implemented as an isolated, switchable module;
if it shows no improvement it will be reported as such and removed from the default stack. No
claim of quantum advantage, entanglement, or hardware of any kind is made or will be made.

# 5. Measured foundation (all numbers from committed artifacts)

**Multi-hop retrieval (MuSiQue answerable 2-hop, n=300, k=8):** MIRA 0.7016 MRR vs flat vector
0.4347 (Δ=0.267, 95 % CI [0.224, 0.311], p≈0.0001), BM25 0.3066, hierarchical RAG 0.2367.
Recall@8 statistically ties flat vectors. Replicated on HotpotQA bridge questions (MRR 0.9175 vs
0.6774, p≈0.0001). Answer stage (n=50, 1.5B local model, identical compression+LLM for both
arms): token-F1 0.0260 vs 0.0259, p=0.0086 — consistent but small.

**Component attribution (leave-one-out):** semantic 0.2594, structural 0.0442, activation 0.0179
(each p≤0.022); radial 0.0098 (p=0.052); graph and recency exactly inert on this corpus
(delta 0.0, p=0.0001 by bootstrap degeneracy) — an honest negative that motivates NCM's overlap
mechanism to give structural terms work to do beyond the document tree.

**Cross-lingual boundary:** with an English-centric MiniLM embedder, Hindi/Marathi dense MRR is
0.02–0.04 while unicode-aware BM25 holds 0.39–0.45; a multilingual embedder lifts Hindi flat
MRR ~14× (0.028 → 0.394), yet at full scale (3,151 questions) BM25 remains strongest. MIRA
does not currently beat baselines on single-paragraph extractive QA in Indic languages.

**Continual-memory lab (six sequential corpora, 30 questions/task, all prior tasks re-tested
after each arrival; 30 simulated days of disuse between arrivals):**

| variant | MRR | avg forgetting | retention | Δ MRR vs B (95 % CI) | p |
|---|---|---|---|---|---|
| A vector RAG | 0.4869 | 0.0432 | 0.9224 | −0.2670 [−0.324, −0.211] | 0.0001 |
| B MIRA | 0.7539 | 0.1074 | 0.8731 | baseline | — |
| C MIRA radial | 0.8020 | 0.0490 | 0.9420 | +0.0481 [+0.024, +0.074] | 0.0001 |
| E MIRA hierarchy | 0.7887 | 0.0443 | 0.9461 | +0.0348 [+0.007, +0.063] | 0.0118 |
| I B. dynamics | 0.7654 | 0.0527 | 0.9384 | +0.0115 [−0.014, +0.037] | 0.378 |
| J tuned stack | 0.7655 | 0.0533 | 0.9383 | +0.0117 [−0.014, +0.037] | 0.374 |
| K + homeostasis | 0.7444 | 0.0769 | 0.9138 | −0.0095 [−0.043, +0.024] | 0.577 |

Reading this table honestly: low forgetting alone is worthless — vector RAG has the second-lowest
forgetting *and* the worst MRR, because weaker memories have less to interfere with. The
meaningful result is the joint criterion: the radial variant improves both MRR (+0.048) *and*
retention (+0.069) simultaneously. Decay alone and consolidation alone are byte-identical to
baseline (inert); replay alone is noise (p=0.747); only the *interaction* of decay and bounded
replay moves retention. The store grows monotonically 4,487 → 23,516 nodes across arrivals —
nothing is deleted, so "interference" here is purely a ranking phenomenon, not data loss.

**Sleep consolidation under aging:** after 60 simulated days over 66,527 nodes, replayed
memories *recover* (MRR +0.0307) while faded ones decline (−0.0653); decay retention is 0.9789
per pass with no measurable retrieval cost (Δ=0.0017, p=0.26). This is the complementary-
learning-systems pattern — fast trace, selective replay, gist abstraction [8][9] — implemented
as retrieval dynamics, and it is the strongest existing evidence that dynamics can *target*
what consolidates rather than merely rotting everything.

# 6. MIRA-NCM: architecture and formalization

```
                 MIRA (existing, preserved as ablation baseline)
      radial ─┬─ graph ─┬─ hierarchy ─┬─ vector ─┬─ ten-term scoring
              └─────────┴─────────────┴──────────┘
                                │
                     CONSTELLATIONS (new)
              sparse overlapping soft membership, K per node
                                │
                   NESTED MEMORY NODES (new)
              bounded-depth parents, children = references
                                │
              PERSISTENT VERSIONED STORAGE (new)
              immutable history, supersession, provenance, checksums
                                │
              ADAPTIVE DYNAMICS (existing, tunable)
              decay · replay · consolidation · stability
                                │
         multi-path retrieval + per-component explanation
                                │
        [SPEC] optional quantum-inspired probabilistic backend
```

**Constellations.** A constellation is a soft cluster induced from the corpus — never a
hand-coded category: centroids are seeded from embedding space (k-means over node vectors),
entities, and graph communities, then grow by assignment. A memory node *m* joins its top-K
constellations with weights derived from similarity to the centroid and normalized in L1:
`w_c(m) = sim(e_m, μ_c) / Σ_{c' ∈ topK} sim(e_m, μ_{c'})`, with `CONSTELLATION_TOP_K` ∈ {1,2,3,5}
as an ablation axis (K=1 reproduces non-overlapping soft clustering; the hypothesis is precisely
that K>1 is what helps). Overlap between memories uses the shared-membership inner product
`overlap(a,b) = Σ_c w_c(a)·w_c(b)` — a bounded, sparse kernel: candidate pairs are generated
only through a constellation-inverted index, so construction is O(N·K²) expected, not O(N²),
and `MAX_OVERLAP_EDGES` per constellation plus an `OVERLAP_THRESHOLD` floor prevent explosion.
Radial distance `d_r = |ring_a − ring_b| + λ·ang(θ_a, θ_b)` and constellation distance
`d_c = 1 − overlap(a,b)` are independent, unit-testable functions; the geometry is an
organizational coordinate system, and no physical interpretation is claimed for it.

**Nesting.** A node may declare a bounded-depth parent (`MAX_DEPTH`, default 4; children are
node-id references with provenance, never copies). Retrieval may traverse parent/summary links
as one more scoring path (`w_hierarchy`); nesting never replaces vector or graph access.

**Persistence.** Every update appends a version: immutable history, `supersedes` /
`superseded_by` links, content hash (`SHA-256` over content + metadata) for integrity and
duplicate detection, and full provenance. Current-ness is a *scoring property*: superseded
versions remain retrievable as history (a temporal query about March must be able to find the
March belief) but are down-weighted for "what is true now" queries. Consolidation merges store
the summary *and* its source links, so reconstruction is always possible.

**Retrieval scoring** extends the existing ten-term function with two explicit terms:
`+ μ·constellation_overlap(q,m) + ν·version_currentness(m)`, each individually ablatable,
logged per retrieval, and returned in the explanation payload ("retrieved because:
semantic 0.82, overlap 0.76, path 0.91, …"). Everything else about MIRA — compress-then-cite,
the no-evidence state, the memory gate — is untouched.

# 7. Why this should help AI memory, and how each effect is measured

The program rests on one observed regularity: **retrieval interference is a structural
phenomenon, and structure is what MIRA already manipulates.** Mechanism by mechanism:

**Multi-hop binding (tests H2).** A 4-hop question connects person → project → hardware → event.
Flat similarity only helps the hop with lexical overlap; embeddings miss the rest. Overlapping
constellations create a sparse evidence graph in which two facts that share *any* context
(the person, the project, the time) are one traversal step apart. The measured prior — removing
structural/path terms costs 0.0442 MRR while semantic remains dominant — says binding structure
carries real weight; the hypothesis is that *overlap* is the right shape of that structure
because it does not force each memory into one bucket. *Measured by*: per-hop retrieval accuracy
and path accuracy on synthetic 1–4-hop chains built with the same generator discipline as the
existing benchmarks, plus the real MuSiQue/HotpotQA suites for external validity.

**Continual retention (tests H5, extended by constellations).** In the forgetting lab, new
corpora degrade old rankings purely by adding competitors. Constellations partition the
interference set: a distractor from an unrelated constellation competes less with a target
(its overlap term is ~0), so the effective distractor pool per query shrinks as the corpus
diversifies. Meanwhile nested gists give replay a rehearsal target — and the aging experiment
already shows targeted replay *recovers* aged memories (+0.0307) rather than merely slowing
decay. This is why the forgetting claim is scoped: MIRA measures *retrieval* interference with
the LLM frozen; it does not and cannot speak to parametric forgetting in neural weights, and no
experiment here claims otherwise. *Measured by*: the existing sequential A→F protocol extended
to 8–10 corpora, reporting forgetting, retention, MRR jointly — never forgetting alone, which
weak systems trivially win.

**Temporal correctness (tests H4).** Contradiction handling fails today the way it fails in all
unversioned stores: the newest fact wins by recency weight regardless of question tense. Version
chains turn "what did the user use in March?" into a lookup and "which preference is current?"
into a supersession traversal, both deterministic and auditable. *Measured by*: a generated
temporal benchmark (belief-revision sequences, e.g. Python → Rust → Python) scored on
current/historical/superseded/uncertain classification, with wrong-current-answer rate as the
headline error.

**Interpretability as a capability, not decoration.** Because every term is explicit and logged,
a wrong retrieval can be attributed to a component, and a component that only hurts can be
removed — the ablation table is the mechanism for this, and it already exposed two inert terms
(graph, recency on this corpus) and two harmful dynamics (homeostasis, ring migration). The NCM
program inherits that discipline: any mechanism that cannot pay for itself gets deleted.

**Why a local, frozen-LLM setting is the right testbed.** With the model fixed, every delta
between systems is attributable to memory organization — the variable the program is trying to
understand — and the whole stack runs on 16 GB RAM-class hardware, which is also where agent
memory actually has to live. The cost is external validity to larger models; that is a
limitation, stated here, and revisit-able later, not a hidden one.

# 8. Experimental design

**Benchmark suite (MIRA-MemoryBench).** Single-hop; multi-hop (1–4 hop, synthetic + MuSiQue +
HotpotQA); temporal belief revision; contradiction; continual sequence (8–10 corpora);
cross-lingual (en/hi/mr, with the multilingual embedder as the reported configuration and the
English-centric one as a documented failure mode); noise and distractor stress; memory update
and version-history drills; explanation audit (do logged components predict human-judged
relevance better than chance?). Each benchmark run emits `manifest.json` (experiment id, git
commit, model, embedder, dataset version, seed, hardware, K, context budget, memory count),
`results.json`/`results.csv`, `metrics.json`, plots, and logs. Datasets are synthetic-generated
with committed generators (source, license, seed, statistics) plus public sets where licensing
permits; leakage tests check for duplicate documents, repeated questions, and answer exposure.

**Ablation ladder (all selectable by configuration, existing MIRA preserved).** B0 keyword ·
B1 vector RAG · B2 graph RAG · B3 hierarchical RAG · M1 MIRA minus radial · M2 MIRA ·
M3 MIRA + graph · M4 + constellations · M5 + nesting · M6 + versioning · M7 + adaptive
dynamics · M8 full MIRA-NCM · M9 = M8 + quantum-inspired backend. MODE configuration keeps
`classic_mira`, `mira_ncm`, `bio_mira`, `quantum_inspired` all loadable in the same process.

**Fairness controls.** Same questions, embedder, LLM, context budget, retrieval K, temperature,
seeds, preprocessing, and evaluation code for every arm. If MIRA-NCM needs more storage or
latency, that is reported in the same table as the accuracy it buys; trade-offs are results.

**Statistics.** ≥ 3 seeds where variance exists; paired bootstrap for per-question metrics with
95 % CIs; Wilcoxon signed-rank as the non-parametric cross-check (both already implemented);
effect sizes reported beside every p-value; "statistically significant" is never conflated with
"practically meaningful" — the answer-stage F1 result (p=0.0086, Δ=0.0001 absolute) is the
standing example of the difference. Hypotheses H1–H6 are pre-registered in
`docs/EXPERIMENT_PROTOCOL.md` before the final runs; thresholds will not move after results.

**Failure analysis.** Every benchmark exports `failure_examples.json` with a taxonomy:
semantic confusion, wrong constellation, wrong graph path, wrong temporal ordering,
over/under-retrieval, stale-memory errors, excessive overlap, storage blowup, multilingual
failure. Failures are inspected before any headline is written.

**Resource envelope.** Target: retrieval p95 < 2 s at 10⁵ nodes on 16 GB RAM / 4 GB VRAM,
storage overhead of NCM structures < 25 % of node payload, measured via the existing resource
instrumentation at 1K/10K/50K/100K nodes. Scaling claims beyond what is measured will not be
made.

# 9. Prior work and bounded novelty claims

MIRA-NCM combines known ideas; the contribution is the *composition and its measurement*, not
invention of the ingredients. Sparse distributed memory with decaying counters and positional
structure is Kanerva's [14]; associative spreading activation is classical [11]; hierarchical
control and episodic-storage separation in agent memory appears in Generative Agents [15];
graph-structured retrieval for multi-hop QA is HippoRAG's premise (a PageRank-style KG index)
[16] and GraphRAG's (community summaries) [17]; continual-learning taxonomies and replay
baselines are standard [3–7]; complementary learning systems motivate the decay/replay/gist
triangle [8][9][10]; BM25 remains a baseline no structured system may skip [13][18] — MIRA's own
IndicQA results are the proof of that rule. What MIRA-NCM adds, contingent on H1–H3: an
*overlap-kernel* definition of memory association that is sparse by construction and ablatable
per K; a forgetting lab that scores forgetting jointly with ranking quality (most continual
benchmarks report only accuracy); and a retrieval stack whose every component is individually
measured and removable. Where a competing system already demonstrates an effect, the paper will
cite it and claim only the difference.

# 10. Risks and kill criteria

| Risk | Mitigation / kill criterion |
|---|---|
| Overlap explosion (K large, corpus dense) | inverted-index construction; MAX_OVERLAP_EDGES; if overhead > 25 % without MRR gain, mechanism is cut |
| Embedding dominance swamps new terms | ablation gates: each new term must move its target metric with others fixed, else weight → 0 |
| Constellations re-create single-assignment (K=1 behavior) | K sweep is a mandatory axis; K∈{2,3} must differ from K=1 to keep H2 alive |
| Version history grows unbounded | history capped per node with archived cold storage; H4 rejection if history > 30 % of store |
| Dynamics help retention, hurt MRR (as K showed) | joint criterion only: a variant that fails the MRR CI test is disabled regardless of retention |
| Gist abstraction loses members | the measured 0.975 gist hit-rate is the floor; regressions trigger regeneration |
| Any baseline wins a suite | reported as-is; per §61 principle the goal is a true memory principle, not a win column |

# 11. Future scope

**Phase 1 — Constellation core (weeks 1–3).** Centroid induction, top-K membership, overlap
index, `w_constellation` term, unit tests, K-sweep ablation on the existing MuSiQue harness.
Ship criterion: H2 pilot on synthetic 3-hop chains.

**Phase 2 — Nesting + versioning (weeks 3–6).** Bounded-depth parents, version chains,
supersession scoring, temporal + contradiction benchmarks; H3/H4 pilots.

**Phase 3 — Continual suite scale-up (weeks 6–8).** Extend the forgetting lab to 8–10 corpora
with NCM variants in the ladder; joint-criterion analysis; H5 final test.

**Phase 4 — Explanation audit + resource sweeps (weeks 8–10).** Component-predicts-relevance
study, 10⁵-node scaling run, complexity documentation from the actual implementations.

**Phase 5 — Optional backends (quarter 2).** Quantum-inspired probabilistic membership as a
switchable scorer (H6, explicitly allowed to fail); pluggable storage (SQLite/FAISS backends
already implicit in the local-first design); SDK (`MemoryEngine.add/retrieve/history/explain`)
over the existing server API; multi-tenant namespaces only if an actual deployment demands them.

**Longer-range [SPEC].** Agentic self-curriculum (the agent schedules its own replays from
prediction error); federated memory across devices with provenance-preserving merge; learned
constellation induction (currently centroid-based by design — learned clustering is a
confound until the fixed version is characterized); learned term weights replacing the hand
set (only after the fixed-weight behavior is fully mapped, otherwise attribution dies).

**Product trajectory (deliberately modest).** The engine first: inspectable long-term memory
infrastructure for local AI agents — a defensible niche precisely because every retrieval is
auditable and every claim reproducible. Local → SDK → optional managed deployment, in that
order, and only after the benchmarks justify the engineering.

# Reproducibility statement

Every measured number in this document is generated from committed run artifacts and reproduced
by `paper/export_results.py --real`; the forgetting-lab table traces to
`data_lab/forgetting_results.json` (variants A–K, 6 tasks, 30 questions each, complete); the
retrieval tables trace to `data_bench/*.json`; consolidation and aging to
`data_bench/consolidation_results.json` and `data_bench/aging_results.json`. Nothing here is
hand-entered. Hypotheses H1–H6 are forward-looking and marked as such; the NCM components they
describe are **not yet evaluated** and no result is claimed for them. Full protocol (datasets,
thresholds, seeds, tests) to be committed as `docs/EXPERIMENT_PROTOCOL.md` before the first
final-variant run, per pre-registration.

# References (selected)

1. Trivedi, H. et al. (2022). MuSiQue: Multihop Questions via Single-hop Question Composition. *TACL*, 10, 657–680.
2. Yang, Z. et al. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. *EMNLP*.
3. McCloskey, M. & Cohen, N. (1989). Neural networks and catastrophic forgetting. *Neural Computation*, 11(7), 1649–1671.
4. Parisi, T. I. et al. (2019). Continual lifelong learning with neural networks: A review. *Neural Networks*, 113, 54–71.
5. Kirkpatrick, J. et al. (2019). Overcoming catastrophic forgetting in neural networks. *PNAS*, 116(13), 6521–6529.
6. Lopez-Paz, D. & Ranzato, M. (2017). Gradient Episodic Memory for Continual Learning. *NeurIPS*.
7. De Lange, M. et al. (2022). A continual learning survey. *IEEE TPAMI*, 44(7), 3366–3385.
8. McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Complementary learning systems. *Psychological Review*, 102(3), 419–457.
9. Wilson, M. A. & McNaughton, B. L. (1994). Reactivation of hippocampal ensemble memories during sleep. *Science*, 265, 676–679.
10. Marr, D. (1971). Simple memory: A theory for archicortex. *Phil. Trans. R. Soc. B*, 262, 23–81.
11. Hebb, D. O. (1949). *The Organization of Behavior.* Wiley.
12. Bi, G.-Q. & Poo, M.-M. (1998). Activity-driven Hebbian modification of synaptic strengths. *J. Neuroscience*, 18, 10464–10472.
13. Robertson, S. & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and beyond. *Foundations and Trends in IR*, 3(4).
14. Kanerva, P. (1988). *Sparse Distributed Memory.* MIT Press.
15. Park, J. S. et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior. *UIST*.
16. Gutiérrez, B. J. et al. (2024). HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. *NeurIPS*.
17. Edge, D. et al. (2024). From Local to Global: A Graph RAG Approach to Query-Focused Summarization. arXiv:2404.16130.
18. Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.
