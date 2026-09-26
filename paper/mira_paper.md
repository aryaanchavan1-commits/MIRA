# Radial Memory Topologies for Retrieval-Augmented Generation: A Controlled Study of Mandala-Inspired Organization

**Draft v0.1 — results pending experiments. Every number in the results section is
generated from real runs by `paper/export_results.py`; this draft contains none.**

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
**[Results TBF — populated only from measured runs.]** We release the platform, including
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

## 6. Results

**[TBF — this section is populated exclusively by `python paper/export_results.py`
from a saved `study_type: research` manifest. Smoke runs are rejected unless
`--allow-smoke` is supplied for inspection. No numbers are written by hand.]**

Planned tables:
1. Main table: all systems × {recall, MRR, token-F1, judge, context tokens, latency}
2. Ablation deltas vs full MIRA, per dataset
3. Retrieval overlap (Jaccard) between systems, showing *where* components diverge
4. Latency/token cost of each component

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

---

## 8. Conclusion

**[TBF after valid runs.]** The platform is runnable, but the honest state of
this research is *hypothesis stated, instrument built, results pending*. The
current smoke harness does not establish answer quality, multi-hop reasoning, or
causal component benefits.
A negative result — radial organization adding nothing beyond graph+hierarchy — is a
publishable outcome of this instrument, and the harness is built to make that outcome
easy to demonstrate.

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
