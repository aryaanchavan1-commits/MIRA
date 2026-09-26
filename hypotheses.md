# MIRA Hypotheses

> **Current evidence status:** H1–H3 are preregistered hypotheses, not findings.
> The default CLI and console runs are smoke tests unless a declared research
> protocol, exact evidence labels, and a complete provenance manifest are present.

## H1 (primary)
> A radial hierarchical memory topology combining semantic similarity, hierarchical organization, radial proximity, graph relationships, importance, confidence, provenance and temporal information improves evidence retrieval and multi-hop reasoning while reducing irrelevant context.

**Operationalization:** a valid run must compare MIRA (full) against declared
Baselines A–C on identical datasets, questions, models, context budget, and
hardware, with exact supporting-evidence labels and a predeclared primary endpoint.
The current retrieval-only smoke harness does not satisfy this protocol.

**Falsification:** define a minimum practically important effect and report a
paired effect size with a 95% interval before running. If the interval excludes
that effect in the predeclared direction, H1 is not supported; a non-significant
point estimate alone is not a falsification.

## H2 (radial signal)
> Radial proximity (the α/β/γ/δ weighted radial distance, §12) is a useful retrieval feature beyond what semantic similarity + graph features already provide.

**Operationalization:** use a frozen candidate pool and compare full MIRA with
full-minus-radial and full-minus-hierarchy arms. The current named scoring
ablations are useful diagnostics, but they are not causal component isolations
because candidate generation still uses shared channels.

## H3 (compression)
> Path-based retrieval + compression (§22-23) reduces context tokens without accuracy loss.

**Operationalization:** run every system through one answer/context stage at
multiple explicit token budgets. A valid comparison uses exact evidence labels,
records supplied context and completion tokens, and evaluates answer quality
independently. The current retrieval-only tables cannot test H3.

## H4 (sector semantics)
> Embedding-discovered sectors (§10) align with human-meaningful topic partitions.

**Operationalization:** qualitative + cluster-coherence metric; exploratory, not required for H1.

## H5 (controlled topology)
> Under a fixed node set, query set, embedding vectors, and reported degree/edge
> statistics, the structured radial/hierarchical graph retrieves supporting nodes
> differently from a deterministic degree-preserving rewiring and a descriptive
> regular square lattice. The lattice is not degree- or edge-matched, so its
> differences cannot independently isolate radial organization.

**Operationalization:** run `scripts/run_rotational_benchmark.py` with fixed seed,
timestamps, in-memory stores, and the three conditions in `evaluation/rotation.py`.
Report paired top-k Jaccard, top-1 agreement, recall@k, MRR, degree histograms, and
rewiring invariants. Do not attribute a difference to radial organization if edge
count, degree distribution, or placement recomputation differs.

**Falsification:** if the structured condition has no measurable advantage, or its
advantage disappears when the confound controls are matched, H5 is not supported.

## H6 (coordinate-rotation null control)
> Applying one common orthogonal transformation to every node embedding and query
> vector preserves the current symbolic retriever's ranking.

**Operationalization:** compare each reference query with its rotated counterpart while
keeping node IDs, text, graph topology, timestamps, and query text fixed. The expected
result is paired top-k Jaccard approximately 1.0 because angular metadata is currently
visualization-only. This tests implementation invariance; it is not a biological
rotational-cell claim.

## H7 (Indic-pattern extension)
> A provenance-controlled, culturally attributed Indic pattern benchmark can test
> whether the same topology controls generalize beyond synthetic geometry.

**Operationalization:** obtain licensed/permissioned kolam, yantra, or rangoli images
with source and tradition metadata; define a held-out split and a human-validated
relevance protocol. Synthetic patterns may be used only as smoke-test controls and
must be labeled synthetic.

---

## Honesty clauses (spec §49)
- All weights (§12/§21) are **experimental defaults**, not optimal values.
- A **negative result is a valid result** and will be documented as such.
- The system does **not** claim historical mandalas were neural networks or that ancient traditions contained modern AI. MIRA is an engineering artifact *inspired by* the organizational and visual principles of mandalas/yantras; its usefulness is an empirical question.
- `lif_like` and `stdp` are deterministic Bio-NN-inspired heuristics, not biological simulations. Predictive/error-driven plasticity still requires trusted held-out feedback.
