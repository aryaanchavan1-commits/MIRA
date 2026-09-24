# MIRA Hypotheses

## H1 (primary)
> A radial hierarchical memory topology combining semantic similarity, hierarchical organization, radial proximity, graph relationships, importance, confidence, provenance and temporal information improves evidence retrieval and multi-hop reasoning while reducing irrelevant context.

**Operationalization:** compare MIRA (full) against Baselines A–C (§28) on identical datasets, questions, models, context budget, hardware. Metrics per §31. Registered before running: any outcome is a result.

**Falsification:** if MIRA's recall@k, MRR and multi-hop accuracy are not statistically distinguishable from Baseline B (hybrid vector+graph) at equal token budgets, H1 is not supported → document the negative result (§52).

## H2 (radial signal)
> Radial proximity (the α/β/γ/δ weighted radial distance, §12) is a useful retrieval feature beyond what semantic similarity + graph features already provide.

**Operationalization:** ablation pair — full MIRA vs. full MIRA with γ=0 (radial term removed) and β=0 (hierarchy term removed). If removing radial/hierarchy terms does not degrade any §31 metric, the mandala topology adds nothing beyond its ingredients → H2 falsified.

## H3 (compression)
> Path-based retrieval + compression (§22-23) reduces context tokens without accuracy loss.

**Operationalization:** equal-token comparison: Baseline A given the same token budget as MIRA's compressed context. If MIRA at equal tokens doesn't beat Vector RAG, the compression pipeline isn't earning its complexity.

## H4 (sector semantics)
> Embedding-discovered sectors (§10) align with human-meaningful topic partitions.

**Operationalization:** qualitative + cluster-coherence metric; exploratory, not required for H1.

---

## Honesty clauses (spec §49)
- All weights (§12/§21) are **experimental defaults**, not optimal values.
- A **negative result is a valid result** and will be documented as such.
- The system does **not** claim historical mandalas were neural networks or that ancient traditions contained modern AI. MIRA is an engineering artifact *inspired by* the organizational and visual principles of mandalas/yantras; its usefulness is an empirical question.
