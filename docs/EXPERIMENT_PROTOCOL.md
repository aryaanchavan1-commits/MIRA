# Experiment Protocol — MIRA-NCM (pre-registered)

Written **before** the final NCM evaluation runs. Success thresholds are fixed
here and will not move after results are seen. Exploratory extensions must be
labeled exploratory. Statuses follow the taxonomy in `paper/ncm_program.md`:
[CS] established CS, [ENG] MIRA engineering, [HYP] hypothesis, [DEMO]
demonstrated by committed artifacts, [SPEC] speculation.

## Research questions

Primary: *Does structured, sparse, overlapping constellation-based memory
organization improve long-term retrieval, multi-hop retrieval, temporal
reasoning, and continual retention vs conventional vector, graph, hierarchical,
and non-overlapping radial organizations?* Secondary: nesting without
duplication. Tertiary: versioned persistence + adaptive dynamics under bounded
cost. Optional: quantum-inspired representations. No answer is assumed.

## Hypotheses, thresholds, rejection conditions

**H1 [radial]** — Supported if removing the radial term costs ≥ 0.02 MRR on
multi-hop corpora at p < 0.05. *Current status [DEMO], borderline*: minus-radial
costs 0.0098 (p=0.052) on MuSiQue. Rejected if no significant cost at larger n.

**H2 [constellation overlap]** — Supported if the best K>1 constellation
variant beats the strongest K=1 and no-constellation variants by ≥ 0.03 MRR on
2-hop and ≥ 0.05 on 3–4-hop benchmarks, with recall@8 not significantly worse
(paired CI includes 0) and latency overhead ≤ +15 %. Rejected if gains do not
increase with hop depth or if K=1 performs equally (that would mean plain
clustering, not overlap, explains any effect).

**H3 [nesting]** — Supported if hierarchy-sensitive tasks gain ≥ 0.03 MRR with
total node growth < 10 % (summaries replace duplication). Rejected if gains
require MAX_DEPTH > 4 or duplicated storage.

**H4 [versioning]** — Supported if the versioned variant cuts
wrong-current-answer errors by ≥ 50 % on the temporal and contradiction suites
relative to last-write-wins. Rejected if recency weighting alone performs
equally, or version history exceeds 30 % of store size.

**H5 [adaptive dynamics]** — Supported if average forgetting drops ≥ 30 % with
MRR difference not significantly worse than static MIRA (paired CI ≤ 0) across
intervals {7, 30, 90 days}. *Current status [DEMO], partial*: forgetting
0.1074 → 0.0527 replicated at 7/30/90 d (0.039/0.053/0.038); MRR CI
[−0.014, +0.037] does not exclude a small loss (p=0.378). Rejected if the MRR
cost becomes significantly negative at final n.

**H6 [quantum-inspired]** — Supported only if the optional classical
probabilistic backend beats the identical classical scorer on a pre-defined
ambiguous-membership benchmark at equal compute. Rejected (and removed from
the default stack) if no improvement. No claim of quantum advantage is
permitted at any stage.

## Variants and controls

Baselines B0 keyword/BM25 · B1 vector · B2 graph · B3 hierarchical. MIRA
ladder M1 (−radial) · M2 MIRA · M3 +graph · M4 +constellation (K sweep 1/2/3/5)
· M5 +nesting · M6 +versioning · M7 +dynamics · M8 full NCM · M9 +quantum-
inspired. Fairness: same questions, embedder, LLM, context budget, K,
temperature, seeds, preprocessing, evaluation code for every arm; storage and
latency reported beside accuracy. Existing MIRA must remain bit-for-bit the
default (`ncm.enabled=false`).

## Datasets and leakage

Synthetic generators (committed, seeded, versioned) for: multi-hop chains
(1–4 hop), temporal belief revision (current/historical/superseded/uncertain
labels), contradiction sequences, distractor-heavy noise. External: MuSiQue,
HotpotQA, IndicQA as already committed. Leakage tests: duplicate documents,
repeated questions, answer exposure, embedding contamination.

## Statistics

≥ 3 seeds where variance exists; paired bootstrap with 95 % CI as primary
test (per-question paired structure); Wilcoxon signed-rank as non-parametric
cross-check; effect size reported beside every p; significance never conflated
with practical meaning (standing example: answer-stage F1 p=0.0086 with Δ≈0).

## Metrics

MRR, recall@8, path accuracy (multi-hop), wrong-current-answer rate (temporal),
forgetting = best−final per task, retention, memory growth, consolidation and
replay counts, retrieval/ingestion latency, RAM, storage, overhead of NCM
structures. Failure cases exported (`failure_examples.json`) with the taxonomy:
semantic confusion, wrong constellation, wrong path, wrong temporal ordering,
over/under-retrieval, stale memory, excessive overlap, storage blowup,
multilingual failure.

## Resource envelope

16 GB RAM / 4 GB VRAM class. Targets: retrieval p95 < 2 s at 10⁵ nodes; NCM
structure overhead < 25 % of node payload; scaling measured at 1K/10K/50K/100K
before any larger-scale claim.
