# The MIRA Framework — Memory as an Attributed, Ablatable, Versioned Structure

One framework unifies everything in this repository. It is stated once, formally, here;
every other document (the two papers, the protocol, the audit, the module docstrings)
is an instance of it, and every future mechanism must enter through it.

## 0. The one-sentence definition

> **A memory system is an explicit, attributable scoring function over a structured,
> versioned store — and no mechanism is allowed to exist unless its removal measurably
> matters.**

Three properties define membership in the framework:

1. **Attributed** — every score decomposes into named, logged terms.
2. **Ablatable** — every term can be switched off; every claim carries its ablation.
3. **Versioned** — facts are never silently overwritten; current = chain head, with
   provenance and rollback.

Anything that cannot be stated in this shape is not part of MIRA, however attractive.

## 1. Formal model

**Store.** Nodes `n` with coordinates (ring, sector, depth, radial_distance), text,
type, metadata; edges `e` typed and weighted; documents/chunks; version records.
Durable state in SQLite; vectors in FAISS; topology in NetworkX.

**Scoring.** Retrieval score for node `n` given query `q`:

```
score(q, n) = α·semantic + β·structural + γ·radial + δ·graph
            + ε·importance + ζ·confidence + η·recency + θ·path
            + ι·activation + κ·stability + μ·constellation
```

Eleven terms, one Greek letter each, identical to the code (`core/retrieval.py`).
Per-retrieval, every term's value is logged and returned — attribution is not a
feature, it *is* the score.

**Constellations (NCM).** k-means over embeddings induces ≤ C centroids; each node
joins its top-K clusters with L1-normalized weights. The association kernel:

```
overlap(q, n) = Σ_c w_q(c) · w_n(c)            (linear, default)
born(q, n)    = ( Σ_c √(w_q(c)·w_n(c)) )²      (Born-rule, H6 — must earn its place)
```

**Versioning.** Chain key = sha256(subject)[:16]; every update appends a record
`{content, hash, supersedes, at, source, provenance}`; rollback appends a new head
citing the superseded hash. Never rewritten. Cap `max_depth` in memory; SQLite
`versions` table is cold history. Append-only is the śruti-shaped property and is
unit-enforced.

**Provenance labels (pramāṇa).** Every version record carries a derived epistemic
class — pratyakṣa (direct input), anumāna (system-derived), upamāna (comparison),
śabda (testimony, the default) — computed at read time from the source. Independent
knowledge sources can conflict differently; contradiction-aware scoring builds on
this label (future protocol work).

**Dynamics.** Decay (configurable half-life), bounded replay during consolidation,
gist abstraction over members, Hebbian/STDP-inspired association traces, stability.
All engineering analogies; all measured by the forgetting lab (A–K), scored
**jointly**: retention alone is meaningless for a system that cannot rank.

**Governance (the entry gate).** Every mechanism: (a) enters with weight 0 and
default-off; (b) states a pre-registered hypothesis with success AND rejection
thresholds (`docs/EXPERIMENT_PROTOCOL.md`); (c) is ablated leave-one-out on real
benchmarks with paired-bootstrap CIs against mandatory floors (BM25, flat vector,
hierarchical); (d) is **removed** if it fails its own gate. The ledger so far:
structural/activation earn their weight (Δ −0.0442 / −0.0179); radial is marginal
(−0.0098, p=0.052); homeostasis hurt and was cut (K variant); the default-K
constellation term is currently net-negative at scale (0.6897 vs 0.6985, p=0.0002)
and the Born kernel worse still (0.6354, p=0.0001) — both remain default-off until
their pre-registered rescues (K-sweep; ambiguity benchmark) or removal.

## 2. The five layers (one implementation each)

| Layer | Question it answers | Implementation |
|---|---|---|
| 1. Store | Does it persist, exactly? | `storage/` (SQLite, FAISS, graph), `versions` table |
| 2. Structure | How is it organized? | rings/sectors, hierarchy, typed edges, constellations |
| 3. Dynamics | How does it change? | decay/replay/gist, Hebbian traces, staleness |
| 4. Attribution | Why was it retrieved? | 11-term logging, evidence paths, explanations |
| 5. Governance | Is any of it true? | protocol, ablation ladder, baselines, kill criteria |

The layers are strictly ordered: governance can veto anything below it; no layer
below may bypass the one above. Attribution is the contract between layers 1–3 and
layer 5: structure and dynamics are only visible to science through their terms.

## 3. Interfaces

- **Engine:** `core.workspace.Workspace` — ingest/ask/refresh/stats (server and console ride on it).
- **SDK (the framework's single programmatic surface):** `mira_sdk.MemoryEngine` —
  `add / retrieve / explain / ask / remember / history / rollback / consolidate / stats`.
  Explanations carry all eleven components; version methods carry pramāṇa + Kaṭapayādi
  labels. `MemoryEngine(load_llm=False)` runs retrieval-only on small machines.
- **Evaluation:** `evaluation/benchmark.py` + `scripts/eval_*` — every condition is a
  named scoring configuration; nothing is hand-measured.
- **Self-check:** `scripts/check_memory_system.py` — the end-to-end runnable proof.

## 4. What the framework claims — and refuses to claim

Claims (measured): multi-hop ranking wins on MuSiQue/HotpotQA with the full attributed
stack; replay recovers aged memories (+0.0307 MRR) while fade declines (−0.0653);
component attribution predicts relevance; the store grows monotonically with measured
caps. Honest boundaries: recall@8 ties flat vectors; IndicQA belongs to BM25; graph
and recency are inert on this corpus; NCM's default configuration is net-negative at
scale pending its K-sweep.

Refusals (permanent): no "solves catastrophic forgetting" (we measure retrieval
interference with the LLM frozen); no quantum hardware/speedup/memory claims (the
Born kernel is classical math on CPUs and must beat its linear twin or die); no AGI
or superintelligence capability claims (the oversight-relevant property is
auditability — attributable retrieval, versioned beliefs with rollback — not
intelligence); no effectiveness claims for the Indic mappings (they are testable
organizational priors, the ledger decides).

## 5. Porting the framework

To move the framework to another system, port five things, in this order: the version
table (1), the term-logging score function (4), the ablation harness + baselines (5),
the consolidation pass (3), the structure (2). A port without (4) and (5) is not MIRA —
it is a memory system with the same name and none of the evidence.
