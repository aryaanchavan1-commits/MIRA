# Related Work and Bounded Novelty Claims

Per spec §37: novelty claims are bounded by prior art. For each relevant line
of work: idea, similarity, difference, what MIRA-NCM adds. If a system already
demonstrates an effect, we cite it and claim only the difference.

*Provenance note:* the core table was compiled from the research program's
references and standard literature knowledge. The §2026-landscape section
below was added after a live web search (Oct 2026). Vendor-reported benchmark
numbers (Mem0, ByteRover, Supermemory, Hindsight marketing pages) are cited
**as vendor-reported** — they were not independently verified and several are
contested publicly (see the MemPalace/LoCoMo saturation note). Before
publication, spot-check every arXiv ID against the primary source.

| Work | Year | Idea | Similarity to MIRA | Difference | What MIRA-NCM adds |
|---|---|---|---|---|---|
| Kanerva, *Sparse Distributed Memory* | 1988 | Address-space memory with decay counters and positional structure | Radial/positional organization; decay | SDM is a fixed-address store; MIRA organizes by corpus-induced geometry | Overlap kernel over soft cluster membership, ablatable per K |
| Anderson, *Spreading Activation* / ACT-R lineage | 1983 | Activation spreads through associative links | `core/activation.py` spreading activation [ENG] | ACT-R models cognition; MIRA retrieves documents | Activation is one ablatable term among ten, measured by leave-one-out |
| Hebb (1949); Bi & Poo (1998) | 1949/1998 | Associative and time-sensitive plasticity | Hebbian/STDP-*inspired* traces | Biological fidelity not claimed | Explicit biologically-inspired-but-not-equivalent scoping |
| McClelland et al., Complementary Learning Systems | 1995 | Fast trace + slow consolidation + gist | decay/replay/gist triangle in BioMIRA | CLS is a neuroscience theory | Implemented as retrieval dynamics; measured recovery (+0.0307) of replayed aged memories |
| Wilson & McNaughton (1994); Marr (1971) | — | Replay during consolidation | Sleep-pass replay | Same difference as CLS | Quantified replay targeting, not just decay |
| Lewis et al., RAG | 2020 | Retrieve-then-generate | Whole setting | RAG is memoryless per query | Persistent organized store with provenance and explanation |
| GraphRAG (Edge et al.) | 2024 | Community summaries for global queries | Hierarchical summary path | GraphRAG builds communities offline for summarization | Interference measured jointly with ranking under a frozen LLM |
| HippoRAG (Gutiérrez et al.) | 2024 | KG + PageRank retrieval for multi-hop | Graph path scoring, multi-hop focus | HippoRAG's personalization is seed-based PPV | Constellation overlap as a *soft* membership alternative; K-sweep ablation; honest IndicQA boundary |
| Generative Agents (Park et al.) | 2023 | Memory stream + recency/importance/relevance scoring + reflection | Weighted retrieval over stored events; consolidation as reflection | No geometry, no versioning, no forgetting metrics | Ring/sector coordinates, version chains, forgetting-lab protocol |
| MemGPT | 2023 | OS-style paged context management | Long-term memory for agents | MemGPT manages the context window; MIRA organizes the store | Store-side structure, not context paging |
| Soft/overlapping clustering; MoE routing | — | Soft membership, routers | Constellation membership is soft clustering over embeddings | Those are general tools, not memory ablations | Membership-as-scoring-term with attribution per component |
| Quantum-inspired IR / ML | 1990s– | Probabilistic amplitude/interference representations | Optional H6 backend | Almost all such work is classification/ranking heuristics | Explicitly optional, required to earn its place or be removed; never called "quantum memory" |
| Continual learning surveys (Parisi, De Lange, GEM) | 2017–2022 | Forgetting metrics, replay baselines | Forgetting lab protocol [ENG] | Those benchmarks train weights | Frozen-LLM retrieval interference; forgetting scored *jointly* with ranking quality (weak systems trivially "win" forgetting alone) |
| BM25 (Robertson & Zaragoza) | 2009 | Lexical ranking | Mandatory baseline | — | MIRA's own IndicQA results prove the rule: BM25 wins where dense fails |

## 2026 landscape (live search, Oct 2026)

What changed since the program was drafted, and where MIRA-NCM stands:

| Work (2025–2026) | Idea | Similarity to MIRA | Difference | What MIRA-NCM adds |
|---|---|---|---|---|
| Mem0 (production agent-memory layer; vendor-reported 93.4% LongMemEval, 92.5% LoCoMo) | LLM-driven memory extraction with ADD/UPDATE/DELETE/NOOP operations over a vector store | Persistent store-side agent memory with update semantics | Mem0's intelligence lives in LLM extraction prompts; the store itself is an unattributed vector index | Store-side *structure* (constellations, version chains) with every scoring term ablatable and attributable; no LLM in the retrieval loop |
| Hu et al., *Memory in the Age of AI Agents* (arXiv 2512.13564) | Survey arguing long/short-term taxonomy is insufficient for agent memory | Motivates richer memory organization | Survey/taxonomy, no measured mechanism | Measured geometric organization with leave-one-out attribution on real benchmarks |
| LongMemEval / LongMemEval-V2 (arXiv 2605.12493) | Long-horizon conversational memory benchmarks (single-hop, temporal, multi-hop, open-domain) | Multi-hop + temporal evaluation axes | Chat-session transcripts; answers scored end-to-end | Retrieval-level MuSiQue/HotpotQA with node-level gold, plus a forgetting protocol scored jointly with ranking |
| BEAM (BEAM-1M / BEAM-10M) | Million-scale memory benchmarks stressing store size | Scale axis (memory_count monotonicity measured at 4.5k–23.5k) | MIRA has no million-node result — an honest open boundary | Budget caps (MAX_CONSTELLATIONS, MAX_DEPTH) designed for that scale; million-node runs are future work |
| MemPalace "100% LoCoMo" claim + community critique | Marketing claim of saturated benchmark scores; its own docs reportedly call the scores meaningless | — | Cautionary tale | MIRA reports MRR with paired-bootstrap CIs and keeps a mandatory BM25/vector floor; we treat saturated vendor scores as unverified |
| Letta/MemGPT "is a filesystem all you need" benchmarking | Simple baselines remain competitive against fancy memory systems | Same finding as our IndicQA result (BM25 strongest) | Their conclusion is about context management; ours is about retrieval components | Eleven components, each individually removable; baselines always reported |
| FOREVER, forgetting-curve-inspired replay for LLM continual learning (arXiv 2601.03938) | Ebbinghaus-style curves schedule weight-space replay | Forgetting curves + replay are MIRA's consolidation pass | FOREVER fine-tunes weights; MIRA freezes the LLM and measures retrieval interference | Retrieval-level forgetting lab (A–K) scoring capability *and* retention jointly; measured replay recovery (+0.0307 MRR) of aged memories |
| Spurious-forgetting results (OpenReview, cited by 82) | Regularization/generative replay/merging fail against *spurious* forgetting in LLM continual learning | Supports the thesis that forgetting needs task-specific measurement | Weight-training setting | Frozen-LLM interference measurement; no claim that MIRA "solves" catastrophic forgetting |
| Huynh et al., *Quantum-inspired machine learning: a survey* (2017–2025 corpus) | First dedicated QiML survey: tensor networks, amplitude encodings, Born-rule probabilities in classical models | Born-rule amplitude kernel (`BornRuleIndex`) is exactly a classical Born-rule representation | QiML work targets classification/compression; none (found) applies Born-rule kernels to memory *retrieval scoring* with ablation | H6: an interference-like retrieval kernel that must beat the linear kernel or be removed; interference property unit-verified (2.0× vs 1.41×) |
| Tensor-network quantum-kernel frameworks (2026) | TN contraction simulates quantum kernels for classification | Kernel methods with amplitude structure | Quantum-kernel classification, not persistent memory | Explicit scoping: classical math on CPU; never called "quantum memory" |

**Reading of the landscape:** the 2026 agent-memory market has consolidated
around LLM-driven extraction pipelines over unattributed vector stores, with
benchmark numbers racing toward saturation and public disputes about their
validity. The under-explored gap MIRA-NCM targets is exactly there: *store-side
structure whose every term is measurable and removable*, and forgetting
evaluated jointly with capability rather than as a marketing axis. That gap
statement is a positioning claim, not a superiority claim — it stands or falls
with the H2/H6 measurements.

## Bounded novelty statement

MIRA-NCM combines known ingredients (soft clustering, spreading activation,
decay/replay, versioned records, weighted retrieval). The claimable
contribution — **contingent on H2/H3/H4 outcomes** — is: (1) the *overlap
kernel* `overlap(a,b) = Σ_c w_c(a)·w_c(b)` used as an ablatable retrieval term
with per-K attribution; (2) a continual-memory benchmark that scores forgetting
jointly with ranking quality; (3) a fully component-attributed retrieval stack
where every mechanism is measured and removable. Where prior work already
demonstrates an effect, we cite it and claim only the difference.
