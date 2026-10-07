# Related Work and Bounded Novelty Claims

Per spec §37: novelty claims are bounded by prior art. For each relevant line
of work: idea, similarity, difference, what MIRA-NCM adds. If a system already
demonstrates an effect, we cite it and claim only the difference. *This table
was compiled from the research program's references and standard literature
knowledge; entries were not re-verified against primary sources this session
and should be spot-checked before publication.*

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

## Bounded novelty statement

MIRA-NCM combines known ingredients (soft clustering, spreading activation,
decay/replay, versioned records, weighted retrieval). The claimable
contribution — **contingent on H2/H3/H4 outcomes** — is: (1) the *overlap
kernel* `overlap(a,b) = Σ_c w_c(a)·w_c(b)` used as an ablatable retrieval term
with per-K attribution; (2) a continual-memory benchmark that scores forgetting
jointly with ranking quality; (3) a fully component-attributed retrieval stack
where every mechanism is measured and removable. Where prior work already
demonstrates an effect, we cite it and claim only the difference.
