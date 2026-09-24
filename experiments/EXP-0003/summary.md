# acceptance-run

- experiment: `EXP-0003`
- created: 2026-09-22T20:59:21.902868+00:00
- git: `` · seed: 42

| system | retrieval_recall | mrr | answer_token_f1 | context_tokens | latency_ms | questions_per_s | n_questions |
|---|---|---|---|---|---|---|---|
| vector_rag | 0.5 | 0.125 | — | 1.0 | 3.31 | 52.95 | 2 |
| graph_rag | 0.5 | 0.125 | — | 1.0 | 3.785 | 19.82 | 2 |
| hierarchical_rag | 0.0 | 0.0 | — | 1.0 | 0.23 | 78.47 | 2 |
| full_mira | 0.5 | 0.125 | — | 1.0 | 6.03 | 53.69 | 2 |
| vector_only | 0.5 | 0.125 | — | 1.0 | 4.835 | 61.46 | 2 |
| graph_only | 0.5 | 0.125 | — | 1.0 | 5.47 | 64.24 | 2 |
| hierarchy_only | 0.5 | 0.125 | — | 1.0 | 6.005 | 52.4 | 2 |
| radial_only | 0.5 | 0.125 | — | 1.0 | 5.825 | 51.39 | 2 |
| vector_graph | 0.5 | 0.125 | — | 1.0 | 5.96 | 53.62 | 2 |
| vector_hierarchy | 0.5 | 0.125 | — | 1.0 | 5.655 | 15.55 | 2 |
| vector_radial | 0.5 | 0.125 | — | 1.0 | 4.9 | 62.46 | 2 |
| graph_hierarchy | 0.5 | 0.125 | — | 1.0 | 6.0 | 57.76 | 2 |
| graph_radial | 0.5 | 0.125 | — | 1.0 | 5.195 | 60.49 | 2 |

Metrics are computed from actual retrieval output. answer_token_f1 is a lexical proxy, not an LLM judge.
