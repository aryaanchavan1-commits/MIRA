# acceptance-run

- experiment: `EXP-0002`
- created: 2026-09-22T19:59:22.324943+00:00
- git: `` · seed: 42

| system | retrieval_recall | mrr | answer_token_f1 | context_tokens | latency_ms | questions_per_s | n_questions |
|---|---|---|---|---|---|---|---|
| vector_rag | 0.0 | 0.0 | — | 1.0 | 10.575 | 17.54 | 2 |
| graph_rag | 0.0 | 0.0 | — | 1.0 | 8.87 | 21.09 | 2 |
| hierarchical_rag | 0.0 | 0.0 | — | 1.0 | 1.415 | 37.87 | 2 |
| full_mira | 0.5 | 0.125 | — | 1.0 | 11.42 | 28.73 | 2 |
| vector_only | 0.0 | 0.0 | — | 1.0 | 11.41 | 26.89 | 2 |
| graph_only | 0.0 | 0.0 | — | 1.0 | 10.89 | 30.58 | 2 |
| hierarchy_only | 0.0 | 0.0 | — | 1.0 | 10.96 | 21.3 | 2 |
| radial_only | 0.0 | 0.0 | — | 1.0 | 9.665 | 31.45 | 2 |
| vector_graph | 0.0 | 0.0 | — | 1.0 | 10.01 | 33.53 | 2 |
| vector_hierarchy | 0.0 | 0.0 | — | 1.0 | 8.715 | 30.51 | 2 |
| vector_radial | 0.0 | 0.0 | — | 1.0 | 7.795 | 31.3 | 2 |
| graph_hierarchy | 0.0 | 0.0 | — | 1.0 | 12.425 | 25.6 | 2 |
| graph_radial | 0.0 | 0.0 | — | 1.0 | 10.66 | 21.37 | 2 |

Metrics are computed from actual retrieval output. answer_token_f1 is a lexical proxy, not an LLM judge.
