"""MIRA — Mandala-Inspired Memory Architecture.

Streamlit multipage app entrypoint. Pages live in ui/pages/ (Streamlit's
multipage convention). Run with:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from ui.common import get_ctx, header, sidebar_status

st.set_page_config(page_title="MIRA", page_icon="◎", layout="wide")
sidebar_status()

header("◎ MIRA",
       "Mandala-Inspired Memory Architecture — a local research laboratory.")

st.markdown("""
MIRA is an **experimental architecture** inspired by the organizational and
visual principles of mandalas/yantras. It does **not** claim that historical
mandalas were artificial neural networks or that ancient traditions contained
modern AI technology. The usefulness of radial memory organization is an
**empirical research question** — this tool exists to measure it, including
measuring *no benefit*.

### What the research compares

| System | Retrieval |
|---|---|
| Baseline A | Naive Vector RAG |
| Baseline B | Hybrid Vector + Graph RAG |
| Baseline C | Hierarchical retrieval |
| **Baseline D** | **MIRA (radial + hierarchical + graph + semantic)** |

### Where to go

- **Documents** — ingest PDF/TXT/MD/JSON/CSV/DOCX into memory
- **Chat** — ask questions with auditable retrieval paths and sources
- **Mandala** — inspect the radial topology
- **Research Lab** — compare ablation sets on one query
- **Benchmarks** — run baseline vs MIRA comparisons, export results
""")

ctx = get_ctx()
if ctx.warnings:
    for w in ctx.warnings:
        st.warning(w)
