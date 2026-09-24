"""Chat page (spec §27): answer, memories, path, sources, metrics.

Shows only the auditable retrieval path and evidence — never a hidden
chain-of-thought.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.retrieval import ablation_configs
from ui.common import get_ctx, get_workspace, header, sidebar_status

st.set_page_config(page_title="Chat · MIRA", page_icon="💬", layout="wide")
sidebar_status()
ctx = get_ctx()
ws = get_workspace(ctx)

header("Chat",
       "Answers come from ingested memories only. Every answer shows its "
       "retrieved memories, reasoning path, and source documents.")

if ws.stats()["nodes"] == 0:
    st.info("Ingest a document first on the **Documents** page.")
    st.stop()

with st.sidebar:
    st.subheader("Retrieval settings")
    ablation = st.selectbox("Ablation set (§29)", list(ablation_configs().keys()),
                            index=0, help="Switch retrieval components on/off to "
                            "measure which contribute.")
    budget = st.selectbox("Context budget (tokens)", [512, 1024, 2048], index=2)
    show_components = st.checkbox("Show per-component scores", value=False)

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask about your ingested documents…")
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["text"])

if question:
    st.session_state.history.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and composing…"):
            ans = ws.ask(question, active_components=ablation_configs()[ablation],
                         system_name="mira" if ablation == "full_mira" else ablation)
        st.markdown(ans.text)
        if ans.confidence_note:
            st.caption(f"⚠️ {ans.confidence_note}")
        st.caption(f"answer mode: **{ans.mode}** "
                   f"({'local GGUF' if ans.mode == 'llm' else 'deterministic fallback'})")

        tabs = st.tabs(["Retrieved memories", "Reasoning path", "Sources", "Metrics"])
        with tabs[0]:
            if ans.memories:
                st.dataframe(pd.DataFrame([{
                    "#": i + 1, "concept": m["concept"], "type": m["type"],
                    "ring": m["ring"], "sector": m["sector"], "score": m["score"],
                    **({"components": m["components"]} if show_components else {}),
                } for i, m in enumerate(ans.memories)]),
                    hide_index=True, use_container_width=True)
            else:
                st.caption("no memories retrieved")
        with tabs[1]:
            if ans.path_labels:
                for p in ans.path_labels:
                    st.markdown(f"- `{p}`")
            else:
                st.caption("no multi-hop path was needed for this answer")
        with tabs[2]:
            st.write("\n".join(f"- {s}" for s in ans.sources) or "_no provenance_")
        with tabs[3]:
            mcols = st.columns(4)
            mcols[0].metric("Latency", f"{ans.metrics['latency_ms']:.0f} ms")
            mcols[1].metric("Memories used", ans.metrics["n_memories"])
            mcols[2].metric("Context tokens", ans.metrics["context_tokens"])
            mcols[3].metric("Compression",
                            f"{ans.metrics['compression_ratio']:.2f}×")
            st.caption(f"candidates: {ans.metrics['n_candidates']} · "
                       f"retrieval: {ans.metrics['retrieval_ms']:.1f} ms · "
                       f"mode: {ans.metrics['llm_mode']}")

    st.session_state.history.append({"role": "assistant", "text": ans.text})
