"""Shared helpers for all MIRA UI pages (spec §25)."""
from __future__ import annotations

import streamlit as st

from config.auto_config import build_context, runtime_summary
from core.workspace import Workspace


@st.cache_resource(show_spinner="Detecting hardware and selecting models…")
def get_ctx():
    """Resolved hardware/model context (cached across reruns)."""
    return build_context()


@st.cache_resource(show_spinner="Opening memory workspace…")
def get_workspace(_ctx) -> Workspace:
    ws = Workspace(embeddings=_ctx.embeddings, llm=_ctx.llm, config=_ctx.cfg)
    return ws


def header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def sidebar_status() -> None:
    ctx = get_ctx()
    ws = get_workspace(ctx)
    st.sidebar.success(f"**{ctx.hw.gpu_name or ctx.hw.cpu_name}**")
    st.sidebar.caption(
        f"RAM {ctx.hw.ram_gb:.0f} GB · VRAM {ctx.hw.vram_gb:.1f} GB · "
        f"tier `{ctx.hw.tier()}` · mode `{ctx.rc.performance_mode}`"
    )
    s = ws.stats()
    st.sidebar.metric("Memories", s["nodes"])
    st.sidebar.metric("Documents", s["documents"])
    if not ctx.embeddings.is_real_model:
        st.sidebar.warning("Embeddings: deterministic fallback (no sentence-transformers model loaded)")
    if ctx.has_llm:
        st.sidebar.caption(f"LLM: {ctx.llm.info()['model']}")
    else:
        st.sidebar.info("LLM unavailable — extractive answers")
