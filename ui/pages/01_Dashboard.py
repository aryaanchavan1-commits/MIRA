"""Dashboard page (spec §25): system status, stats, memory composition."""
from __future__ import annotations

import pandas as pd
import psutil
import streamlit as st

from ui.common import get_ctx, get_workspace, header, sidebar_status

st.set_page_config(page_title="MIRA", page_icon="◎", layout="wide")
sidebar_status()
ctx = get_ctx()
ws = get_workspace(ctx)

header("MIRA Dashboard",
       "Mandala-Inspired Memory Architecture — local research prototype. "
       "MIRA is an experimental architecture inspired by the organizational and "
       "visual principles of mandalas/yantras; it does not claim historical "
       "mandalas were neural networks. The usefulness of radial organization "
       "is an empirical research question.")

# --- hardware row ---
hw, rc = ctx.hw, ctx.rc
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("CPU", hw.cpu_name or "?", f"{hw.logical_cores} threads")
c2.metric("RAM", f"{hw.ram_gb:.1f} GB", f"{psutil.virtual_memory().percent:.0f}% used")
c3.metric("GPU", hw.gpu_name or "none",
          f"{hw.vram_gb:.1f} GB VRAM" if hw.gpu_name else "CPU-only")
c4.metric("Disk free", f"{hw.free_disk_gb:.1f} GB")
c5.metric("CUDA", "yes" if hw.cuda_available else "no", hw.cuda_version or "")

# --- resolved runtime ---
st.subheader("Resolved runtime configuration")
rt = rc.to_dict()
cols = st.columns(4)
items = list(rt.items())
for i, col in enumerate(cols):
    with col:
        for k, v in items[i::4]:
            st.markdown(f"`{k}`  \n**{v}**")

if ctx.warnings:
    for w in ctx.warnings:
        st.warning(w)

# --- memory composition ---
st.subheader("Memory workspace")
s = ws.stats()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Memory nodes", s["nodes"])
m2.metric("Edges", s["edges"])
m3.metric("Documents", s["documents"])
m4.metric("Vectors (FAISS)", s["vectors"])

if s["by_type"]:
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Nodes by memory type**")
        st.bar_chart(pd.DataFrame(
            [{"type": k, "count": v} for k, v in sorted(s["by_type"].items())],
        ).set_index("type"))
    with right:
        rings = {}
        sectors = {}
        for n in ws.frame.nodes.values():
            rings[n.ring if n.ring is not None else -1] = rings.get(n.ring if n.ring is not None else -1, 0) + 1
            sectors[n.sector or "unassigned"] = sectors.get(n.sector or "unassigned", 0) + 1
        st.markdown("**Nodes by ring (−1 = unplaced)**")
        st.bar_chart(pd.DataFrame(
            [{"ring": f"R{k}", "count": v} for k, v in sorted(rings.items())]
        ).set_index("ring"))
        st.markdown("**Sectors**")
        st.dataframe(pd.DataFrame(
            [{"sector": k, "nodes": v} for k, v in sorted(sectors.items())],
            columns=["sector", "nodes"]), hide_index=True, use_container_width=True)
else:
    st.info("No memories yet — add documents on the **Documents** page, then "
            "ask a question on the **Chat** page.")
