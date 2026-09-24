"""Hardware page (spec §3-6): full profile, live resource monitoring."""
from __future__ import annotations

import plotly.graph_objects as go
import psutil
import streamlit as st

from core.hardware import PROFILE_FRACTIONS
from ui.common import get_ctx, header, sidebar_status

st.set_page_config(page_title="Hardware · MIRA", page_icon="🖥", layout="wide")
sidebar_status()
ctx = get_ctx()

header("Hardware & Runtime",
       "Detected at startup; the runtime configuration below was auto-selected "
       "for this machine (§38). Profiles never assume 100% usage — safety "
       "headroom is reserved.")

hw = ctx.hw
st.subheader("Detected hardware")
d = hw.to_dict()
kc = st.columns(4)
for i, (k, v) in enumerate(sorted(d.items())):
    kc[i % 4].markdown(f"`{k}`  \n**{v}**")

st.subheader("Auto-selected runtime")
for k, v in ctx.rc.to_dict().items():
    st.markdown(f"`{k}` = **{v}**")

st.subheader("Resource policy")
frac = PROFILE_FRACTIONS.get(ctx.rc.performance_mode, {})
st.markdown(f"Mode **{ctx.rc.performance_mode}** uses "
            + ", ".join(f"{k.replace('_frac', '')} {v:.0%}" for k, v in frac.items())
            + " of total RAM/VRAM (rest reserved as safety headroom).")
st.caption(f"usable RAM ≈ {ctx.rc.usable_ram_gb:.1f} GB · "
           f"usable VRAM ≈ {ctx.rc.usable_vram_gb:.1f} GB")

st.subheader("Live resources")
st.plotly_chart(go.Figure(data=[
    go.Indicator(mode="gauge+number", number=dict(suffix=" %"),
                 value=psutil.virtual_memory().percent,
                 title=dict(text="RAM"), gauge=dict(axis=dict(range=[0, 100]))),
], layout=dict(height=220, margin=dict(l=20, r=20, t=30, b=10))),
    use_container_width=True)

if st.button("Re-detect hardware"):
    st.cache_resource.clear()
    st.rerun()
