"""Memory Explorer page (spec §25): searchable table of all memory nodes."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.common import get_ctx, get_workspace, header, sidebar_status

st.set_page_config(page_title="Memory · MIRA", page_icon="🗂", layout="wide")
sidebar_status()
ctx = get_ctx()
ws = get_workspace(ctx)

header("Memory Explorer", "Every memory node with its topology placement and provenance.")

if ws.stats()["nodes"] == 0:
    st.info("No memories yet.")
    st.stop()

with st.sidebar:
    st.subheader("Filters")
    types = st.multiselect("Type", sorted({n.memory_type.value for n in ws.frame.nodes.values()}))
    rings = st.multiselect("Ring", sorted({n.ring for n in ws.frame.nodes.values()
                                           if n.ring is not None}))
    query = st.text_input("Search concept/summary", "")

rows = []
for nid, n in ws.frame.nodes.items():
    if types and n.memory_type.value not in types:
        continue
    if rings and n.ring not in rings:
        continue
    if query and query.lower() not in (n.concept + " " + (n.summary or "")).lower():
        continue
    rows.append({
        "id": nid, "concept": n.concept, "type": n.memory_type.value,
        "ring": n.ring, "sector": n.sector, "radial": (
            round(n.radial_distance, 3)
            if n.radial_distance is not None else None),
        "importance": round(n.importance, 2), "confidence": round(n.confidence, 2),
        "sources": len(n.source_ids), "updated": n.updated_at,
    })
df = pd.DataFrame(rows)
if not df.empty:
    df = df.sort_values(["ring", "concept"])
st.dataframe(df, use_container_width=True, hide_index=True, height=480)
st.caption(f"{len(rows)} of {len(ws.frame.nodes)} nodes shown")

nid = st.selectbox("Inspect node", list(df["id"]) if not df.empty else [],
                   format_func=lambda i: f"{ws.frame.nodes[i].concept} ({i})")
if nid and nid in ws.frame.nodes:
    n = ws.frame.nodes[nid]
    st.markdown(f"**Summary**  \n{n.summary or '_none_'}")
    if n.raw_text:
        with st.expander("Raw text"):
            st.write(n.raw_text)
    st.caption(f"created {n.created_at} · updated {n.updated_at} · "
               f"valid {n.valid_from or '—'} → {n.valid_until or '—'} · "
               f"sources: {', '.join(n.source_ids) or '—'}")
