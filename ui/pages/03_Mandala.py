"""Mandala page (spec §26): topology visualization + node inspector."""
from __future__ import annotations

import streamlit as st

from ui.common import get_ctx, get_workspace, header, sidebar_status
from visualization.mandala_view import figure

st.set_page_config(page_title="Mandala · MIRA", page_icon="◎", layout="wide")
sidebar_status()
ctx = get_ctx()
ws = get_workspace(ctx)

header("Mandala Topology",
       "Radial memory layout: angle = semantic sector, radius = ring / radial "
       "distance. Marker size = importance, color = memory type.")

if ws.stats()["nodes"] == 0:
    st.info("No memories yet — ingest documents first.")
    st.stop()

with st.sidebar:
    st.subheader("View options")
    type_filter = st.multiselect(
        "Memory types",
        sorted({n.memory_type.value for n in ws.frame.nodes.values()}),
        default=sorted({n.memory_type.value for n in ws.frame.nodes.values()}))
    sector_filter = st.multiselect(
        "Sectors",
        sorted({n.sector or "unassigned" for n in ws.frame.nodes.values()}),
        default=sorted({n.sector or "unassigned" for n in ws.frame.nodes.values()}))

keep = sorted(nid for nid, n in ws.frame.nodes.items()
              if n.memory_type.value in type_filter
              and (n.sector or "unassigned") in sector_filter)

fig = figure(ws.frame, height=680)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Node inspector")
nid = st.selectbox("Node", keep,
                   format_func=lambda i: f"{ws.frame.nodes[i].concept}  ({i})")
if nid:
    n = ws.frame.nodes[nid]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ring", n.ring if n.ring is not None else "—")
    c2.metric("Sector", n.sector or "—")
    c3.metric("Confidence", f"{n.confidence:.2f}")
    c4.metric("Importance", f"{n.importance:.2f}")
    st.markdown(f"**Type:** {n.memory_type.value} · **Depth:** {n.depth} · "
                f"**Radial distance:** {n.radial_distance if n.radial_distance is not None else '—'}")
    if n.summary:
        st.markdown(f"**Summary**  \n{n.summary}")
    if n.raw_text:
        with st.expander("Raw text"):
            st.write(n.raw_text)
    kids = n.children
    if kids:
        st.markdown("**Children:** " + ", ".join(
            f"`{ws.frame.nodes[k].concept}`" for k in kids if k in ws.frame.nodes))
    if n.parent_id and n.parent_id in ws.frame.nodes:
        st.markdown(f"**Parent:** `{ws.frame.nodes[n.parent_id].concept}`")
    edges = ws.gs.edges_of(nid)
    if edges:
        rows = [{"relation": e["relation_type"],
                 "other": (ws.frame.nodes[e["target_id"]].concept
                           if e["target_id"] != nid else ws.frame.nodes[e["source_id"]].concept),
                 "weight": e["weight"], "confidence": e["confidence"]}
                for e in edges if e["target_id"] in ws.frame.nodes and e["source_id"] in ws.frame.nodes]
        if rows:
            import pandas as pd
            st.markdown("**Neighbors**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if n.source_ids:
        st.markdown("**Provenance (chunks)**")
        for cid in n.source_ids[:8]:
            ch = ws.store.get_chunk(cid)
            if ch:
                doc = ws.store.get_document(ch["document_id"])
                st.markdown(f"- `{cid}` — {doc['title'] if doc else '?'} "
                            f"p.{ch['page']} · {ch['n_chars']} chars")
