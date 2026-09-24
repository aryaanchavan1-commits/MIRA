"""Documents page (spec §25/§16): ingestion into the memory system."""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from config.auto_config import DATA_DIR
from ui.common import get_ctx, get_workspace, header, sidebar_status

st.set_page_config(page_title="Documents · MIRA", page_icon="📄", layout="wide")
sidebar_status()
ctx = get_ctx()
ws = get_workspace(ctx)

header("Documents",
       "Ingest PDF / TXT / MD / JSON / CSV / DOCX. Each document is chunked, "
       "embedded, and turned into a mandala of memories with full provenance.")

up = st.file_uploader("Upload document", type=["pdf", "txt", "md", "json", "csv", "docx"])
paste = st.text_area("…or paste text", height=140, placeholder="Paste raw text to memorize…")
title = st.text_input("Title (for pasted text)", "")

c1, c2 = st.columns(2)
if c1.button("Ingest uploaded file", type="primary", disabled=up is None) and up:
    path = os.path.join(DATA_DIR, "uploads", up.name)
    with open(path, "wb") as fh:
        fh.write(up.getvalue())
    with st.spinner("Ingesting…"):
        stats = ws.ingest_file(path, title=up.name)
    st.success(f"Ingested **{up.name}**: "
               f"{stats['n_chunks']} chunks → {stats['n_nodes']} memory nodes, "
               f"{stats['n_edges']} edges in {stats['elapsed_s']:.1f}s")
    st.rerun()

if c2.button("Ingest pasted text", type="primary", disabled=not paste.strip()) and paste.strip():
    with st.spinner("Ingesting…"):
        stats = ws.ingest_text(paste, title=title or "pasted text")
    st.success(f"Ingested: {stats['n_chunks']} chunks → {stats['n_nodes']} memory nodes, "
               f"{stats['n_edges']} edges in {stats['elapsed_s']:.1f}s")
    st.rerun()

st.divider()
st.subheader("Ingested documents")
docs = ws.store.list_documents()
if docs:
    st.dataframe(pd.DataFrame([{
        "id": d["id"], "title": d["title"], "type": d["file_type"],
        "chunks": d["n_chunks"], "chars": d["n_chars"], "ingested": d["ingested_at"],
    } for d in docs]), hide_index=True, use_container_width=True)

    with st.expander("Manage"):
        del_id = st.selectbox("Document to delete",
                              [d["id"] for d in docs],
                              format_func=lambda i: next(
                                  d["title"] for d in docs if d["id"] == i))
        if st.button("Delete document and its memories", type="secondary"):
            removed = ws.delete_document(del_id)
            st.success(f"Deleted document and {removed} memory nodes.")
            st.rerun()

    with st.expander("View chunks of a document"):
        cid = st.selectbox("Document", [d["id"] for d in docs],
                           format_func=lambda i: next(
                               d["title"] for d in docs if d["id"] == i),
                           key="chunks_doc")
        chunks = ws.store.document_chunks(cid)
        for ch in chunks[:12]:
            st.markdown(f"**chunk {ch['seq']}** (p.{ch['page']}) — {ch['n_chars']} chars")
            st.caption(ch["text"][:400] + ("…" if len(ch["text"]) > 400 else ""))
else:
    st.info("No documents yet.")
