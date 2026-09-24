"""Experiments page (spec §25/§33): browse and export saved experiments."""
from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from config.auto_config import EXPERIMENTS_DIR
from ui.common import get_ctx, get_workspace, header, sidebar_status

st.set_page_config(page_title="Experiments · MIRA", page_icon="🧪", layout="wide")
sidebar_status()
ctx = get_ctx()
ws = get_workspace(ctx)

header("Experiments",
       "Every benchmark run is stored with config, hardware, seed, and results "
       "for reproducibility (§33).")

exps = ws.store.list_experiments()
disk_dirs = sorted(d for d in os.listdir(EXPERIMENTS_DIR)
                   if os.path.isdir(os.path.join(EXPERIMENTS_DIR, d))) \
    if os.path.isdir(EXPERIMENTS_DIR) else []

if not exps:
    st.info("No experiments yet — run one on the **Benchmarks** page.")
else:
    ids = [e["id"] for e in exps]
    sel = st.selectbox("Experiment", ids,
                       format_func=lambda i: f"{i} — {next(e['name'] for e in exps if e['id'] == i)}")
    e = next(x for x in exps if x["id"] == sel)
    c1, c2, c3 = st.columns(3)
    c1.metric("Created", e["created_at"] or "—")
    c2.metric("Seed", e.get("seed", "—"))
    c3.metric("Git commit", (e.get("git_commit") or "—")[:10])

    with st.expander("Stored configuration", expanded=False):
        st.code(json.dumps(e.get("config", {}), indent=2, ensure_ascii=False))
    with st.expander("Hardware snapshot", expanded=False):
        st.code(json.dumps(e.get("hardware", {}), indent=2, ensure_ascii=False))

    st.subheader("Results")
    results = e.get("results", {})
    st.code(json.dumps(results, indent=2, ensure_ascii=False), language="json")

    st.subheader("Export")
    ec1, ec2, ec3 = st.columns(3)
    payload = json.dumps({k: e[k] for k in ("id", "created_at", "name", "config",
                                            "results", "hardware", "seed")},
                         indent=2, ensure_ascii=False, default=str)
    ec1.download_button("JSON", payload, f"{sel}.json", "application/json")
    flat = pd.json_normalize(results).to_csv(index=False)
    ec2.download_button("CSV", flat, f"{sel}_results.csv", "text/csv")
    md = [f"# Experiment {sel} — {e['name']}", "",
          f"- created: {e['created_at']}", f"- seed: {e.get('seed')}", "",
          "## Results", "", "```json", payload, "```"]
    ec3.download_button("Markdown", "\n".join(md), f"{sel}.md", "text/markdown")

if disk_dirs:
    st.divider()
    st.subheader("Experiment directories on disk")
    st.caption(", ".join(f"`{d}`" for d in disk_dirs))
