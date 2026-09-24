"""Logs page (spec §25/§42): retrieval logs + log file tail."""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from config.auto_config import LOGS_DIR, PROJECT_ROOT
from ui.common import get_ctx, get_workspace, header, sidebar_status

st.set_page_config(page_title="Logs · MIRA", page_icon="📜", layout="wide")
sidebar_status()
ctx = get_ctx()
ws = get_workspace(ctx)

header("Logs", "Structured retrieval telemetry (§42). Document contents are never logged.")

st.subheader("Retrieval logs (SQLite)")
rows = ws.store._conn.execute(
    "SELECT at, system, n_candidates, n_final, latency_ms, context_tokens "
    "FROM retrieval_logs ORDER BY id DESC LIMIT 200").fetchall()
if rows:
    st.dataframe(pd.DataFrame([dict(r) for r in rows]),
                 use_container_width=True, hide_index=True, height=380)
    lat = [r["latency_ms"] for r in rows]
    c1, c2, c3 = st.columns(3)
    c1.metric("Median latency", f"{sorted(lat)[len(lat)//2]:.0f} ms")
    c2.metric("Mean candidates", f"{sum(r['n_candidates'] for r in rows)/len(rows):.0f}")
    c3.metric("Logged queries", len(rows))
else:
    st.caption("No retrieval logged yet — ask something on the Chat page.")

st.divider()
st.subheader("Application log tail")
log_path = os.path.join(LOGS_DIR, "mira.log")
if os.path.exists(log_path):
    n = st.slider("Last N lines", 50, 2000, 200)
    with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()[-n:]
    st.code("".join(lines) or "(empty)", language="text")
else:
    st.caption(f"No log file at {log_path}")
