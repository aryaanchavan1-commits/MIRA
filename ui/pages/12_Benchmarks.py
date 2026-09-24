"""Benchmarks page (spec §30-33): run systems on a dataset, compare, export."""
from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from config.auto_config import DATA_DIR, EXPERIMENTS_DIR
from evaluation.ablation import run_all_systems
from evaluation.benchmark import load_dataset, sample_dataset
from evaluation.report import comparison_table, save_experiment
from ui.common import get_ctx, get_workspace, header, sidebar_status

st.set_page_config(page_title="Benchmarks · MIRA", page_icon="🏁", layout="wide")
sidebar_status()
ctx = get_ctx()
ws = get_workspace(ctx)

header("Benchmarks",
       "All systems share the same frame, stores, embeddings, dataset, and "
       "hardware. Metrics come from actual retrieval output; answer scoring "
       "uses lexical token-F1 (a stated proxy, not an LLM judge).")

if ws.stats()["nodes"] == 0:
    st.info("Ingest documents first — benchmarks run against real memories.")
    st.stop()

st.subheader("1. Dataset")
dataset_path = st.text_input(
    "Local dataset path (JSON/JSONL: [{question, answer, supporting_ids?}])",
    os.path.join(DATA_DIR, "datasets", "custom.json"))
st.caption("Datasets are never downloaded automatically (§30). Create one from "
           "your documents, or point to a compatible local file.")
n_sample = st.number_input("Max questions (0 = all)", 0, 500, 20)

st.subheader("2. Systems")
c1, c2 = st.columns(2)
run_baselines = c1.checkbox("Baselines (vector / graph / hierarchical)", True)
run_ablations = c2.checkbox("MIRA + all ablation sets (§29)", True)
k = st.slider("Retrieval k", 3, 15, 8)

if st.button("Run benchmark", type="primary",
             disabled=not (run_baselines or run_ablations)):
    if not os.path.exists(dataset_path):
        st.error(f"Dataset not found: {dataset_path}")
        st.stop()
    records = load_dataset(dataset_path)
    if not records:
        st.error("Dataset loaded but contained no {question, answer} records.")
        st.stop()
    records = sample_dataset(records, int(n_sample))
    st.info(f"Running {len(records)} questions…")

    progress = st.progress(0.0)
    results = {}
    systems = run_all_systems(ws, records, k=k,
                              include_baselines=run_baselines,
                              include_ablations=run_ablations)
    progress.progress(1.0)
    results.update(systems)

    st.subheader("3. Results")
    st.dataframe(pd.DataFrame(comparison_table(results)),
                 hide_index=True, use_container_width=True)

    metric = st.selectbox("Chart metric", ["retrieval_recall", "mrr",
                                           "context_tokens", "latency_ms"])
    chart_df = pd.DataFrame(comparison_table(results)).set_index("system")
    if metric in chart_df:
        st.bar_chart(chart_df[metric].astype(float))

    exp_name = st.text_input("Experiment name", "bench-run")
    if st.button("Save experiment", type="primary"):
        exp_id = save_experiment(exp_name, {"k": k, "dataset": dataset_path,
                                            "n_questions": len(records)},
                                 results, store=ws.store, hw=ctx.hw)
        st.success(f"Saved as `{exp_id}` under `experiments/` "
                   "(JSON + CSV + Markdown + SQLite).")

st.divider()
st.subheader("Make a tiny dataset from your documents")
st.caption("Creates a lexical-overlap QA set from ingested chunks — a real "
           "smoke benchmark, clearly labeled as synthetic.")
if st.button("Generate custom.json from chunks"):
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    rows = []
    for d in ws.store.list_documents():
        for ch in ws.store.document_chunks(d["id"])[:3]:
            sents = [s.strip() for s in ch["text"].split(".") if len(s.strip()) > 30]
            if not sents:
                continue
            q_sent = sents[0]
            rows.append({"question": f"What does the document say about: {q_sent[:80]}?",
                         "answer": q_sent,
                         "supporting_ids": [n.id for n in ws.frame.nodes.values()
                                            if ch["id"] in n.source_ids][:1]})
    with open(dataset_path, "w", encoding="utf-8") as fh:
        json.dump(rows[:50], fh, ensure_ascii=False, indent=2)
    st.success(f"Wrote {len(rows[:50])} records to `{dataset_path}`")
