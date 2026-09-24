"""Models page (spec §5/§6): local GGUF management, gated downloads."""
from __future__ import annotations

import streamlit as st

from models import model_manager as mm
from ui.common import get_ctx, header, sidebar_status

st.set_page_config(page_title="Models · MIRA", page_icon="🧠", layout="wide")
sidebar_status()
ctx = get_ctx()

header("Models",
       "Priority: local installed model → compatible download → lightweight "
       "recommendation. Remote APIs stay disabled (offline-first, §40).")

st.subheader("Local GGUF models")
locals_ = mm.discover_local_gguf()
if locals_:
    for m in locals_:
        c1, c2, c3 = st.columns([4, 2, 2])
        c1.markdown(f"**{m.name}**")
        c2.caption(f"{m.size_gb:.2f} GB")
        c3.caption(f"~{m.params_b or mm._params_from_size(m.size_gb):.1f}B params (est)")
    if ctx.has_llm:
        st.success(f"Loaded: **{ctx.llm.info()['model']}** — "
                   f"gpu_layers={ctx.llm.info()['gpu_layers']}, ctx={ctx.llm.info()['ctx']}")
else:
    st.warning("No GGUF models found under `models/`. Answers will use the "
               "deterministic extractive fallback until a model is added or "
               "downloaded below.")

st.divider()
st.subheader("Download a model")
if ctx.offline:
    st.info("Offline mode is ON (§40) — downloads are disabled. Enable "
            "`offline: false` plus `models.allow_download: true` in "
            "`config/config.yaml` to allow them.")
else:
    with st.form("dl"):
        repo_id = st.text_input("Hugging Face repo id", "bartowski/Qwen2.5-0.5B-Instruct-GGUF")
        filename = st.text_input("Filename",
                                 "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf")
        submitted = st.form_submit_button("Precheck")
    if submitted and repo_id and filename:
        params_b = {"0.5B": 0.5, "1B": 1.0, "1.5B": 1.5, "3B": 3.0, "4B": 4.0}
        est = 0.5
        for k, v in params_b.items():
            if k.lower() in filename.lower():
                est = v
        ok, msg = mm.precheck_download(est, "Q4_K_M", ctx.hw, ctx.cfg)
        (st.success if ok else st.error)(f"Precheck: {msg}")
        if ok:
            confirm = st.checkbox("I confirm this download (disk + memory checked above)")
            if confirm and st.button("Download now", type="primary"):
                with st.spinner("Downloading…"):
                    try:
                        path = mm.download_gguf(repo_id, filename,
                                                token=mm.hf_token_from_env())
                        st.success(f"Saved to `{path}`. Reload models to use it.")
                    except Exception as exc:
                        st.error(f"Download failed: {exc}")
                if st.button("Reload models"):
                    st.cache_resource.clear()
                    st.rerun()
