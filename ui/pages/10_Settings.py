"""Settings page (spec §25/§39): edit config, reload context."""
from __future__ import annotations

import streamlit as st
import yaml

from config.auto_config import PROJECT_ROOT, build_context
from ui.common import get_ctx, get_workspace, header, sidebar_status

CONFIG_PATH = f"{PROJECT_ROOT}/config/config.yaml"

st.set_page_config(page_title="Settings · MIRA", page_icon="⚙", layout="wide")
sidebar_status()
ctx = get_ctx()
header("Settings",
       "Values are written to `config/config.yaml` and applied on reload. "
       "All retrieval weights are experimental defaults, not optimal values (§21).")

def cfg():
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}

def save(c):
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        yaml.safe_dump(c, fh, sort_keys=False)

c = cfg()
with st.form("settings"):
    st.markdown("**Performance mode (§39)**")
    mode = st.selectbox("Mode", ["safe", "balanced", "fast", "research"],
                        index=["safe", "balanced", "fast", "research"].index(
                            c.get("performance_mode", "balanced")))

    st.markdown("**Mandala topology (§11/§12)**")
    topo = c.get("topology", {})
    strategy = st.selectbox("Placement strategy",
                            ["hybrid_mira", "embedding", "centrality", "hierarchy", "temporal"],
                            index=["hybrid_mira", "embedding", "centrality", "hierarchy",
                                   "temporal"].index(topo.get("placement_strategy", "hybrid_mira")))

    st.markdown("**Retrieval (§21)**")
    retr = c.get("retrieval", {})
    final_k = st.number_input("final_k (memories kept)", 3, 20, int(retr.get("final_k", 8)))
    candidate_k = st.number_input("candidate_k (semantic candidates)", 8, 64,
                                  int(retr.get("candidate_k", 24)))
    max_hops = st.number_input("max_hops (path length)", 1, 5, int(retr.get("max_hops", 3)))

    st.markdown("**Retrieval score weights (experimental)**")
    w = c.get("retrieval_score", {})
    wc1, wc2 = st.columns(2)
    alpha = wc1.slider("α semantic", 0.0, 1.0, float(w.get("alpha_semantic", 0.35)), 0.05)
    beta = wc1.slider("β structural", 0.0, 1.0, float(w.get("beta_structural", 0.15)), 0.05)
    gamma = wc1.slider("γ radial", 0.0, 1.0, float(w.get("gamma_radial", 0.15)), 0.05)
    delta = wc1.slider("δ graph", 0.0, 1.0, float(w.get("delta_graph", 0.15)), 0.05)
    eps = wc2.slider("ε importance", 0.0, 1.0, float(w.get("epsilon_importance", 0.08)), 0.05)
    zeta = wc2.slider("ζ confidence", 0.0, 1.0, float(w.get("zeta_confidence", 0.07)), 0.05)
    eta = wc2.slider("η recency", 0.0, 1.0, float(w.get("eta_recency", 0.03)), 0.05)
    theta = wc2.slider("θ path", 0.0, 1.0, float(w.get("theta_path", 0.02)), 0.05)

    st.markdown("**Context**")
    ctx_cfg = c.get("context", {})
    _auto = str(ctx_cfg.get("max_tokens", 2048)).lower() == "auto"
    budget = st.number_input("max context tokens", 256, 8192,
                             2048 if _auto else int(ctx_cfg.get("max_tokens", 2048)),
                             step=256,
                             help="config had 'auto' — shown as 2048", disabled=False)

    st.markdown("**Offline mode (§40)**")
    offline = st.checkbox("Strict offline (no network, no downloads)",
                          value=bool(c.get("offline", True)))

    saved = st.form_submit_button("Save configuration", type="primary")

if saved:
    c["performance_mode"] = mode
    c.setdefault("topology", {})["placement_strategy"] = strategy
    c.setdefault("retrieval", {}).update(
        final_k=int(final_k), candidate_k=int(candidate_k), max_hops=int(max_hops))
    c["retrieval_score"] = {
        "alpha_semantic": alpha, "beta_structural": beta, "gamma_radial": gamma,
        "delta_graph": delta, "epsilon_importance": eps, "zeta_confidence": zeta,
        "eta_recency": eta, "theta_path": theta,
    }
    c.setdefault("context", {})["max_tokens"] = int(budget)
    c["offline"] = bool(offline)
    save(c)
    st.success("Saved. Reload context to apply (also rebuilds placement).")

st.divider()
st.subheader("Apply changes")
if st.button("Reload context & workspace", type="secondary"):
    ws = get_workspace(get_ctx())
    ws.close()
    st.cache_resource.clear()
    build_context(force_llm_reload=True)
    st.rerun()

with st.expander("Current config.yaml"):
    st.code(c and yaml.safe_dump(c, sort_keys=False) or "{}", language="yaml")
