"""MIRA web API — FastAPI wrapper around the research core.

Same Workspace core the Streamlit UI and tests use. The static frontend in
web/ is served at "/" so the whole product is one local process. Run:

    .venv/Scripts/python.exe -m uvicorn server:app --port 8000
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mira.server")

from config.auto_config import DATA_DIR, build_context, runtime_summary
from core.workspace import Workspace

app = FastAPI(title="MIRA", docs_url="/api/docs")

_ctx = build_context()
_ws: Optional[Workspace] = None
_ws_lock = threading.Lock()  # uvicorn serves sync handlers from a threadpool


def ws() -> Workspace:
    global _ws
    if _ws is None:
        with _ws_lock:  # double-checked: concurrent first requests must build once
            if _ws is None:
                _ws = Workspace(embeddings=_ctx.embeddings, llm=_ctx.llm,
                                config=_ctx.cfg)
    return _ws


WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


@app.on_event("startup")
def _warmup() -> None:
    # Load models before the first request so the UI never stares at a
    # half-initialized workspace. Embeddings warm first (auto_config), then
    # the LLM — first boot takes ~30-60s on this hardware.
    logger.info("warming models (embeddings, then LLM) — first boot takes 30-60s")
    try:
        ws()
        logger.info("workspace ready — all systems available")
    except Exception as exc:
        logger.warning("workspace warmup failed: %s", exc)


@app.get("/api/system")
def system() -> Dict[str, Any]:
    s = ws().stats()
    out = runtime_summary(_ctx)
    out["workspace"] = s
    out["node_count"] = len(ws().frame.nodes)
    return out


# ---- chat ----
from core.retrieval import ablation_configs


@app.post("/api/chat")
def chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    q = (payload.get("question") or "").strip()
    if not q:
        raise HTTPException(400, "question required")
    comp = payload.get("components")
    if comp == "all":
        comp = None
    elif comp is not None:
        allowed = set(ablation_configs().keys())
        if comp not in allowed:
            raise HTTPException(400, f"unknown ablation set {comp!r}")
        comp = ablation_configs()[comp]
    try:
        ans = ws().ask(q, active_components=comp, system_name="web",
                       allow_web=payload.get("allow_web"))
    except ValueError as exc:  # bad component names from API callers
        raise HTTPException(400, str(exc))
    return {
        "answer": ans.text, "mode": ans.mode,
        "agent_mode": getattr(ans, "agent_mode", "memory"),
        "memories": ans.memories,
        "paths": ans.path_labels, "sources": ans.sources,
        "metrics": ans.metrics, "note": ans.confidence_note,
    }


# ---- mandala ----
@app.get("/api/mandala")
def mandala() -> Dict[str, Any]:
    frame = ws().frame
    nodes = [{
        "id": n.id, "concept": n.concept, "type": n.memory_type.value,
        "ring": n.ring, "sector": n.sector,
        "radial": n.radial_distance, "importance": round(n.importance, 3),
        "confidence": round(n.confidence, 3), "summary": (n.summary or "")[:200],
        "n_sources": len(n.source_ids),
    } for n in frame.nodes.values()]
    edges = [{"s": e.source_id, "t": e.target_id, "r": e.relation_type,
              "w": e.weight} for e in frame.edges]
    return {"nodes": nodes, "edges": edges,
            "max_rings": (_ctx.cfg.get("topology", {}) or {}).get("max_rings", 5)}


@app.get("/api/node/{node_id}")
def node_detail(node_id: str) -> Dict[str, Any]:
    frame = ws().frame
    if node_id not in frame.nodes:
        raise HTTPException(404, "node not found")
    n = frame.nodes[node_id]
    edges = ws().gs.edges_of(node_id)
    neighbors = []
    for e in edges:
        other = e["target_id"] if e["target_id"] != node_id else e["source_id"]
        if other in frame.nodes:
            neighbors.append({"id": other, "concept": frame.nodes[other].concept,
                              "relation": e["relation_type"], "weight": e["weight"]})
    provenance = []
    for cid in n.source_ids[:8]:
        ch = ws().store.get_chunk(cid)
        if ch:
            doc = ws().store.get_document(ch["document_id"])
            provenance.append({"chunk_id": cid, "document": doc["title"] if doc else "?",
                               "page": ch["page"], "preview": ch["text"][:200]})
    return {
        "id": n.id, "concept": n.concept, "type": n.memory_type.value,
        "ring": n.ring, "sector": n.sector, "depth": n.depth,
        "radial": n.radial_distance, "importance": n.importance,
        "confidence": n.confidence, "summary": n.summary, "raw_text": n.raw_text[:800],
        "parent": n.parent_id, "children": n.children,
        "neighbors": neighbors, "provenance": provenance,
        "created_at": n.created_at, "updated_at": n.updated_at,
    }


# ---- documents ----
@app.get("/api/documents")
def documents() -> Dict[str, Any]:
    return {"documents": ws().store.list_documents()}


@app.post("/api/documents")
def upload_document(file: UploadFile = File(...), title: str = Form("")) -> Dict[str, Any]:
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in {".pdf", ".txt", ".md", ".json", ".csv", ".docx"}:
        raise HTTPException(400, f"unsupported file type {ext!r}")
    dest = os.path.join(DATA_DIR, "uploads", os.path.basename(file.filename))
    with open(dest, "wb") as fh:
        shutil.copyfileobj(file.file, fh)
    try:
        stats = ws().ingest_file(dest, title=title or file.filename)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return stats


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str) -> Dict[str, Any]:
    removed = ws().delete_document(document_id)
    return {"deleted": document_id, "nodes_removed": removed}


# ---- research lab ----
@app.post("/api/lab")
def lab(payload: Dict[str, Any]) -> Dict[str, Any]:
    q = (payload.get("question") or "").strip()
    sets = payload.get("sets") or ["full_mira", "vector_only", "graph_only", "radial_only"]
    k = int(payload.get("k", 8))
    if not q:
        raise HTTPException(400, "question required")
    w = ws()
    from core.answer import AnswerPipeline
    pipe = AnswerPipeline(w.frame, w.vs, w.gs, w.embeddings, llm=w.llm,
                          config=w.config, doc_titles=w._doc_titles())
    retriever = pipe.retriever
    qvec = w.embeddings.encode([q])[0]
    out = {}
    for name in sets:
        comps = ablation_configs().get(name)
        if comps is None and name != "full_mira":
            continue
        res = retriever.retrieve(q, qvec, active_components=comps, final_k=k)
        out[name] = {
            "latency_ms": res.latency_ms, "n_candidates": res.n_candidates,
            "items": [{"id": it.node.id, "concept": it.node.concept,
                       "type": it.node.memory_type.value, "ring": it.node.ring,
                       "sector": it.node.sector, "score": round(it.score, 4),
                       "hops": max(0, len(it.path) - 1)} for it in res.items],
        }
    ids = {n: {it["id"] for it in v["items"]} for n, v in out.items()}
    jacc = {a: {b: round(len(ids[a] & ids[b]) / len(ids[a] | ids[b]), 3)
                if ids[a] | ids[b] else 1.0 for b in ids} for a in ids}
    return {"results": out, "jaccard": jacc}


# ---- placement strategy lab (§11) ----
from core.placement import STRATEGIES as PLACEMENT_STRATEGIES


@app.get("/api/strategies")
def strategies_list() -> Dict[str, Any]:
    return {"strategies": list(PLACEMENT_STRATEGIES),
            "current": (_ctx.cfg.get("topology", {}) or {})
                       .get("placement_strategy", "hybrid_mira")}


@app.post("/api/strategies/{name}/apply")
def strategies_apply(name: str) -> Dict[str, Any]:
    if name not in PLACEMENT_STRATEGIES:
        raise HTTPException(400, f"unknown strategy {name!r}; "
                                 f"choose from {list(PLACEMENT_STRATEGIES)}")
    info = ws().replace_all(strategy=name)
    return {"applied": name, "info": info}


@app.post("/api/strategies/sweep")
def strategies_sweep() -> Dict[str, Any]:
    """Benchmark every placement strategy on the corpus QA set: place →
    measure → restore the default. Real computation, no cached results."""
    from evaluation.datasets import corpus_dataset
    from evaluation.report import comparison_table
    w = ws()
    current = (_ctx.cfg.get("topology", {}) or {})
    if isinstance(current, dict):
        current = current.get("placement_strategy", "hybrid_mira")
    records = corpus_dataset(w, limit=25)
    if not records:
        raise HTTPException(400, "ingest documents first")
    from evaluation.benchmark import mira_retrieve_fn, run_system
    table, best_name, best_mrr = [], current, -1.0
    for name in PLACEMENT_STRATEGIES:
        w.replace_all(strategy=name)
        res = run_system("full_mira", mira_retrieve_fn(w, active_components=None, k=8),
                         w.embeddings, records, k=8)
        row = comparison_table({"full_mira": res})
        mrr = row[0].get("mrr") or 0.0
        if mrr > best_mrr:
            best_name, best_mrr = name, mrr
        table.append({"system": name, **{k: row[0].get(k) for k in
                      ("retrieval_recall", "mrr", "latency_ms")}})
    w.replace_all(strategy=current)  # restore the user's default
    return {"table": table, "best": best_name, "restored": current}


# ---- live web search (consent-gated, spec §40/§41) ----
from tools.websearch import available_backends, fetch_page_text, search as web_search_fn


@app.get("/api/websearch/status")
def websearch_status() -> Dict[str, Any]:
    return {"backends": available_backends(),
            "enabled": bool((_ctx.cfg.get("web_search", {}) or {}).get("enabled"))}


@app.post("/api/websearch")
def websearch(payload: Dict[str, Any]) -> Dict[str, Any]:
    q = (payload.get("query") or "").strip()
    if not q:
        raise HTTPException(400, "query required")
    # explicit user action in this request counts as consent
    out = web_search_fn(q, _ctx.cfg, allow_web=True,
                        backend=payload.get("backend", "auto"))
    return out


@app.post("/api/websearch/ingest")
def websearch_ingest(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch a search result page and turn it into mandala memories with
    URL provenance. Text is treated as untrusted data (spec §41)."""
    url = (payload.get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "valid http(s) url required")
    try:
        text = fetch_page_text(url, max_chars=20000)
    except Exception as exc:
        raise HTTPException(502, f"fetch failed: {exc}")
    if len(text) < 200:
        raise HTTPException(422, "page yielded too little text to ingest")
    stats = ws().ingest_text(text, title=payload.get("title") or url)
    return {"ingested": stats, "url": url, "chars": len(text)}


# ---- benchmarks ----
@app.post("/api/benchmark")
def benchmark(payload: Dict[str, Any]) -> Dict[str, Any]:
    dataset_path = payload.get("dataset_path") or ""
    fmt = payload.get("format", "custom")
    limit = int(payload.get("limit", 20))
    k = int(payload.get("k", 8))
    with_judge = bool(payload.get("judge", False))
    save = bool(payload.get("save", True))
    if dataset_path and not os.path.exists(dataset_path):
        raise HTTPException(400, f"dataset not found: {dataset_path}")
    w = ws()
    from evaluation.datasets import corpus_dataset, load_any
    if dataset_path:
        records = load_any(dataset_path, fmt, ws=w, limit=limit)
    else:  # no file given: self-supervised sanity set from the ingested corpus
        records = corpus_dataset(w, limit=limit)
        fmt = "corpus"
    if not records:
        raise HTTPException(400, "no usable questions — ingest documents first, "
                                 "or point dataset_path at a JSONL/JSON dataset")
    from evaluation.ablation import run_all_systems
    from evaluation.benchmark import answer_retrieve_fn, run_system
    results = run_all_systems(w, records, k=k,
                              include_baselines=bool(payload.get("baselines", True)),
                              include_ablations=bool(payload.get("ablations", True)))
    judge_summary = None
    if with_judge and _ctx.has_llm:
        ans_res = run_system("full_mira_answered", answer_retrieve_fn(w, k=k),
                             w.embeddings, records, k=k)
        results["full_mira_answered"] = ans_res
        from evaluation.judge import LLMJudge, judge_records
        judge_summary = judge_records(LLMJudge(_ctx.llm), ans_res["rows"],
                                      answers={i: r.get("evidence_text", "")
                                               for i, r in enumerate(ans_res["rows"])})
    exp_id = None
    if save:
        from evaluation.report import save_experiment
        exp_id = save_experiment(
            payload.get("name", "web-run"),
            {"dataset": dataset_path or "<corpus-generated>", "format": fmt,
             "k": k, "limit": limit},
            results, store=w.store, hw=_ctx.hw)
    from evaluation.report import comparison_table
    return {"table": comparison_table(results), "judge": judge_summary,
            "experiment_id": exp_id, "n_questions": len(records)}


@app.get("/api/experiments")
def experiments() -> Dict[str, Any]:
    exps = ws().store.list_experiments()
    return {"experiments": [{"id": e["id"], "name": e["name"],
                             "created_at": e["created_at"]} for e in exps]}


if os.path.isdir(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
