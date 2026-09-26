"""MIRA web API — FastAPI wrapper around the research core.

Same Workspace core the Streamlit UI and tests use. The static frontend in
web/ is served at "/" so the whole product is one local process. Run:

    .venv/Scripts/python.exe -m uvicorn server:app --port 8000
"""
from __future__ import annotations

import logging
import os
import threading
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from config.auto_config import DATA_DIR, build_context, runtime_summary
from core.affect import AFFECT_DISCLOSURE
from core.placement import STRATEGIES as PLACEMENT_STRATEGIES
from core.retrieval import ablation_configs
from core.workspace import Workspace
from tools.websearch import (
    available_backends,
    fetch_page_text,
    search as web_search_fn,
    web_allowed,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mira.server")

app = FastAPI(title="MIRA", docs_url="/api/docs")

_ctx = build_context()
_ws: Optional[Workspace] = None
_ws_lock = threading.RLock()  # uvicorn serves sync handlers from a threadpool


def ws() -> Workspace:
    global _ws
    if _ws is None:
        with _ws_lock:  # double-checked: concurrent first requests must build once
            if _ws is None:
                _ws = Workspace(embeddings=_ctx.embeddings, llm=_ctx.llm,
                                config=_ctx.cfg)
    return _ws


def _payload_bool(payload: Dict[str, Any], key: str, default=None):
    """Read an optional consent flag without accepting truthy strings/numbers."""
    if key not in payload:
        return default
    value = payload.get(key)
    if type(value) is not bool:
        raise HTTPException(400, f"{key} must be boolean")
    return value


def _payload_text(payload: Dict[str, Any], key: str, max_length: int,
                  required: bool = False) -> str:
    value = payload.get(key, "")
    if not isinstance(value, str):
        raise HTTPException(400, f"{key} must be a string")
    value = value.strip()
    if required and not value:
        raise HTTPException(400, f"{key} required")
    if len(value) > max_length:
        raise HTTPException(400, f"{key} is too long")
    return value


def _payload_int(payload: Dict[str, Any], key: str, default: int,
                 minimum: int, maximum: int) -> int:
    value = payload.get(key, default)
    if type(value) is not int:
        raise HTTPException(400, f"{key} must be an integer")
    if not minimum <= value <= maximum:
        raise HTTPException(400, f"{key} must be between {minimum} and {maximum}")
    return value


_MAX_UPLOAD_BYTES = 32 * 1024 * 1024
_WEB_BACKENDS = {"auto", "agent-reach", "opencli", "duckduckgo"}
_DATASET_FORMATS = {"custom", "hotpotqa", "2wiki", "musique", "corpus"}


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
    w = ws()
    with w.lock:
        s = w.stats()
        out = runtime_summary(_ctx)
        out["workspace"] = s
        out["node_count"] = len(w.frame.nodes)
    return out


# ---- chat ----
@app.post("/api/chat")
def chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    q = _payload_text(payload, "question", 4000, required=True)
    comp = payload.get("components")
    if comp is not None and not isinstance(comp, str):
        raise HTTPException(400, "components must be a string")
    if comp == "all":
        comp = None
    elif comp is not None:
        allowed = set(ablation_configs().keys())
        if comp not in allowed:
            raise HTTPException(400, f"unknown ablation set {comp!r}")
        comp = ablation_configs()[comp]
    allow_web = _payload_bool(payload, "allow_web")
    try:
        w = ws()
        with w.lock:
            ans = w.ask(q, active_components=comp, system_name="web",
                        allow_web=allow_web)
    except ValueError as exc:  # bad component names from API callers
        raise HTTPException(400, str(exc))
    return {
        "answer": ans.text, "mode": ans.mode,
        "agent_mode": getattr(ans, "agent_mode", "memory"),
        "memories": ans.memories,
        "selected_evidence_ids": list(getattr(ans, "selected_evidence_ids", [])),
        "paths": ans.path_labels, "sources": ans.sources,
        "source_refs": list(getattr(ans, "source_refs", [])),
        "metrics": ans.metrics, "note": ans.confidence_note,
        "affect_snapshot": dict(getattr(ans, "affect_snapshot", {})),
        "affect": dict(getattr(ans, "affect_snapshot", {})),
    }


# ---- simulated affect (algorithmic bookkeeping only) ----
def _affect_response() -> Dict[str, Any]:
    w = ws()
    with w.lock:
        state = w.affect
        snapshot = dict(state.snapshot())
        snapshot["disclosure"] = AFFECT_DISCLOSURE
        return snapshot


@app.get("/api/affect")
def affect_state() -> Dict[str, Any]:
    return _affect_response()


def _apply_affect_feedback(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Apply bounded user-supplied deltas; never infer affect from text."""
    if not isinstance(payload, dict) or not payload:
        raise HTTPException(400, "feedback object required")
    wrapper_keys = {"feedback", "update"}
    wrappers = wrapper_keys & set(payload)
    if len(wrappers) > 1:
        raise HTTPException(400, "provide only one of feedback or update")
    if wrappers:
        unknown_outer = sorted(set(payload) - wrapper_keys)
        if unknown_outer:
            raise HTTPException(400, f"unknown affect feedback field(s): {', '.join(unknown_outer)}")
        raw = payload.get("feedback", payload.get("update"))
    else:
        raw = payload
    if not isinstance(raw, dict) or not raw:
        raise HTTPException(400, "feedback object required")
    allowed = {"valence", "arousal", "confidence", "stress", "label", "reason"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise HTTPException(400, f"unknown affect feedback field(s): {', '.join(unknown)}")
    w = ws()
    with w.lock:
        try:
            w.affect.apply_feedback(**raw)
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return _affect_response()


@app.post("/api/affect/feedback")
def affect_feedback(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _apply_affect_feedback(payload)


# Accept the short form too; the original chat/document endpoints remain unchanged.
@app.post("/api/affect")
def affect_update(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _apply_affect_feedback(payload)


# ---- mandala ----
@app.get("/api/mandala")
def mandala() -> Dict[str, Any]:
    w = ws()
    with w.lock:
        frame = w.frame
        nodes = [{
            "id": n.id, "concept": n.concept, "type": n.memory_type.value,
            "ring": n.ring, "sector": n.sector,
            "radial": n.radial_distance, "importance": round(n.importance, 3),
            "confidence": round(n.confidence, 3), "summary": (n.summary or "")[:200],
            "n_sources": len(n.source_ids),
        } for n in sorted(frame.nodes.values(), key=lambda node: node.id)]
        edges = [{"s": e.source_id, "t": e.target_id, "r": e.relation_type,
                  "w": e.weight} for e in frame.edges]
        topology = w.config.get("topology", {}) or {}
        if not isinstance(topology, dict):
            topology = {}
        return {"nodes": nodes, "edges": edges,
                "max_rings": topology.get("max_rings", 5)}


@app.get("/api/node/{node_id}")
def node_detail(node_id: str) -> Dict[str, Any]:
    with ws().lock:
        return _node_detail(node_id)


def _node_detail(node_id: str) -> Dict[str, Any]:
    w = ws()
    frame = w.frame
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
    w = ws()
    with w.lock:
        return {"documents": w.store.list_documents()}


@app.post("/api/documents")
def upload_document(file: UploadFile = File(...), title: str = Form("")) -> Dict[str, Any]:
    filename = os.path.basename(file.filename or "")
    if not filename:
        raise HTTPException(400, "file name required")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in {".pdf", ".txt", ".md", ".json", ".csv", ".docx"}:
        raise HTTPException(400, f"unsupported file type {ext!r}")
    if len(title or "") > 300:
        raise HTTPException(400, "title is too long")
    upload_dir = os.path.join(DATA_DIR, "uploads", uuid.uuid4().hex)
    os.makedirs(upload_dir, exist_ok=True)
    dest = os.path.join(upload_dir, filename)
    written = 0
    try:
        with open(dest, "wb") as fh:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > _MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "file exceeds 32 MB limit")
                fh.write(chunk)
    except HTTPException:
        try:
            os.remove(dest)
            os.rmdir(upload_dir)
        except FileNotFoundError:
            pass
        raise
    try:
        w = ws()
        with w.lock:
            stats = w.ingest_file(dest, title=title or filename)
    except ValueError as exc:
        try:
            os.remove(dest)
            os.rmdir(upload_dir)
        except FileNotFoundError:
            pass
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        try:
            os.remove(dest)
            os.rmdir(upload_dir)
        except FileNotFoundError:
            pass
        raise HTTPException(500, "document ingestion failed") from exc
    return stats


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str) -> Dict[str, Any]:
    w = ws()
    with w.lock:
        removed = w.delete_document(document_id)
    return {"deleted": document_id, "nodes_removed": removed}


# ---- research lab ----
@app.post("/api/lab")
def lab(payload: Dict[str, Any]) -> Dict[str, Any]:
    w = ws()
    with w.lock:
        return _lab(payload, w)


def _lab(payload: Dict[str, Any], w: Workspace) -> Dict[str, Any]:
    q = _payload_text(payload, "question", 4000, required=True)
    raw_sets = payload.get("sets") or ["full_mira", "vector_only", "graph_only", "radial_only"]
    if not isinstance(raw_sets, list) or not all(isinstance(name, str) for name in raw_sets):
        raise HTTPException(400, "sets must be a list of strings")
    sets = raw_sets[:20]
    valid_sets = set(ablation_configs())
    unknown_sets = sorted(set(sets) - valid_sets)
    if unknown_sets:
        raise HTTPException(400, f"unknown ablation set(s): {', '.join(unknown_sets)}")
    k = _payload_int(payload, "k", 8, 1, 50)
    from core.answer import AnswerPipeline
    pipe = AnswerPipeline(w.frame, w.vs, w.gs, w.embeddings, llm=w.llm,
                          config=w.config, doc_titles=w._doc_titles(),
                          operation_lock=w.lock)
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
@app.get("/api/strategies")
def strategies_list() -> Dict[str, Any]:
    w = ws()
    with w.lock:
        topology = w.config.get("topology", {}) or {}
        if not isinstance(topology, dict):
            topology = {}
        return {"strategies": list(PLACEMENT_STRATEGIES),
                "current": topology.get("placement_strategy", "hybrid_mira")}


@app.post("/api/strategies/{name}/apply")
def strategies_apply(name: str) -> Dict[str, Any]:
    if name not in PLACEMENT_STRATEGIES:
        raise HTTPException(400, f"unknown strategy {name!r}; "
                                 f"choose from {list(PLACEMENT_STRATEGIES)}")
    w = ws()
    with w.lock:
        info = w.replace_all(strategy=name)
        topology = w.config.setdefault("topology", {})
        if not isinstance(topology, dict):
            topology = {}
            w.config["topology"] = topology
        topology["placement_strategy"] = name
    return {"applied": name, "info": info}


@app.post("/api/strategies/sweep")
def strategies_sweep() -> Dict[str, Any]:
    """Benchmark every placement strategy on the corpus QA set: place →
    measure → restore the default. Real computation, no cached results."""
    from evaluation.datasets import corpus_dataset
    from evaluation.report import comparison_table
    w = ws()
    with w.lock:
        topology = w.config.get("topology", {}) or {}
        raw_current = (topology.get("placement_strategy")
                       if isinstance(topology, dict) else None)
        current = (raw_current if isinstance(raw_current, str)
                   and raw_current in PLACEMENT_STRATEGIES else "hybrid_mira")
        table, best_name, best_mrr = [], current, -1.0
        try:
            records = corpus_dataset(w, limit=25)
            if not records:
                raise HTTPException(400, "ingest documents first")
            from evaluation.benchmark import mira_retrieve_fn, run_system
            for name in PLACEMENT_STRATEGIES:
                w.replace_all(strategy=name)
                res = run_system(
                    "full_mira", mira_retrieve_fn(w, active_components=None, k=8),
                    w.embeddings, records, k=8,
                )
                row = comparison_table({"full_mira": res})
                mrr = row[0].get("mrr") or 0.0
                if mrr > best_mrr:
                    best_name, best_mrr = name, mrr
                table.append({"system": name, **{k: row[0].get(k) for k in
                              ("retrieval_recall", "mrr", "latency_ms")}})
        finally:
            # A failed benchmark must not leave the workspace in the last
            # experimental strategy.  replace_all also restores live config.
            w.replace_all(strategy=current)
    return {"table": table, "best": best_name, "restored": current}


# ---- live web search (consent-gated, spec §40/§41) ----
@app.get("/api/websearch/status")
def websearch_status() -> Dict[str, Any]:
    try:
        enabled = web_allowed(_ctx.cfg, None)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"backends": available_backends(), "enabled": enabled}


@app.post("/api/websearch")
def websearch(payload: Dict[str, Any]) -> Dict[str, Any]:
    q = _payload_text(payload, "query", 2000, required=True)
    backend = payload.get("backend", "auto")
    if not isinstance(backend, str) or backend not in _WEB_BACKENDS:
        raise HTTPException(400, f"backend must be one of {sorted(_WEB_BACKENDS)}")
    allow_web = _payload_bool(payload, "allow_web", None)
    try:
        allowed = web_allowed(_ctx.cfg, allow_web)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not allowed:
        return {"backend": None, "results": [],
                "note": "web search disabled (explicit consent and offline=false required)"}
    return web_search_fn(q, _ctx.cfg, allow_web=True, backend=backend)


@app.post("/api/websearch/ingest")
def websearch_ingest(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch a search result page and turn it into mandala memories with
    URL provenance. Text is treated as untrusted data (spec §41)."""
    url = _payload_text(payload, "url", 2048, required=True)
    title = _payload_text(payload, "title", 300)
    allow_web = _payload_bool(payload, "allow_web", None)
    try:
        allowed = web_allowed(_ctx.cfg, allow_web)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not allowed:
        raise HTTPException(403, "web access disabled in offline mode or without consent")
    try:
        text = fetch_page_text(url, max_chars=20000,
                               config=_ctx.cfg, allow_web=allowed)
    except Exception as exc:
        raise HTTPException(502, f"fetch failed: {exc}")
    if len(text) < 200:
        raise HTTPException(422, "page yielded too little text to ingest")
    w = ws()
    with w.lock:
        stats = w.ingest_text(text, title=title or url, source_path=url)
    return {"ingested": stats, "url": url, "chars": len(text)}


# ---- benchmarks ----
@app.post("/api/benchmark")
def benchmark(payload: Dict[str, Any]) -> Dict[str, Any]:
    w = ws()
    with w.lock:
        return _benchmark(payload, w)


def _safe_dataset_path(value: str) -> str:
    if not value:
        return ""
    root = os.path.realpath(os.path.join(DATA_DIR, "datasets"))
    candidate = os.path.realpath(value)
    try:
        inside = os.path.commonpath((root, candidate)) == root
    except ValueError:
        inside = False
    if not inside:
        raise HTTPException(400, "dataset_path must be inside data/datasets")
    if not os.path.isfile(candidate):
        raise HTTPException(400, f"dataset not found: {value}")
    return candidate


def _benchmark(payload: Dict[str, Any], w: Workspace) -> Dict[str, Any]:
    dataset_path = _safe_dataset_path(
        _payload_text(payload, "dataset_path", 1024))
    fmt = payload.get("format", "custom")
    if not isinstance(fmt, str) or fmt not in _DATASET_FORMATS:
        raise HTTPException(400, f"format must be one of {sorted(_DATASET_FORMATS)}")
    limit = _payload_int(payload, "limit", 20, 1, 500)
    k = _payload_int(payload, "k", 8, 1, 50)
    with_judge = _payload_bool(payload, "judge", False)
    save = _payload_bool(payload, "save", True)
    include_baselines = _payload_bool(payload, "baselines", True)
    include_ablations = _payload_bool(payload, "ablations", True)
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
                              include_baselines=include_baselines,
                              include_ablations=include_ablations)
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
        name = _payload_text(payload, "name", 120) or "web-run"
        exp_id = save_experiment(
            name,
            {"dataset": dataset_path or "<corpus-generated>", "format": fmt,
             "k": k, "limit": limit},
            results, store=w.store, hw=_ctx.hw)
    from evaluation.report import comparison_table
    return {"table": comparison_table(results), "judge": judge_summary,
            "experiment_id": exp_id, "n_questions": len(records)}


@app.get("/api/experiments")
def experiments() -> Dict[str, Any]:
    w = ws()
    with w.lock:
        exps = w.store.list_experiments()
        return {"experiments": [{"id": e["id"], "name": e["name"],
                                 "created_at": e["created_at"]} for e in exps]}


if os.path.isdir(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
