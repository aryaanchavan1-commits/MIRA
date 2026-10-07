"""MIRA-NCM tests (spec §49): membership, overlap kernel, versioning,
default-off invariance, and the enabled-path through MIRARetriever."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core import ncm
from core.memory import MemoryFrame, MemoryNode, MemoryType
from core.placement import place
from core.retrieval import MIRARetriever
from models.embeddings import init_embeddings
from storage.graph_store import GraphStore
from storage.vector_store import VectorStore

FAILURES = []


def check(name, fn):
    try:
        fn()
        print(f"PASS {name}")
    except Exception as exc:  # noqa: BLE001 — test harness must not die early
        FAILURES.append((name, exc))
        print(f"FAIL {name}: {exc}")


# ---------------------------------------------------------------- helpers
TOPICS = ["tomato pasta basil recipe", "telescope galaxy orbit astronomy",
          "piano chord melody concert", "rivers forest hiking trail"]


def constellation_world(n_clusters: int = 4, per: int = 3) -> MemoryFrame:
    """Deterministic world: distinct topic vocabularies per cluster, so any
    embedder (including the hash stub) groups them. Embeddings are assigned
    by the caller via the stub embedder — 384-dim everywhere."""
    frame = MemoryFrame()
    for c in range(n_clusters):
        for i in range(per):
            frame.add_node(MemoryNode(
                id=f"n{c}_{i}", concept=f"topic{c}",
                summary=f"{TOPICS[c]} note {i}",
                raw_text=f"{TOPICS[c]} note {i}"))
    return frame


BASE_CONFIG = {"retrieval": {"candidate_k": 12, "final_k": 8},
               "topology": {"max_rings": 5}}


# ---------------------------------------------------------------- units
def _embed(frame: MemoryFrame):
    emb = init_embeddings("stub")
    for n, v in zip(frame.nodes.values(),
                    emb.encode([n.summary for n in frame.nodes.values()])):
        n.embedding = v
    return emb


def test_membership_topk_and_normalization():
    frame = constellation_world()
    _embed(frame)
    idx = ncm.ConstellationIndex(top_k=2)
    assert idx.fit(frame), idx._fit_info
    assert len(idx.members) == len(frame.nodes)
    for nid, w in idx.weight_of.items():
        assert 1 <= len(idx.members[nid]) <= 2, (nid, idx.members[nid])
        assert abs(sum(w) - 1.0) < 1e-6, (nid, w)
    # K sweep changes the assignment — the H2 ablation axis is real
    idx1 = ncm.ConstellationIndex(top_k=1)
    idx1.fit(frame)
    assert all(len(v) == 1 for v in idx1.members.values())


def test_overlap_kernel_separates_clusters():
    frame = constellation_world()
    emb = _embed(frame)
    idx = ncm.ConstellationIndex(top_k=2)
    idx.fit(frame)
    q = np.asarray(emb.encode([TOPICS[1] + " note 0"])[0])
    q_ids, q_w = idx.query_membership(q)
    same = idx.node_score("n1_0", q_ids, q_w)
    far = idx.node_score("n3_2", q_ids, q_w)
    assert 0.0 <= far < same <= 1.0, (same, far)
    assert same > 0.4


def test_version_chain_semantics():
    vc = ncm.VersionChain(max_depth=4)
    vc.update("ide", "User likes Python", source="chat1")
    v2 = vc.update("ide", "User prefers Rust for systems", source="chat2")
    hist = vc.history("ide")
    assert len(hist) == 2
    assert hist[-1]["hash"] == v2["hash"]
    assert hist[-1]["supersedes"] == hist[0]["hash"]      # supersession chain
    assert vc.update("ide", "User prefers Rust for systems")["n"] == 2  # no dup
    rb = vc.rollback("ide", 1)                             # append-only rollback
    assert rb["content"] == "User likes Python"
    assert len(vc.history("ide")) == 3
    assert vc.rollback("ide", 99) is None
    for i in range(6):                                     # MAX_DEPTH cap
        vc.update("lang", f"content {i}")
    assert len(vc.history("lang")) == 4
    assert vc.current("lang")["content"] == "content 5"    # newest survives


def test_fit_degrades_below_min_nodes():
    frame = MemoryFrame()
    for i in range(3):
        frame.add_node(MemoryNode(id=f"m{i}", concept=f"m{i}",
                                  summary=f"s{i}", raw_text=f"s{i}",
                                  embedding=np.ones(8, dtype=np.float32)))
    idx = ncm.ConstellationIndex()
    assert not idx.fit(frame)
    assert ncm.constellation_score(idx, "m0", np.ones(8)) == 0.0
    assert ncm.constellation_score(None, "m0", np.ones(8)) == 0.0


# ------------------------------------------------- retrieval integration
def build_retriever_world(config):
    frame = constellation_world(n_clusters=4, per=3)
    place(frame, "hybrid_mira")
    emb = init_embeddings("stub")
    vecs = emb.encode([n.summary for n in frame.nodes.values()])
    for n, v in zip(frame.nodes.values(), vecs):
        n.embedding = v
    vs = VectorStore(dim=vecs.shape[1])
    vs.add(vecs, [{"node_id": n.id} for n in frame.nodes.values()])
    gs = GraphStore()
    gs.build_from([n.to_row() for n in frame.nodes.values()], [])
    return frame, MIRARetriever(frame, vs, gs, config), emb


def test_default_off_is_invariant():
    frame, r, emb = build_retriever_world(dict(BASE_CONFIG))
    assert r.constellation_idx is None
    assert r.weights["constellation"] == 0.0
    q = np.asarray(emb.encode([TOPICS[0] + " note 1"])[0])
    full = r.retrieve(TOPICS[0] + " note 1", q)                       # "all"
    no_ncm = r.retrieve(TOPICS[0] + " note 1", q,
                        active_components=("semantic", "structural", "radial",
                                           "graph", "importance", "confidence",
                                           "recency", "path", "activation",
                                           "stability"))
    assert full.items[0].components["constellation"] == 0.0, \
        "constellation term must be 0 when NCM is disabled"
    assert [i.node.id for i in full.items] == [i.node.id for i in no_ncm.items], \
        "ranking must not change when NCM is disabled"
    # scores: the recency term reads the wall clock, so two retrieves can
    # differ in the 9th decimal even with NCM off — compare within that jitter.
    assert all(abs(a.score - b.score) < 1e-8
               for a, b in zip(full.items, no_ncm.items)), \
        "NCM disabled must not move scores beyond clock jitter"


def test_enabled_changes_ranking_via_constellation():
    cfg = dict(BASE_CONFIG)
    cfg["ncm"] = {"enabled": True, "top_k": 2, "max_constellations": 8}
    frame, r, emb = build_retriever_world(cfg)
    assert r.constellation_idx is not None
    assert r.weights["constellation"] > 0
    q = np.asarray(emb.encode([TOPICS[2] + " note 0"])[0])
    res = r.retrieve(TOPICS[2] + " note 0", q,
                     active_components=("constellation",))   # term alone
    top = res.items[0]
    assert top.components["constellation"] > 0.0
    assert top.node.id.startswith("n2_"), top.node.id  # query's own cluster
    full = r.retrieve(TOPICS[2] + " note 0", q)             # all components
    assert full.items[0].components["constellation"] > 0.0
    # metadata round-trip carries the membership (spec §4 schema)
    meta = top.node.metadata.get(ncm.META_KEY)
    assert meta and len(meta["ids"]) == len(meta["weights"]) >= 1


def run_all() -> int:
    check("membership_topk_and_normalization", test_membership_topk_and_normalization)
    check("overlap_kernel_separates_clusters", test_overlap_kernel_separates_clusters)
    check("version_chain_semantics", test_version_chain_semantics)
    check("fit_degrades_below_min_nodes", test_fit_degrades_below_min_nodes)
    check("default_off_is_invariant", test_default_off_is_invariant)
    check("enabled_changes_ranking_via_constellation",
          test_enabled_changes_ranking_via_constellation)
    print(f"{6 - len(FAILURES)}/6 passed")
    if FAILURES:
        for name, exc in FAILURES:
            print(f"  {name}: {exc}")
        return 1
    print("NCM TESTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_all())
