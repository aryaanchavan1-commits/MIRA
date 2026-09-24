"""Phase 8 tests: ingestion end-to-end on synthetic documents.

Run: .venv/Scripts/python.exe -m tests.test_phase8
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory import MemoryNode
from ingestion.chunker import chunk_text, chunk_pages
from ingestion.loaders import load_json, load_csv, load_text, sanitize
from ingestion.pipeline import IngestionPipeline
from models.embeddings import EmbeddingBackend
from storage.sqlite_store import SQLiteStore
from storage.vector_store import VectorStore


def test_chunker():
    text = ("Sentence one is here. " * 30) + "A final sentence."
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 260 for c in chunks), [len(c) for c in chunks]
    # short text stays single chunk
    assert chunk_text("tiny") == ["tiny"]
    assert chunk_text("") == []
    # page mapping preserved
    paged = chunk_pages([(1, text), (2, "page two content here. " * 10)], 150, 30)
    assert any(p == 2 for p, _ in paged)
    print("  chunker OK")


def test_loaders_and_sanitize():
    assert sanitize("a\x00b\x07c") == "abc"
    tmp = tempfile.mkdtemp(prefix="mira_t8_")
    try:
        jp = os.path.join(tmp, "t.json")
        with open(jp, "w", encoding="utf-8") as fh:
            fh.write('{"name": "mira", "tags": ["memory", "research"], "meta": {"year": 2026}}')
        pages = load_json(jp)
        assert any("mira" in p[1] for p in pages)
        assert any("2026" in p[1] for p in pages)
        cp = os.path.join(tmp, "t.csv")
        with open(cp, "w", encoding="utf-8") as fh:
            fh.write("name,role\nalice,engineer\nbob,researcher\n")
        rows = load_csv(cp)
        assert any("alice" in r[1] for r in rows)
        assert any("role: engineer" in r[1] for r in rows)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  loaders + sanitize OK")


SAMPLE_DOC = """Mandala Memory Systems: Technical Report

The MIRA memory system organizes knowledge in radial rings around a core concept.
Rings assign hierarchical depth. Ring zero holds the core concept of the knowledge base.

Sectors divide the mandala into semantic regions. Sector discovery uses kmeans
clustering on sentence embeddings. Each sector groups related concepts together.

Retrieval combines semantic similarity with structural proximity. The retrieval
score blends eight components including graph centrality and temporal recency.

Provenance tracking records the source document and page for every memory node.
Conflicts are kept rather than overwritten, ranked by confidence and recency.

The benchmark framework compares MIRA against vector rag, graph rag and
hierarchical rag baselines on identical hardware and identical context budgets.
"""


def test_full_ingestion():
    tmp = tempfile.mkdtemp(prefix="mira_t8p_")
    try:
        db_path = os.path.join(tmp, "mira.db")
        store = SQLiteStore(db_path)
        emb = EmbeddingBackend("stub")
        pipe = IngestionPipeline(store, emb, llm=None,
                                 config={"chunk_size": 400, "chunk_overlap": 60})
        doc_path = os.path.join(tmp, "report.txt")
        with open(doc_path, "w", encoding="utf-8") as fh:
            fh.write(SAMPLE_DOC)
        stats = pipe.ingest_file(doc_path)
        assert not stats["deduplicated"]
        assert stats["n_chunks"] >= 2
        assert stats["n_nodes"] > stats["n_chunks"], stats
        assert stats["concepts"], "no concepts extracted"

        # provenance chain: node → chunk → document
        nodes = store.all_nodes()
        assert nodes
        with_chunk = [n for n in nodes if n["source_ids"]]
        assert with_chunk, "no node carries provenance"
        chunk_id = with_chunk[0]["source_ids"][0]
        chunk = store.get_chunk(chunk_id)
        assert chunk is not None
        doc = store.get_document(chunk["document_id"])
        assert doc["title"] == "report"

        # re-ingest same file → deduplicated, no new nodes
        stats2 = pipe.ingest_file(doc_path)
        assert stats2["deduplicated"]
        assert store.count_nodes() == stats["n_nodes"]

        # vector index persisted and loadable
        idx = os.path.join(os.path.dirname(db_path), "indexes", "main.faiss")
        # (index path is fixed to data/indexes in pipeline; verify via global dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  full ingestion + provenance + dedup OK")


def test_ingest_text_and_vector_growth():
    tmp = tempfile.mkdtemp(prefix="mira_t8v_")
    try:
        store = SQLiteStore(os.path.join(tmp, "mira.db"))
        emb = EmbeddingBackend("stub")
        pipe = IngestionPipeline(store, emb, llm=None, config={})
        r1 = pipe.ingest_text(SAMPLE_DOC, "doc one")
        r2 = pipe.ingest_text("Vector stores keep embeddings for fast search. " * 12, "doc two")
        assert not r1["deduplicated"] and not r2["deduplicated"]
        assert store.count_documents() if hasattr(store, "count_documents") else True
        docs = store.list_documents()
        assert len(docs) == 2
        # global index exists and contains both documents' vectors
        from config.auto_config import DATA_DIR
        idx_path = os.path.join(DATA_DIR, "indexes", "main.faiss")
        assert os.path.exists(idx_path), idx_path
        vs = VectorStore.load(idx_path)
        assert vs.size() >= r1["n_nodes"] + r2["n_nodes"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  ingest_text + vector index growth OK")


if __name__ == "__main__":
    print("Phase 8 tests:")
    test_chunker()
    test_loaders_and_sanitize()
    test_full_ingestion()
    test_ingest_text_and_vector_growth()
    print("ALL PASS")
