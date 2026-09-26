"""Final acceptance test (spec §50).

End-to-end against the real Workspace on the real machine:
ingest → memories → graph → vector index → mandala placement →
answer with path + sources → baselines vs MIRA → experiment export.

Run:  .venv/Scripts/python.exe -m tests.test_acceptance
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TEST_TEXT = """The Eiffel Tower is a wrought-iron lattice tower in Paris, France.
It was designed by Gustave Eiffel and completed in 1889 for the World's Fair.
The tower stands 330 metres tall and was the tallest man-made structure in the
world until 1930.

The Louvre Museum is the world's largest art museum, located in Paris on the
right bank of the Seine. Its collection includes the Mona Lisa, painted by
Leonardo da Vinci in the early sixteenth century. The museum opened in 1793.

Gustave Eiffel also engineered the internal frame of the Statue of Liberty,
which was a gift from France to the United States, dedicated in 1886 in New
York Harbor. The statue was designed by Frederic Auguste Bartholdi.

Berlin is the capital of Germany and its largest city. The Brandenburg Gate,
built between 1788 and 1791, is one of Berlin's most famous landmarks."""


def main() -> None:
    from config.auto_config import build_context
    from core.workspace import Workspace
    from evaluation.ablation import run_all_systems
    from evaluation.report import comparison_table

    # 1-3. hardware detection + model selection (§50 steps 1-3)
    ctx = build_context()
    ws = Workspace(embeddings=ctx.embeddings, llm=ctx.llm, config=ctx.cfg)
    print(f"[1] hardware: {ctx.hw.summary()}")
    print(f"[2] tier={ctx.hw.tier()} mode={ctx.rc.performance_mode} "
          f"embed={ctx.rc.embedding_model}/{ctx.rc.embedding_device} "
          f"llm={'yes' if ctx.has_llm else 'extractive-fallback'}")

    # 4-9. ingest → chunks → memories → graph → vector index → mandala (§50 4-9)
    for d in ws.store.list_documents():
        if d["title"] == "acceptance_test.txt":
            ws.delete_document(d["id"])  # re-run friendly: drop previous run
    stats = ws.ingest_text(TEST_TEXT, title="acceptance_test.txt")
    assert stats["n_chunks"] >= 2, "chunker must split the text"
    assert stats["n_nodes"] > 0 and stats["n_edges"] > 0, "memories must be created"
    s = ws.stats()
    assert s["vectors"] == s["nodes"], "every node must be indexed in FAISS"
    placed = [n for n in ws.frame.nodes.values()
              if n.ring is not None and n.sector]
    assert placed, "mandala placement must assign rings and sectors"
    print(f"[3] ingest: {stats['n_chunks']} chunks -> {stats['n_nodes']} nodes, "
          f"{stats['n_edges']} edges; {len(placed)} nodes placed on the mandala")

    # 10. mandala figure builds without error
    from visualization.mandala_view import figure
    fig = figure(ws.frame)
    assert len(fig.data) > 2, "mandala figure must have traces"
    print(f"[4] mandala figure: {len(fig.data)} traces")

    # 11-14. answer with path + sources + metrics (§50 11-14)
    ans = ws.ask("Who designed the Eiffel Tower and what else did he engineer?",
                 system_name="acceptance")
    assert ans.memories, "answer must include retrieved memories"
    assert ans.metrics["latency_ms"] > 0
    assert ans.metrics["context_tokens"] > 0
    print(f"[5] answer ({ans.mode}): {ans.text[:90]}...")
    print(f"    path: {ans.path_labels or '(single-hop)'}")
    print(f"    sources: {ans.sources}")
    print(f"    metrics: {ans.metrics}")

    # 15-19. baselines vs MIRA (§50 15-19)
    with tempfile.TemporaryDirectory() as td:
        ds_path = Path(td) / "ds.json"
        ds_path.write_text(json.dumps([
            {"question": "Who designed the Eiffel Tower?",
             "answer": "Gustave Eiffel designed the Eiffel Tower.",
             "supporting_ids": [n.id for n in ws.frame.nodes.values()
                                if "Eiffel Tower" in n.concept][:1]},
            {"question": "Where is the Louvre Museum?",
             "answer": "The Louvre Museum is located in Paris.",
             "supporting_ids": [n.id for n in ws.frame.nodes.values()
                                if "Louvre" in n.concept][:1]},
        ]), encoding="utf-8")
        from evaluation.benchmark import load_dataset
        records = load_dataset(str(ds_path))
        results = run_all_systems(ws, records, k=5,
                                  include_baselines=True, include_ablations=True)
        table = comparison_table(results)
        assert len(table) == 15, "3 baselines + 12 MIRA configurations"
        print(f"[6] benchmark: {len(table)} systems compared")
        for row in table[:4]:
            print(f"    {row['system']:>18}: recall={row['retrieval_recall']} "
                  f"mrr={row['mrr']} ctx={row['context_tokens']}")

        # 20. export experiment (§50 step 20)
        import evaluation.report as report
        real_dir = report.EXPERIMENTS_DIR
        report.EXPERIMENTS_DIR = Path("experiments")
        try:
            exp_id = report.save_experiment(
                "acceptance-run",
                {"dataset": "synthetic-acceptance", "k": 5,
                 "weights": ws.config.get("retrieval_score", {})},
                results, store=ws.store, hw=ctx.hw)
        finally:
            report.EXPERIMENTS_DIR = real_dir
        d = Path("experiments") / exp_id
        assert (d / "summary.md").exists() and (d / "results.csv").exists()
        print(f"[7] experiment exported: experiments/{exp_id}/ "
              "(config.json, results.json, results.csv, summary.md)")

    ws.close()
    print("ACCEPTANCE TEST PASS")


if __name__ == "__main__":
    main()
