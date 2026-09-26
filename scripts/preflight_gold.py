"""Pre-flight: verify gold-id resolution on bench workspace (title prefix fix)."""
import json, os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import core.workspace as cw
cw.DATA_DIR = os.path.join(ROOT, "data_bench")
import config.auto_config as ac
ac.DATA_DIR = cw.DATA_DIR

from config.auto_config import build_context
from core.workspace import Workspace
from evaluation.datasets import resolve_titles_to_ids

with open(os.path.join(cw.DATA_DIR, "benchmarks", "musique_bench.json"), encoding="utf-8") as f:
    bench = json.load(f)
qs = bench["questions"] if isinstance(bench, dict) and "questions" in bench else bench
print(f"bench records: {len(qs)}")

ctx = build_context()
ws = Workspace(embeddings=ctx.embeddings, llm=None, config=ctx.cfg)
print(f"nodes: {len(ws.frame.nodes)}, vectors: {ws.vs.size() if ws.vs is not None else 0}")

n_hit = 0
per_q_counts = []
for q in qs[:20]:
    ids = resolve_titles_to_ids(ws, q["supporting_titles"])
    per_q_counts.append(len(ids))
    if ids:
        n_hit += 1
print(f"gold resolved for {n_hit}/20 sample questions; per-q node counts: {per_q_counts}")

# full sweep without loading chunks repeatedly: count over all questions
total_hit = sum(1 for q in qs if resolve_titles_to_ids(ws, q["supporting_titles"]))
print(f"gold resolved for {total_hit}/{len(qs)} questions (full sweep)")
assert total_hit > len(qs) * 0.9, "gold resolution below 90% — investigate before bench run"
print("PREFLIGHT OK")
