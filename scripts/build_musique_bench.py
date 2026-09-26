"""Build the real-data benchmark from MuSiQue (Trivedi et al., 2021).

Reads the local musique_ans_v1.0_dev.jsonl, selects multi-hop questions
(>=2 supporting paragraphs, per the dataset's own hops metadata), and emits:
  - data/benchmarks/musique_bench.json : {questions: [...], paragraphs: {...}}
    questions: [{id, question, answer, answer_aliases, hops, supporting_titles}]
    paragraphs: {title: text}   (shared corpus, supporting + distractors)

The corpus is the union of every paragraph shown to the annotators for the
selected questions — the standard MuSiQue distractor setting. Sizes are
reported so the benchmark can state its own footprint honestly.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SRC = "data/benchmarks/musique_ans_dev.jsonl"
OUT = "data/benchmarks/musique_bench.json"


def main() -> int:
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    items = []
    with open(SRC, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    print(f"loaded {len(items)} MuSiQue items")

    # keep answerable multi-hop questions with clean short answers
    seen_q = set()
    selected = []
    for it in items:
        q = (it.get("question") or "").strip()
        a = (it.get("answer") or "").strip()
        if not q or not a or len(a) > 80:
            continue
        if q.lower() in seen_q:
            continue
        sup = [p for p in (it.get("paragraphs") or []) if p.get("is_supporting")]
        if len(sup) < 2:
            continue
        seen_q.add(q.lower())
        selected.append({
            "id": it.get("id"),
            "question": q,
            "answer": a,
            "answer_aliases": list(it.get("answer_aliases") or []),
            "hops": len(sup),
            "supporting_titles": [p.get("title") for p in sup if p.get("title")],
            "paragraphs": [
                {"title": p.get("title"), "text": p.get("paragraph_text"),
                 "is_supporting": bool(p.get("is_supporting"))}
                for p in (it.get("paragraphs") or []) if p.get("title")
            ],
        })
        if len(selected) >= target:
            break
    print(f"selected {len(selected)} multi-hop questions "
          f"(hop distribution: {sorted({s['hops'] for s in selected})})")

    paras: dict = {}
    n_sup = 0
    for s in selected:
        for p in s["paragraphs"]:
            t, txt = p["title"], (p.get("text") or "").strip()
            if t and txt and t not in paras:
                paras[t] = txt
                if p["is_supporting"]:
                    n_sup += 1
    total_chars = sum(len(t) for t in paras.values())
    print(f"shared corpus: {len(paras)} unique paragraphs, {total_chars / 1e6:.2f}M chars")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"source": "MuSiQue v1.0 ans dev (dgslibisey/MuSiQue)",
                   "n_questions": len(selected), "n_paragraphs": len(paras),
                   "corpus_chars": total_chars,
                   "questions": selected, "paragraphs": paras}, fh)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
