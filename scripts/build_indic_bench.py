"""Build IndicQA (hi + mr) retrieval benches from the downloaded raw JSONs.

SQuAD-style extractive QA (single supporting paragraph). Deterministic subset:
120 questions per language; corpus = each question's gold paragraph + 2
distractor paragraphs sampled deterministically (rng seeded per language).

Writes data_indic/benchmarks/indicqa_{hi,mr}_bench.json (same schema as the
other benches). Cross-lingual caveat is documented in the paper: MiniLM-L6 is
English-centric, so absolute numbers are expected to be low; the comparison
between systems under the same embedder is the point of the experiment.
"""
from __future__ import annotations

import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, ".tmp", "datasets")
OUT_DIR = os.path.join(ROOT, "data_indic", "benchmarks")


def build_lang(lang: str, n_questions: int = 120) -> None:
    with open(os.path.join(SRC, f"indicqa.{lang}.json"), encoding="utf-8") as fh:
        data = json.load(fh)["data"]

    rng = random.Random(99 + hash(lang) % 1000)
    paragraphs: dict = {}
    for art in data:
        for p in art["paragraphs"]:
            key = p["context"].strip()
            if key not in paragraphs:  # duplicate contexts share the same qas
                paragraphs[key] = p["qas"]

    pool = list(paragraphs.items())
    rng.shuffle(pool)
    questions, used_titles = [], []
    for ctx, qas_list in pool:
        qas = [q for q in qas_list if q.get("answers")]
        if not qas:
            continue
        rng.shuffle(qas)
        for q in qas:
            questions.append({"id": q["id"], "question": q["question"],
                              "answer": q["answers"][0]["text"],
                              "answer_aliases": [], "hops": 1,
                              "supporting_titles": [ctx[:60]],
                              "paragraphs": []})
            used_titles.append(ctx[:60])
            break
        if len(questions) >= n_questions:
            break

    # corpus: gold paragraph of each question + 2 deterministic distractors
    distractors = [ctx for ctx, _ in pool if ctx.strip()[:60] not in
                   {p["title"] for q in questions for p in q["paragraphs"]}]
    titles: dict = {}
    key_to_text = {ctx.strip()[:60]: ctx for ctx, _ in pool}
    for q, key in zip(questions, used_titles):
        text = key_to_text[key]
        q["paragraphs"] = [{"title": key, "text": text[:1500],
                            "is_supporting": True}]
        titles[key] = text[:1500]
    # two distractors per question
    for i, q in enumerate(questions):
        for j in range(2):
            dt = distractors[(i * 2 + j) % len(distractors)]
            key = dt.strip()[:60]
            if key not in titles:
                titles[key] = dt[:1500]
            q["paragraphs"].append({"title": key, "text": dt[:1500],
                                    "is_supporting": False})

    out = os.path.join(OUT_DIR, f"indicqa_{lang}_bench.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"dataset": f"indicqa_{lang}", "config": lang,
                   "split": "validation", "questions": questions,
                   "paragraphs": titles}, fh, ensure_ascii=False)
    print(f"{lang}: {len(questions)} questions, corpus {len(titles)} paragraphs "
          f"-> {out}", flush=True)


def main() -> int:
    build_lang("hi")
    build_lang("mr")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
