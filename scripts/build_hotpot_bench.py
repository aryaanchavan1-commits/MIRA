"""Build a HotpotQA (distractor) benchmark with the same schema as the
MuSiQue bench — via the HF datasets-server REST API (no new deps, no HF hub
client). Deterministic subset: 300 questions, bridge/comparison mix,
2 supporting docs each; corpus = the union of those questions' contexts.

Writes data_bench/benchmarks/hotpotqa_bench.json.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data_bench", "benchmarks", "hotpotqa_bench.json")

API = ("https://datasets-server.huggingface.co/rows"
       "?dataset=hotpotqa%2Fhotpot_qa&config=distractor&split=validation"
       "&offset={offset}&length=100")


def fetch_rows() -> list:
    rows, offset = [], 0
    while len(rows) < 1600:
        url = API.format(offset=offset)
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = json.load(resp)
        batch = data.get("rows", [])
        if not batch:
            break
        rows.extend(r["row"] for r in batch)
        offset += len(batch)
        if offset >= data.get("num_rows_total", 0):
            break
        time.sleep(0.4)
    return rows


def main() -> int:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print("fetching hotpot_qa distractor validation rows...", flush=True)
    rows = fetch_rows()
    print(f"fetched {len(rows)} rows", flush=True)

    # deterministic: prefer bridge (true 2-hop) then comparison; skip yes/no
    # answers (trivial for lexical overlap and noisy for token-F1).
    rng = random.Random(2024)
    rng.shuffle(rows)
    picked, titles = [], {}
    for r in rows:
        if r["type"] != "bridge" or r["answer"].strip().lower() in {"yes", "no"}:
            continue
        sf_titles = list(dict.fromkeys(r["supporting_facts"]["title"]))
        if len(sf_titles) != 2:
            continue
        ctx = r["context"]
        paras = {t: "".join(sents)[:1500]
                 for t, sents in zip(ctx["title"], ctx["sentences"])}
        if not all(t in paras and len(paras[t]) > 100 for t in sf_titles):
            continue
        picked.append({"id": r["id"], "question": r["question"],
                       "answer": r["answer"], "answer_aliases": [],
                       "hops": 2, "supporting_titles": sf_titles,
                       "paragraphs": [{"title": t, "text": txt,
                                       "is_supporting": t in sf_titles}
                                      for t, txt in paras.items()]})
        for t in sf_titles:
            titles.setdefault(t, paras[t])
        if len(picked) >= 300:
            break
    # distractor doc for each question's supporting pair (adds 1 non-supporting
    # doc per question when it's not already in the corpus)
    for q in picked:
        non_sup = [p["title"] for p in q["paragraphs"] if not p["is_supporting"]]
        if non_sup:
            titles.setdefault(non_sup[0], q["paragraphs"][0]["text"])
    print(f"picked {len(picked)} bridge questions, corpus {len(titles)} titles",
          flush=True)

    bench = {"dataset": "hotpot_qa", "config": "distractor", "split": "validation",
             "questions": picked, "paragraphs": titles}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(bench, fh)
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
