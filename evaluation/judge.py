"""LLM-as-judge metric (evaluation upgrade).

Uses the local GGUF model as a judge: scores each produced answer for
- correctness (vs the reference answer), and
- faithfulness (grounding in the provided retrieved evidence)

on a 1-5 scale. The judge is the SAME small local model used for answering —
its judgments are weak but real, self-preference bias is possible, and this
is stated wherever judge scores appear. Output parsing is defensive: a judge
that fails to emit an integer is recorded as judge_error, never as a score.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_CORRECTNESS_PROMPT = """You are grading a question-answering system. \
Rate the ASSISTANT ANSWER's correctness compared to the REFERENCE answer \
on a scale of 1-5 (5 = correct and complete, 1 = wrong or no answer). \
Reply with ONLY the integer.

Question: {question}
Reference answer: {reference}
Assistant answer: {answer}

Integer (1-5):"""

_FAITHFULNESS_PROMPT = """You are grading a question-answering system. \
Rate how well the ASSISTANT ANSWER is grounded in the EVIDENCE (no invented \
facts) on a scale of 1-5 (5 = every claim comes from the evidence, 1 = mostly \
invented). Reply with ONLY the integer.

Evidence:
{evidence}

Assistant answer: {answer}

Integer (1-5):"""


class LLMJudge:
    def __init__(self, llm):
        self.llm = llm
        self._mem: Dict[str, int] = {}   # exact (prompt) -> score memo

    def _score(self, prompt: str) -> Optional[int]:
        if prompt in self._mem:
            return self._mem[prompt]
        raw = (self.llm.chat([{"role": "user", "content": prompt}],
                             max_tokens=8, temperature=0.0) or "").strip()
        m = re.search(r"[1-5]", raw)
        score = int(m.group(0)) if m else None
        self._mem[prompt] = score
        return score

    def correctness(self, question: str, answer: str, reference: str) -> Optional[int]:
        return self._score(_CORRECTNESS_PROMPT.format(
            question=question[:500], reference=reference[:300],
            answer=(answer or "")[:500]))

    def faithfulness(self, answer: str, evidence: str) -> Optional[int]:
        return self._score(_FAITHFULNESS_PROMPT.format(
            evidence=(evidence or "")[:1500], answer=(answer or "")[:500]))


def judge_records(judge: LLMJudge, rows: List[Dict[str, Any]],
                  answers: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
    """Attach judge scores to benchmark rows (in place) and return aggregates.

    rows: per-question rows as produced by run_system(); row index aligns with
    the record list when answers are provided positionally.
    answers: optional {row_index: evidence_text} for faithfulness scoring.
    """
    correct: List[float] = []
    faith: List[float] = []
    n_err = 0
    for i, row in enumerate(rows):
        ans = row.get("answer_text") or ""
        ref = row.get("reference_answer") or ""
        c = judge.correctness(row.get("question", ""), ans, ref)
        f = judge.faithfulness(ans, (answers or {}).get(i, ""))
        if c is None or f is None:
            n_err += 1
            continue
        row["judge_correctness"] = c
        row["judge_faithfulness"] = f
        correct.append(c)
        faith.append(f)
    agg = {"n_judged": len(correct)}
    if correct:
        agg["judge_correctness"] = round(sum(correct) / len(correct), 3)
    if faith:
        agg["judge_faithfulness"] = round(sum(faith) / len(faith), 3)
    agg["judge_errors"] = n_err
    return agg
