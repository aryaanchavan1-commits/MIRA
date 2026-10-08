"""End-to-end memory-system self-check (the "is it working" runnable check).

Exercises the full write→read→version→rollback→restart path through the same
MemoryEngine the SDK exposes, against the real local store:

  1. ingest text → nodes/edges created
  2. retrieve → scored hits with the full component explanation
  3. remember v1, v2 → append-only history with supersession links
  4. rollback to v1 → current head is v1's content (append-only)
  5. NEW engine over the same store → history survives the restart (SQLite)

Usage: .venv/Scripts/python.exe scripts/check_memory_system.py
Exit 0 = all checks passed. LLM is never loaded (retrieval-only path).
"""
from __future__ import annotations

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mira_sdk import MemoryEngine  # noqa: E402

TS = time.strftime("%Y%m%d%H%M%S")
SUBJECT = f"selfcheck deployment topic {TS}"   # unique per run: prior runs persist


def main() -> int:
    failures = []

    def check(name, cond, detail=""):
        print(f"{'PASS' if cond else 'FAIL'} {name}" + (f" :: {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    t0 = time.time()
    eng = MemoryEngine(load_llm=False)

    # 1. ingest
    stats = eng.add(f"The selfcheck-{TS} satellite orbits at 540 km altitude. "
                    "Its primary sensor is a multispectral imager.",
                    title="selfcheck doc")
    check("ingest_creates_nodes", stats.get("n_nodes", 0) > 0, str(stats)[:120])

    # 2. retrieve with explanations
    hits = eng.retrieve("What altitude does the selfcheck satellite orbit at?")
    check("retrieve_hits", len(hits["results"]) > 0)
    if hits["results"]:
        top = hits["results"][0]
        comps = top.get("explanation", {})
        check("retrieval_explained", len(comps) >= 8, f"{len(comps)} components")
        check("retrieval_relevant",
              "satellite" in (top.get("text", "") + top.get("concept", "")).lower(),
              top.get("concept", ""))

    # 3. versioned memory
    v1 = eng.remember(SUBJECT, "Deployment status: planned for October.",
                      source="chat selfcheck")
    v2 = eng.remember(SUBJECT, "Deployment status: completed on schedule.",
                      source="chat selfcheck")
    hist = eng.history(SUBJECT)
    check("versions_append", len(hist) == 2, f"len={len(hist)}")
    check("supersession_linked",
          len(hist) == 2 and hist[1]["supersedes"] == hist[0]["hash"])
    check("pramana_labeled",
          all(r.get("pramana") == "pratyaksa" for r in hist),  # chat = direct
          str([r.get("pramana") for r in hist]))

    # 4. rollback is append-only
    rb = eng.rollback(SUBJECT, 1)
    check("rollback_appends", eng.history(SUBJECT)[-1]["content"].startswith("Deployment status: planned")
          and len(eng.history(SUBJECT)) == 3, f"len={len(eng.history(SUBJECT))}")
    eng.close()

    # 5. persistence across restart (new engine, same store)
    eng2 = MemoryEngine(load_llm=False)
    hist2 = eng2.history(SUBJECT)
    check("restart_survives", len(hist2) == 3, f"len={len(hist2)}")
    check("restart_head_is_rollback", hist2 and hist2[-1]["content"].startswith("Deployment status: planned"))
    check("restart_annotation", bool(hist2) and hist2[-1].get("pramana") == "anumana"
          and len(hist2[-1].get("katapayadi", "")) > 0,
          str(hist2[-1].get("pramana")))
    eng2.close()

    print(f"\n{'MEMORY SYSTEM CHECK FAILURES: ' + ', '.join(failures) if failures else 'MEMORY SYSTEM OK'}"
          f" ({time.time() - t0:.0f}s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
