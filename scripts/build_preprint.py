"""Assemble the submission-ready preprint: paper/preprint.md (+ PDF via pandoc).

Stitches mira_paper.md with the auto-generated results file and a generated
front-matter block (title/abstract placeholder from the draft, reproducibility
statement, artifact manifest). Numbers are never duplicated here — the
results file is included verbatim so the preprint always matches artifacts.
"""
import datetime
import os
import re
import shutil
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")


def read(name: str) -> str:
    with open(os.path.join(PAPER, name), encoding="utf-8") as fh:
        return fh.read()


def main() -> int:
    paper = read("mira_paper.md")
    results = read("results_real.md")

    # strip the draft header (first heading block) from the paper body; the
    # preprint carries its own front matter
    body = re.sub(r"^# .*?\n\*\*Draft.*?\*\*\n+", "", paper, count=1, flags=re.S)

    manifest = "\n".join(
        f"- `{p}`" for p in [
            "data_bench/bench_real_results.json",
            "data_bench/bench_real_answers.json",
            "data_bench/ablation_real_results.json",
            "data_bench/scale_sweep_results.json",
            "data_bench/hotpotqa_bench_real_results.json",
            "data_bench/consolidation_results.json",
            "data_bench/aging_results.json",
            "data_indic/indicqa_results.json",
            "data_indic_ml/indicqa_ml_results.json",
            "experiments/neural_validation.json",
        ])

    front = f"""---
title: "Radial Memory Topologies for Retrieval-Augmented Generation: A Controlled Study of Mandala-Inspired Organization"
date: "{datetime.date.today().isoformat()}"
keywords: retrieval-augmented generation, memory topology, radial organization, ablation study, local-first AI, Indic languages
---

## Reproducibility statement

Every quantitative claim in this paper is generated from committed run
artifacts by `paper/export_results.py --real`; no table is hand-edited. All
experiments run on a single consumer laptop (RTX 3050 4GB, 16 GB RAM) with
zero cloud calls. Artifacts and builders:

{manifest}

"""
    preprint = front + body.replace(
        "**Measured (retrieval stage):** the canonical result tables are auto-generated in\n"
        "[`paper/results_real.md`](results_real.md) from the `data_bench/` artifacts.",
        "**Measured (retrieval stage):** canonical result tables follow.\n", 1
    )
    # insert the full results section right after the Results heading section
    preprint = preprint.replace(
        "---\n\n## 7. Threats to Validity",
        "---\n\n" + results + "\n---\n\n## 7. Threats to Validity", 1)

    out_md = os.path.join(PAPER, "preprint.md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(preprint)
    print(f"wrote {out_md} ({len(preprint)} chars)")

    pandoc = shutil.which("pandoc") or os.path.expandvars(
        r"%LOCALAPPDATA%\\Pandoc\\pandoc.exe")
    pdf_path = os.path.join(PAPER, "preprint.pdf")
    if os.path.exists(pandoc):
        try:
            subprocess.run([pandoc, out_md, "-o", os.path.join(PAPER, "preprint.html"),
                            "--standalone"], check=True, cwd=PAPER, timeout=600)
            print("wrote paper/preprint.html")
            chrome = None
            for cand in (r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
                         r"C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"):
                if os.path.exists(cand):
                    chrome = cand
                    break
            if chrome:
                subprocess.run([chrome, "--headless", "--disable-gpu",
                                "--no-pdf-header-footer",
                                "--print-to-pdf=" + pdf_path,
                                "file:///" + out_md.replace("\\", "/")],
                               check=True, timeout=300,
                               capture_output=True)
                print("wrote paper/preprint.pdf (chrome headless)")
            else:
                print("chrome not found; preprint.html is the print-ready form")
        except Exception as exc:
            print(f"PDF skipped ({exc}); markdown preprint is the deliverable")
    else:
        print("pandoc not installed; markdown preprint is the deliverable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
