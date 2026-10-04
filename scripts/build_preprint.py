"""Assemble the submission-ready preprint: paper/preprint.md (+ typeset PDF).

Stitches mira_paper.md with the auto-generated results file and a generated
front-matter block (author, reproducibility statement, artifact manifest).
Numbers are never duplicated here — the results file is included verbatim so the
preprint always matches artifacts.

PDF: pandoc renders the markdown to an HTML fragment, which is wrapped in a
print stylesheet (title block, hanging-indent references, break-safe tables) and
printed by headless Chrome. Printing the .md directly does not work — Chrome
does not render Markdown, it just emits the raw source as a text PDF.
"""
import datetime
import html
import os
import re
import shutil
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")

TITLE = ("Radial Memory Topologies for Retrieval-Augmented Generation: "
         "A Controlled Study of Mandala-Inspired Organization")
AUTHOR = "Aryan Chavan"
AFFILIATION = "Arynox Research Group · independent researcher · local-first AI systems"
CONTACT = "https://github.com/aryaanchavan1-commits/MIRA"
KEYWORDS = ("retrieval-augmented generation · memory topology · radial organization · "
            "catastrophic forgetting · continual learning · ablation study · local-first AI")

# Print stylesheet. Everything a journal would want and a raw dump cannot have:
# a real title block, hanging-indent references, tables that survive page breaks.
CSS = """
:root { --ink:#111; --dim:#555; --rule:#c9c4b8; --accent:#7a5c1e; --bg:#fbfaf7; }
@page { size: A4; margin: 22mm 20mm 20mm; }
* { box-sizing: border-box; }
body {
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  font-size: 10.6pt; line-height: 1.55; color: var(--ink); background: var(--bg);
  max-width: 172mm; margin: 0 auto; padding: 20mm 8mm; text-align: justify;
  hyphens: auto; -webkit-hyphens: auto;
}
.title-block { text-align: left; border-bottom: 2px solid var(--ink); padding-bottom: 1rem; }
.title-block h1 { font-size: 20pt; line-height: 1.22; margin: 0 0 0.7rem; letter-spacing: -0.015em; }
.authors { font-size: 12pt; margin: 0 0 0.2rem; }
.affiliation { color: var(--dim); font-size: 9.6pt; margin: 0 0 0.2rem; }
.contact, .preprint-note { color: var(--dim); font-size: 9pt; margin: 0; }
.preprint-note { margin-top: 0.6rem; }
h2 {
  font-size: 13pt; margin: 2.1rem 0 0.7rem; padding-bottom: 0.25rem;
  border-bottom: 1px solid var(--rule); letter-spacing: -0.01em; break-after: avoid;
}
h3 { font-size: 11.2pt; margin: 1.5rem 0 0.5rem; break-after: avoid; }
h4 { font-size: 10.4pt; margin: 1.1rem 0 0.4rem; break-after: avoid; }
p { margin: 0 0 0.75rem; orphans: 3; widows: 3; }
a { color: var(--accent); text-decoration: none; overflow-wrap: anywhere; }
strong { font-weight: 600; }
ul, ol { margin: 0 0 0.8rem; padding-left: 1.35rem; }
li { margin-bottom: 0.3rem; }
blockquote {
  margin: 1rem 0; padding: 0.5rem 0 0.5rem 1rem; border-left: 3px solid var(--rule);
  color: var(--dim); font-style: italic;
}
code {
  font-family: "Cascadia Mono", Consolas, "SF Mono", Menlo, monospace;
  font-size: 0.88em; background: #f0ece2; padding: 0.05em 0.28em; border-radius: 3px;
}
pre {
  background: #f2efe7; border: 1px solid var(--rule); border-radius: 4px;
  padding: 0.7rem 0.9rem; overflow-x: auto; break-inside: avoid;
  font-size: 8.6pt; line-height: 1.4;
}
pre code { background: none; padding: 0; }
hr { border: 0; border-top: 1px solid var(--rule); margin: 1.6rem 0; }
table {
  width: 100%; border-collapse: collapse; margin: 0.9rem 0 1.1rem;
  font-size: 8.5pt; font-variant-numeric: tabular-nums; break-inside: auto;
}
caption { caption-side: top; text-align: left; color: var(--dim); font-size: 8.6pt;
  padding-bottom: 0.4rem; }
th, td { padding: 0.28rem 0.4rem; border-bottom: 1px solid #e3ded1; text-align: left; }
thead th { border-bottom: 1.5px solid var(--ink); font-weight: 600; }
tbody tr:nth-child(even) { background: #f5f2ea; }
tr { break-inside: avoid; }
/* references and contribution lists read as hanging-indent bibliography entries */
#references + ul li, section:last-of-type ul li {
  margin-left: 1.6rem; text-indent: -1.6rem; margin-bottom: 0.45rem;
}
.abstract { font-size: 10pt; }
@media print { body { background: #fff; padding: 0; max-width: none; } }
"""


def read(name: str) -> str:
    with open(os.path.join(PAPER, name), encoding="utf-8") as fh:
        return fh.read()


def find_chrome() -> str:
    for cand in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                 r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"):
        if os.path.exists(cand):
            return cand
    return ""


def main() -> int:
    paper = read("mira_paper.md")
    results = read("results_real.md")
    today = datetime.date.today().isoformat()

    # strip the draft header (first heading block) from the paper body; the
    # preprint carries its own front matter
    body = re.sub(r"^# .*?\n\*\*Draft.*?\*\*\n+", "", paper, count=1, flags=re.S)
    # the markdown deliverable keeps the paper's own title/author block; the
    # typeset HTML replaces it with a print title block, so pandoc is handed
    # the front matter and the body without it
    m = re.match(r"\A.*?\n---\n\n", body, flags=re.S)
    header, body = (m.group(0), body[m.end():]) if m else ("", body)

    manifest = "\n".join(
        f"- `{p}`" for p in [
            "data_bench/bench_real_results.json",
            "data_bench/bench_real_answers.json",
            "data_bench/ablation_real_results.json",
            "data_bench/scale_sweep_results.json",
            "data_bench/hotpotqa_bench_real_results.json",
            "data_bench/consolidation_results.json",
            "data_bench/aging_results.json",
            "data_lab/forgetting_results.json",
            "data_indic/indicqa_results.json",
            "data_indic_ml/indicqa_ml_results.json",
            "experiments/neural_validation.json",
        ])

    front = f"""## Reproducibility statement

Every quantitative claim in this paper is generated from committed run
artifacts by `paper/export_results.py --real`; no table is hand-edited. All
experiments run on a single consumer laptop (RTX 3050 4GB, 16 GB RAM) with
zero cloud calls. Artifacts and builders:

{manifest}

---

"""
    # results section is inserted between the narrative sections and Threats
    rest = body.replace("## 7. Threats to Validity",
                        results + "\n## 7. Threats to Validity", 1)

    out_md = os.path.join(PAPER, "preprint.md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(header + front + rest)
    print(f"wrote {out_md} ({len(header) + len(front) + len(rest)} chars)")

    pandoc = shutil.which("pandoc") or os.path.expandvars(
        r"%LOCALAPPDATA%\\Pandoc\\pandoc.exe")
    chrome = find_chrome()
    if not (pandoc and chrome):
        print("pandoc and/or chrome missing; markdown preprint is the deliverable")
        return 0
    try:
        frag = subprocess.run([pandoc, "-f", "markdown+pipe_tables", "-t", "html5"],
                              input=front + rest, check=True, cwd=PAPER,
                              timeout=600, capture_output=True, text=True,
                              encoding="utf-8", errors="replace").stdout
        if not frag.strip():
            raise RuntimeError("pandoc produced no HTML")
        # the abstract keeps its own heading in the body; the title block above
        # carries title/authorship only
        doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(TITLE)} — {html.escape(AUTHOR)}</title>
<meta name="author" content="{html.escape(AUTHOR)}">
<meta name="description" content="{html.escape(KEYWORDS)}">
<style>{CSS}</style></head><body>
<div class="title-block">
  <h1>{html.escape(TITLE)}</h1>
  <p class="authors">{html.escape(AUTHOR)}</p>
  <p class="affiliation">{html.escape(AFFILIATION)}</p>
  <p class="contact">{html.escape(CONTACT)}</p>
  <p class="preprint-note">Preprint, {today}. Not peer reviewed. Every number is
  generated from committed run artifacts; see the reproducibility statement.</p>
</div>
{frag}
</body></html>"""
        out_html = os.path.join(PAPER, "preprint.html")
        with open(out_html, "w", encoding="utf-8") as fh:
            fh.write(doc)
        print(f"wrote {out_html}")

        pdf_path = os.path.join(PAPER, "preprint.pdf")
        subprocess.run([chrome, "--headless", "--disable-gpu",
                        "--no-pdf-header-footer", "--print-to-pdf=" + pdf_path,
                        "file:///" + out_html.replace("\\", "/")],
                       check=True, timeout=600, capture_output=True)
        print(f"wrote {pdf_path} ({os.path.getsize(pdf_path)} bytes)")
    except Exception as exc:  # noqa: BLE001 - deliver the markdown regardless
        print(f"PDF skipped ({exc}); markdown preprint is the deliverable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())