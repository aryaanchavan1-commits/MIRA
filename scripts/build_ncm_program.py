"""Typeset the MIRA-NCM research-program document: paper/ncm_program.pdf.

Mirrors build_preprint.py's print pipeline (pandoc -> Chrome -> pypdf metadata)
and reuses its stylesheet verbatim so both PDFs read as one series.
"""
import html
import os
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_preprint import CSS, PAPER, find_chrome  # noqa: E402

# Slightly denser set than the preprint so the program fits ~10 pages.
CSS += ("\nbody { font-size: 10.2pt; line-height: 1.48; }"
        "\nh2 { margin: 1.7rem 0 0.6rem; }"
        "\np { margin: 0 0 0.65rem; }"
        "\ntable { font-size: 8.2pt; }")

TITLE = ("MIRA-NCM: Nested Constellation Memory — Research Program, Hypotheses, "
         "and Future Scope for Structured Long-Term AI Memory")
AUTHOR = "Aryan Chavan"
CONTACT = "https://github.com/aryaanchavan1-commits/MIRA"
KEYWORDS = ("nested constellation memory · long-term AI memory · retrieval-augmented "
            "generation · catastrophic forgetting · continual learning · ablation "
            "study · versioned memory · local-first AI")


def main() -> int:
    src = os.path.join(PAPER, "ncm_program.md")
    with open(src, encoding="utf-8") as fh:
        body = fh.read()

    pandoc = subprocess.run(["pandoc", "-f", "markdown+pipe_tables", "-t", "html5"],
                            input=body, check=True, timeout=600,
                            capture_output=True, text=True,
                            encoding="utf-8", errors="replace").stdout
    if not pandoc.strip():
        raise RuntimeError("pandoc produced no HTML")

    # The markdown carries its own abstract heading; add the print title block.
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(TITLE)} — {html.escape(AUTHOR)}</title>
<meta name="author" content="{html.escape(AUTHOR)}">
<meta name="description" content="{html.escape(KEYWORDS)}">
<style>{CSS}</style></head><body>
<div class="title-block">
  <h1>{html.escape(TITLE)}</h1>
  <p class="authors">{html.escape(AUTHOR)}</p>
  <p class="contact">{html.escape(CONTACT)}</p>
  <p class="preprint-note">Research program document. Hypotheses and future work;
  no unevaluated component is reported as a result. Every measured number cites a
  committed run artifact.</p>
</div>
{pandoc}
</body></html>"""
    out_html = os.path.join(PAPER, "ncm_program.html")
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(doc)

    chrome = find_chrome()
    if not chrome:
        raise RuntimeError("chrome not found")
    pdf_path = os.path.join(PAPER, "ncm_program.pdf")
    subprocess.run([chrome, "--headless", "--disable-gpu",
                    "--no-pdf-header-footer", "--print-to-pdf=" + pdf_path,
                    "file:///" + out_html.replace("\\", "/")],
                   check=True, timeout=600, capture_output=True)

    from pypdf import PdfReader, PdfWriter
    reader = PdfReader(pdf_path)
    writer = PdfWriter(clone_from=reader)
    writer.add_metadata({"/Title": f"{TITLE} — {AUTHOR}", "/Author": AUTHOR,
                         "/Keywords": KEYWORDS})
    with open(pdf_path, "wb") as fh:
        writer.write(fh)
    print(f"wrote {pdf_path} ({os.path.getsize(pdf_path)} bytes, "
          f"{len(reader.pages)} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
