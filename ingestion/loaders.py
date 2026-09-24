"""Document text extraction (spec §16/§41).

Loaders for pdf/txt/md/json/csv/docx return [(page, text)]. All extracted
text is treated as untrusted data: control characters stripped, nothing
found in a document is ever executed (§41).
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
from typing import Any, List, Tuple

logger = logging.getLogger("mira.loaders")

_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize(text: str) -> str:
    """Strip control chars; normalize newlines. Document text is untrusted (§41)."""
    return _CTRL_RE.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))


def load_pdf(path: str) -> List[Tuple[int, str]]:
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt.strip():
            pages.append((i, sanitize(txt)))
    return pages


def load_text(path: str) -> List[Tuple[int, str]]:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as fh:
                return [(1, sanitize(fh.read()))]
        except UnicodeDecodeError:
            continue
    return [(1, "")]


def load_markdown(path: str) -> List[Tuple[int, str]]:
    pages = load_text(path)
    if not pages:
        return []
    text = pages[0][1]
    text = re.sub(r"```[\w-]*\n?", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[#*_>`]{1,3}", "", text)
    return [(1, text)]


def load_json(path: str) -> List[Tuple[int, str]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    out: List[Tuple[int, str]] = []

    def _walk(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, f"{k}: ")
        elif isinstance(obj, list):
            for v in obj:
                _walk(v, prefix)
        elif isinstance(obj, str):
            if obj.strip():
                out.append((1, sanitize(obj)))
        elif isinstance(obj, (int, float, bool)):
            out.append((1, f"{prefix}{obj}"))
    _walk(data)
    return out


def load_csv(path: str) -> List[Tuple[int, str]]:
    rows: List[Tuple[int, str]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(fh, dialect)
        header: List[str] = []
        for i, row in enumerate(reader):
            if i == 0:
                header = [h.strip() for h in row]
                continue
            parts = [f"{h}: {v}" for h, v in zip(header, row) if str(v).strip()]
            if parts:
                rows.append((1, sanitize("; ".join(parts))))
    return rows


def load_docx(path: str) -> List[Tuple[int, str]]:
    import docx
    document = docx.Document(path)
    paras = [sanitize(p.text) for p in document.paragraphs if p.text.strip()]
    return [(1, "\n".join(paras))] if paras else []


LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_text,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".json": load_json,
    ".csv": load_csv,
    ".docx": load_docx,
}


def load_any(path: str) -> List[Tuple[int, str]]:
    ext = os.path.splitext(path)[1].lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"unsupported file type: {ext}")
    return loader(path)
