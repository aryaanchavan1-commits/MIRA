"""Live web search with pluggable backends.

Priority (resolved at call time, never at import):
  1. agent-reach CLI (Panniantong/Agent-Reach) if installed on PATH
  2. opencli      CLI (jackwener/OpenCLI)        if installed on PATH
  3. stdlib urllib fetch of the DuckDuckGo HTML endpoint — zero dependencies

Consent gates (spec §40/§41): no call leaves the machine unless
`web_search.enabled: true` in config or the caller explicitly passes
allow_web=True. Results are plain data (title/url/snippet) — never executed.
"""
from __future__ import annotations

import html as html_mod
import logging
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger("mira.websearch")

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
_TIMEOUT = 15


def _run_cli(cmd: List[str], query: str) -> Optional[str]:
    try:
        p = subprocess.run(cmd + [query], capture_output=True, text=True,
                           timeout=45, shell=False)
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout
        logger.info("backend %s returned rc=%s", cmd[0], p.returncode)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.info("backend %s unavailable: %s", cmd[0], exc)
    return None


def _parse_jsonish(raw: str) -> List[Dict[str, Any]]:
    """Extract results from CLI output: JSON lines, or fenced JSON, or text."""
    import json
    out: List[Dict[str, Any]] = []
    try:  # whole-output JSON or JSONL
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            d = json.loads(line)
            if isinstance(d, dict) and (d.get("url") or d.get("title")):
                out.append({"title": str(d.get("title", "")),
                            "url": str(d.get("url", "")),
                            "snippet": str(d.get("snippet") or d.get("content") or
                                           d.get("description") or "")[:400]})
        if out:
            return out
    except Exception:
        pass
    for m in re.finditer(r"https?://\S+", raw):  # bare-URL fallback
        out.append({"title": "", "url": m.group(0).rstrip(".,)"), "snippet": ""})
    return out


def _agent_reach(query: str) -> Optional[List[Dict[str, Any]]]:
    if not shutil.which("agent-reach"):
        return None
    raw = _run_cli(["agent-reach", "search", "--json"], query)
    return _parse_jsonish(raw) if raw else None


def _opencli(query: str) -> Optional[List[Dict[str, Any]]]:
    if not shutil.which("opencli"):
        return None
    raw = _run_cli(["opencli", "search"], query)
    return _parse_jsonish(raw) if raw else None


def _duckduckgo(query: str) -> List[Dict[str, Any]]:
    """stdlib-only DDG HTML scrape — no dependency, works everywhere."""
    url = ("https://html.duckduckgo.com/html/?" +
           urllib.parse.urlencode({"q": query}))
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        page = r.read().decode("utf-8", "replace")
    out: List[Dict[str, Any]] = []
    for m in re.finditer(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
            r'(?:.*?class="result__snippet"[^>]*>(.*?)</a>)?',
            page, re.S):
        href, title, snip = m.group(1), m.group(2), m.group(3) or ""
        # DDG wraps URLs: /l/?uddg=<encoded>
        q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
        href = q.get("uddg", [href])[0]
        strip = lambda s: html_mod.unescape(re.sub(r"<[^>]+>", "", s)).strip()
        out.append({"title": strip(title), "url": href, "snippet": strip(snip)[:400]})
        if len(out) >= 8:
            break
    return out


def available_backends() -> List[str]:
    b = []
    if shutil.which("agent-reach"):
        b.append("agent-reach")
    if shutil.which("opencli"):
        b.append("opencli")
    b.append("duckduckgo")
    return b


def search(query: str, config: Optional[Dict[str, Any]] = None,
           allow_web: bool = False, backend: str = "auto") -> Dict[str, Any]:
    """Search the live web. Consent-gated; never raises on backend failure.

    Returns {backend, results, note}. Results are plain data.
    """
    cfg = (config or {}).get("web_search", {}) or {}
    if not (allow_web or cfg.get("enabled")):
        return {"backend": None, "results": [],
                "note": "web search disabled (config: web_search.enabled: true "
                        "to enable — every search leaves the machine)"}
    q = (query or "").strip()
    if not q:
        return {"backend": None, "results": [], "note": "empty query"}

    order = ([backend] if backend != "auto"
             else ["agent-reach", "opencli", "duckduckgo"])
    for name in order:
        try:
            if name == "agent-reach":
                res = _agent_reach(q)
            elif name == "opencli":
                res = _opencli(q)
            elif name == "duckduckgo":
                res = _duckduckgo(q)
            else:
                continue
        except Exception as exc:
            logger.warning("backend %s failed: %s", name, exc)
            res = None
        if res:
            return {"backend": name, "results": res, "note": ""}
    return {"backend": None, "results": [],
            "note": "no backend returned results (network down or blocked)"}


def fetch_page_text(url: str, max_chars: int = 20000) -> str:
    """Fetch a result page as plain text for ingestion (untrusted data:
    stripped to text, never executed, hard size cap)."""
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        ctype = r.headers.get("Content-Type", "")
        if "html" not in ctype and "text" not in ctype and "json" not in ctype:
            raise ValueError(f"unsupported content type: {ctype}")
        raw = r.read(max_chars * 4).decode("utf-8", "replace")
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
    text = html_mod.unescape(re.sub(r"<[^>]+>", " ", text))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]
