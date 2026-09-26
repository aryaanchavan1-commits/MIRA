"""Live web search with pluggable backends.

Priority (resolved at call time, never at import):
  1. agent-reach CLI (Panniantong/Agent-Reach) if installed on PATH
  2. opencli      CLI (jackwener/OpenCLI)        if installed on PATH
  3. stdlib urllib fetch of the DuckDuckGo HTML endpoint — zero dependencies

Consent gates (spec §40/§41): no call leaves the machine unless offline mode
is explicitly disabled and either `web_search.enabled: true` is set or the
caller passes the literal boolean allow_web=True. Results are plain data
(title/url/snippet) — never executed.
"""
from __future__ import annotations

import html as html_mod
import ipaddress
import logging
import re
import socket
import shutil
import subprocess
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger("mira.websearch")

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
_TIMEOUT = 15


def is_public_http_url(url: str) -> bool:
    """Reject local, private, malformed, and non-HTTP fetch targets."""
    try:
        parsed = urllib.parse.urlparse(str(url or ""))
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return False
        if parsed.username or parsed.password:
            return False
        port = parsed.port
        if port is None:
            port = 443 if parsed.scheme.lower() == "https" else 80
        if port not in {80, 443}:
            return False
        addresses = socket.getaddrinfo(
            parsed.hostname, port, type=socket.SOCK_STREAM)
        if not addresses:
            return False
        for info in addresses:
            address = ipaddress.ip_address(str(info[4][0]).split("%", 1)[0])
            if (address.is_private or address.is_loopback or address.is_link_local
                    or address.is_reserved or address.is_multicast
                    or address.is_unspecified):
                return False
        return True
    except (OSError, TypeError, ValueError):
        return False


def validate_public_url(url: str) -> None:
    if not is_public_http_url(url):
        raise ValueError("URL must be a public http(s) address")


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


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
    opener = urllib.request.build_opener(_SafeRedirectHandler())
    with opener.open(req, timeout=_TIMEOUT) as r:
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
        def strip(value: str) -> str:
            return html_mod.unescape(re.sub(r"<[^>]+>", "", value)).strip()

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


def web_allowed(config: Optional[Dict[str, Any]] = None,
                allow_web: Optional[bool] = None) -> bool:
    """Return whether a network call is permitted at this boundary.

    ``allow_web`` is deliberately strict: only the literal boolean ``True``
    grants per-request consent.  An explicit ``offline: true`` setting always
    wins; callers must opt out of offline mode in configuration before any
    search or page fetch can leave the process.
    """
    cfg = config if isinstance(config, dict) else {}
    # Offline is fail-closed even when an older config omitted the key.
    offline = cfg.get("offline", True)
    if type(offline) is not bool:
        raise ValueError("offline must be boolean")
    web_cfg = cfg.get("web_search", {}) or {}
    if not isinstance(web_cfg, dict):
        raise ValueError("web_search must be an object")
    enabled = web_cfg.get("enabled", False)
    if type(enabled) is not bool:
        raise ValueError("web_search.enabled must be boolean")
    if allow_web is not None and type(allow_web) is not bool:
        raise ValueError("allow_web must be boolean")
    if offline is True:
        return False
    if allow_web is not None:
        return allow_web is True
    return enabled is True


def search(query: str, config: Optional[Dict[str, Any]] = None,
           allow_web: Optional[bool] = None, backend: str = "auto") -> Dict[str, Any]:
    """Search the live web. Consent-gated; never raises on backend failure.

    Returns {backend, results, note}. Results are plain data.
    """
    if not web_allowed(config, allow_web):
        return {"backend": None, "results": [],
                "note": "web search disabled (explicit consent and offline=false required)"}
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


def fetch_page_text(url: str, max_chars: int = 20000,
                    config: Optional[Dict[str, Any]] = None,
                    allow_web: Optional[bool] = None) -> str:
    """Fetch a public result page as plain text after consent validation."""
    if not web_allowed(config, allow_web):
        raise ValueError("web fetch requires offline=false and explicit consent")
    validate_public_url(url)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    opener = urllib.request.build_opener(_SafeRedirectHandler())
    with opener.open(req, timeout=_TIMEOUT) as r:
        ctype = r.headers.get("Content-Type", "")
        if "html" not in ctype and "text" not in ctype and "json" not in ctype:
            raise ValueError(f"unsupported content type: {ctype}")
        raw = r.read(max_chars * 4).decode("utf-8", "replace")
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
    text = html_mod.unescape(re.sub(r"<[^>]+>", " ", text))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]
