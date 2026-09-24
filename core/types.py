"""Shared MIRA types and helpers.

Small, dependency-light module that everything else imports. Keeps the
dataclass plumbing for the memory model and a few cross-cutting utilities
(id generation, timestamps, safe parsing) in one place.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utcnow().isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    """Short readable id, e.g. mem_a1b2c3d4e5f6."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def safe_json_dumps(obj: Any) -> str:
    def _default(o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        raise TypeError(f"Not JSON serializable: {type(o)}")
    return json.dumps(obj, default=_default, ensure_ascii=False)


_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def parse_int(value: Any, default: int) -> int:
    try:
        if isinstance(value, bool):
            return default
        if isinstance(value, int):
            return value
        s = str(value).strip()
        return int(s) if _INT_RE.match(s) else default
    except Exception:
        return default


def parse_float(value: Any, default: float) -> float:
    try:
        if isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return float(value)
        s = str(value).strip().lower()
        s = {"auto": "", "none": ""}.get(s, s)  # "auto" falls through to default
        return float(s) if _FLOAT_RE.match(s) else default
    except Exception:
        return default


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def chunked(iterable: List[Any], size: int) -> Iterator[List[Any]]:
    for i in range(0, len(iterable), max(1, size)):
        yield iterable[i : i + size]
