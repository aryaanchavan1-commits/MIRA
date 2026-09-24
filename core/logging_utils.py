"""Structured logging for MIRA (spec §42).

One root logger with a console handler and a rotating file handler under
logs/. Structured extra fields are appended as key=value pairs. Document
contents are never logged — only counts, ids, and event names.
"""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Any

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message",
}


class KeyValueFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in _RESERVED and not k.startswith("_")
        }
        if extras:
            kv = " ".join(f"{k}={v}" for k, v in sorted(extras.items()))
            base = f"{base} | {kv}"
        return base


def setup_logging(logs_dir: str, level: str = "INFO", quiet_console: bool = False) -> str:
    os.makedirs(logs_dir, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(KeyValueFormatter(_FORMAT))
    if quiet_console:
        console.setLevel(logging.WARNING)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        os.path.join(logs_dir, "mira.log"), maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(KeyValueFormatter(_FORMAT))
    root.addHandler(file_handler)
    return os.path.join(logs_dir, "mira.log")


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    logger.log(level, event, extra=fields)


def log_memory(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """For memory/pipeline events: same as log_event.

    Kept as a distinct name so call sites are greppable when auditing that
    raw document text is never passed as a field value.
    """
    logger.log(level, event, extra=fields)
