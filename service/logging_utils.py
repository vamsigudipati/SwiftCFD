"""Structured JSON-lines request logging.

Every ``/predict`` call emits one line like::

    {"ts": "2026-11-03T10:00:00.123Z", "event": "predict", "input_hash": "ab12…",
     "latency_ms": 41.2, "in_envelope": true, "status": 200, "outcome": "ok"}

Lines go to stdout (Docker-friendly) and, if ``SWIFTCFD_LOG_FILE`` is set,
are appended to that file as well.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
        }
        extra = getattr(record, "data", None)
        if isinstance(extra, dict):
            payload.update(extra)
        else:
            payload["message"] = record.getMessage()
        return json.dumps(payload, separators=(",", ":"))


def configure_logging(level: str | None = None) -> logging.Logger:
    level = level or os.environ.get("SWIFTCFD_LOG_LEVEL", "INFO")
    root = logging.getLogger("swiftcfd")
    root.setLevel(level)
    root.propagate = False
    if root.handlers:  # already configured (e.g. test re-import)
        return root
    fmt = JsonLineFormatter()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    log_file = os.environ.get("SWIFTCFD_LOG_FILE")
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    return root


def input_hash(payload: dict) -> str:
    """Stable sha256 of a request body (sorted keys, compact separators)."""
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def log_request(logger: logging.Logger, **data) -> None:
    logger.info("request", extra={"data": {"event": "predict", **data}})
