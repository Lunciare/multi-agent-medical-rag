"""Centralised logging configuration for the multi-agent medical RAG system.

Production entry points (`main.py`, `tests/evaluate_*.py`) call
`configure_logging()` once at startup. After that, any module using

    import logging
    logger = logging.getLogger(__name__)

writes structured log lines with timestamps and log levels instead of
bare `print()` calls — which is what the audit's Part A flagged in
`orchestrator.py:answer()` and `specialist.py:answer()`.

Log level can be overridden via the `LOG_LEVEL` environment variable
(`LOG_LEVEL=DEBUG python main.py` surfaces per-chunk retrieval
previews; the default `INFO` hides them but still shows the one-time
FAISS-load and per-query routing / refusal-gate decisions).

`configure_logging()` is idempotent: it calls `logging.basicConfig`
with `force=True` so repeat calls (e.g. one in `main.py`, another in
an eval script that imports `main`) produce the same final state
without duplicate handlers.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Union


DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: Optional[Union[str, int]] = None,
                      *,
                      fmt: str = DEFAULT_FORMAT,
                      datefmt: str = DEFAULT_DATEFMT) -> None:
    """Configure root logging for the multi-agent system.

    Args:
        level: Log level name (`"INFO"`, `"DEBUG"`, …) or numeric level.
            Defaults to the `LOG_LEVEL` env var, or `INFO` if unset.
        fmt: Log line format string.
        datefmt: Timestamp format string.
    """
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        force=True,
    )
