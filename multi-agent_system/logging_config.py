
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
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        force=True,
    )
