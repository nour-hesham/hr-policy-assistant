"""
Central logging configuration.

Call configure_logging() once at application startup.  Uses only
Python's standard logging package — no third-party logging library.

Timestamps are genuinely UTC, achieved by setting the Formatter's
converter to time.gmtime so %(asctime)s always renders in UTC regardless
of the host timezone.
"""

import logging
import sys
import time


def configure_logging(level_name: str = "INFO") -> None:
    """Configure root logger to emit single-line UTC console events."""
    level = getattr(logging, level_name.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    # Force UTC — without this the default is time.localtime.
    formatter.converter = time.gmtime  # type: ignore[assignment]
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid adding duplicate handlers when called more than once.
    if not root.handlers:
        root.addHandler(handler)
    else:
        root.handlers = [handler]

    # The application emits its own privacy-safe completion summary. Uvicorn's
    # per-request access line would add noisy health entries and duplicate every
    # successful request at INFO.
    logging.getLogger("uvicorn.access").disabled = True
