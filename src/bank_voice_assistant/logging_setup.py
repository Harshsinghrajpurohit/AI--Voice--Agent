"""Logging setup: human-readable console output, optionally verbose.

Kept on the standard library only so the project has no logging dependency.
"""

from __future__ import annotations

import logging
import sys

LOGGER_NAME = "bva"
PACKAGE_LOGGER_NAME = "bank_voice_assistant"
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_TIME_FORMAT = "%H:%M:%S"


def configure_logging(verbose: bool = False, quiet: bool = False) -> logging.Logger:
    """Configure and return the project logger.

    Args:
        verbose: enable DEBUG output (per-stage internals, scores, guard events).
        quiet: only warnings and errors (used when stdout is reserved for answers).
    """
    if verbose and quiet:
        raise ValueError("verbose and quiet are mutually exclusive")

    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_TIME_FORMAT))

    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)
    # We own this logger's output; don't let it propagate to the root logger
    # (avoids duplicate lines when a host app configures logging too).
    logger.propagate = False

    # Every module logs through ``logging.getLogger(__name__)``, i.e. a child of the
    # package logger. Without this the stage logs (and the Phase 6 turn telemetry)
    # would fall through to the root logger and be dropped entirely.
    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    package_logger.handlers.clear()
    package_logger.addHandler(handler)
    package_logger.setLevel(level)
    package_logger.propagate = False

    return logger


def get_logger(suffix: str | None = None) -> logging.Logger:
    """Return a child logger, e.g. ``get_logger("retrieval")``."""
    return logging.getLogger(f"{LOGGER_NAME}.{suffix}" if suffix else LOGGER_NAME)
