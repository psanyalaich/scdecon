"""Structured logging setup for scdecon.

All package code logs through :func:`get_logger`, which returns a child of the
top-level ``scdecon`` logger and ensures a single stream handler is configured.
Using this instead of ``print`` gives every pipeline stage consistent, level-
controlled output.
"""

from __future__ import annotations

import logging

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_ROOT_NAME = "scdecon"
_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the top-level ``scdecon`` logger, idempotently.

    Our stream handler is attached exactly once per process (tracked by a
    module-level flag, so foreign handlers such as pytest's do not interfere);
    later calls only adjust the level. Repeated invocations (CLI, tests, library
    use) therefore never duplicate handlers.

    Parameters
    ----------
    level:
        Logging level for the ``scdecon`` logger (e.g. ``logging.INFO``).
    """
    global _configured
    logger = logging.getLogger(_ROOT_NAME)
    if not _configured:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
        _configured = True
    logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced ``scdecon`` logger, ensuring logging is configured.

    Parameters
    ----------
    name:
        Dotted suffix identifying the caller, e.g. ``"preprocessing.qc"``. The
        returned logger is named ``scdecon.<name>``.

    Returns
    -------
    logging.Logger
        A child logger of the top-level ``scdecon`` logger.
    """
    configure_logging()
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
