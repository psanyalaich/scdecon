"""Unit tests for scdecon.logging_utils."""

from __future__ import annotations

import logging

from scdecon.logging_utils import configure_logging, get_logger


def test_get_logger_is_namespaced() -> None:
    logger = get_logger("preprocessing.qc")
    assert logger.name == "scdecon.preprocessing.qc"
    assert isinstance(logger, logging.Logger)


def test_configure_logging_is_idempotent() -> None:
    configure_logging()
    root = logging.getLogger("scdecon")
    before = len(root.handlers)
    configure_logging()
    configure_logging()
    assert len(root.handlers) == before


def test_configure_logging_sets_level() -> None:
    configure_logging(logging.DEBUG)
    assert logging.getLogger("scdecon").level == logging.DEBUG
    # restore the default level for other tests
    configure_logging(logging.INFO)
