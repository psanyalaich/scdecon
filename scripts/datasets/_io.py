"""Shared, transparent text-file opening for dataset ingestion.

Opens plain or gzip-compressed text uniformly, chosen by the ``.gz`` suffix, so
each loader can consume a downloaded ``.gz`` or a small uncompressed test
fixture without branching.
"""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO


@contextmanager
def open_text(path: Path) -> Iterator[IO[str]]:
    """Yield a text handle for ``path``, transparently decompressing ``.gz``.

    Parameters
    ----------
    path:
        File to open. If the suffix is ``.gz`` it is opened with :mod:`gzip`;
        otherwise it is opened as plain UTF-8 text.

    Yields
    ------
    IO[str]
        An open text-mode file handle, closed on exit.
    """
    handle: IO[str]
    if path.suffix == ".gz":
        handle = gzip.open(path, mode="rt", encoding="utf-8")
    else:
        handle = path.open(encoding="utf-8")
    try:
        yield handle
    finally:
        handle.close()
