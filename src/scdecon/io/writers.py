"""Writers that persist scdecon data structures to disk.

Writers mirror the readers in :mod:`scdecon.io.readers`: they accept a string or
:class:`pathlib.Path`, normalise it to a ``Path`` immediately, create any missing
parent directories, and write the data exactly as given without transforming it.
Each writer returns the ``Path`` it wrote, which is convenient for logging and
for wiring pipeline stages together.
"""

from __future__ import annotations

from pathlib import Path

import anndata
import pandas as pd

from scdecon.io.readers import _resolve_separator

__all__ = ["write_h5ad", "write_table"]


def write_table(frame: pd.DataFrame, path: str | Path, sep: str | None = None) -> Path:
    """Write a DataFrame to a delimited text file, preserving row order.

    Supported formats mirror :func:`scdecon.io.read_bulk`:

    - ``.tsv`` -- tab-separated (separator ``"\\t"``)
    - ``.csv`` -- comma-separated (separator ``","``)

    The index is written as the first column so the file round-trips through the
    matching reader. Rows and columns are written in their current order; the
    writer never sorts or reorders them.

    Parameters
    ----------
    frame:
        The table to write. Its index becomes the first column on disk.
    path:
        Destination path. Missing parent directories are created.
    sep:
        Column separator. If ``None``, it is inferred from the file suffix; pass
        a value to write a file with an unsupported suffix.

    Returns
    -------
    pathlib.Path
        The path that was written.

    Raises
    ------
    ValueError
        If the separator cannot be inferred from the suffix and ``sep`` is not
        given.
    """
    resolved = Path(path)
    separator = _resolve_separator(resolved, sep)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(resolved, sep=separator)
    return resolved


def write_h5ad(adata: anndata.AnnData, path: str | Path) -> Path:
    """Write an AnnData object to an ``.h5ad`` file.

    The object is written exactly as given, including a sparse ``.X`` if present.

    Parameters
    ----------
    adata:
        The AnnData object to persist.
    path:
        Destination path. Missing parent directories are created.

    Returns
    -------
    pathlib.Path
        The path that was written.
    """
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(resolved)
    return resolved
