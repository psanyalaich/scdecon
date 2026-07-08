"""Readers for the input data formats scdecon consumes.

Every reader accepts a string or :class:`pathlib.Path`, normalises it to a
``Path`` immediately, and validates only file integrity and basic structure
(existence, non-emptiness, unique identifiers, numeric where required). Readers
never normalise, filter, reorder, or otherwise transform the biological data;
values are returned exactly as stored on disk.
"""

from __future__ import annotations

from pathlib import Path

import anndata
import pandas as pd
from pandas.api.types import is_numeric_dtype

__all__ = ["read_bulk", "read_h5ad", "read_metadata"]

_SEP_BY_SUFFIX = {".tsv": "\t", ".csv": ","}


def _resolve_separator(path: Path, sep: str | None) -> str:
    """Return an explicit separator or infer one from the file suffix."""
    if sep is not None:
        return sep
    try:
        return _SEP_BY_SUFFIX[path.suffix.lower()]
    except KeyError:
        raise ValueError(
            f"Cannot infer a column separator from suffix '{path.suffix}' "
            f"({path}). Use a .tsv or .csv file, or pass sep explicitly."
        ) from None


def read_h5ad(path: str | Path) -> anndata.AnnData:
    """Load an annotated single-cell dataset from an ``.h5ad`` file.

    Parameters
    ----------
    path:
        Path to a ``.h5ad`` file written by AnnData/Scanpy.

    Returns
    -------
    anndata.AnnData
        The dataset exactly as stored, including a sparse ``.X`` if present.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not point to an existing file.
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"h5ad file not found: {resolved}")
    return anndata.read_h5ad(resolved)


def read_bulk(path: str | Path, sep: str | None = None) -> pd.DataFrame:
    """Load a bulk expression matrix as a genes-by-samples DataFrame.

    The first column becomes the gene index; the remaining columns are samples.
    Gene rows are returned in exactly the order they appear in the source file:
    the reader never sorts, filters, or reorders them.

    Supported formats (chosen from the file suffix, never sniffed from content):

    - ``.tsv`` -- tab-separated (separator ``"\\t"``)
    - ``.csv`` -- comma-separated (separator ``","``)

    Any other suffix is intentionally unsupported and raises ``ValueError``
    unless an explicit ``sep`` is provided.

    Parameters
    ----------
    path:
        Path to a bulk expression matrix (``.tsv`` or ``.csv``).
    sep:
        Column separator. If ``None``, it is inferred from the file suffix as
        described above; pass a value to read a file with an unsupported suffix.

    Returns
    -------
    pandas.DataFrame
        Genes (index) by samples (columns), with values unchanged from disk.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not point to an existing file.
    ValueError
        If the separator cannot be inferred, the table is empty, gene
        identifiers are duplicated, or any sample column is non-numeric.
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Bulk matrix file not found: {resolved}")
    separator = _resolve_separator(resolved, sep)
    try:
        frame = pd.read_csv(resolved, sep=separator, index_col=0)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Bulk matrix is empty: {resolved}") from exc

    if frame.empty:
        raise ValueError(f"Bulk matrix has no data rows or sample columns: {resolved}")
    if frame.index.has_duplicates:
        duplicates = sorted(set(frame.index[frame.index.duplicated()]))
        raise ValueError(
            f"Bulk matrix contains duplicate gene identifiers {duplicates}: "
            f"{resolved}. Gene identifiers must be unique."
        )
    non_numeric = [str(c) for c in frame.columns if not is_numeric_dtype(frame[c])]
    if non_numeric:
        raise ValueError(
            f"Bulk matrix has non-numeric sample columns {non_numeric}: "
            f"{resolved}. Expected a numeric genes-by-samples matrix."
        )
    return frame


def read_metadata(path: str | Path, index_col: int | str = 0) -> pd.DataFrame:
    """Load a sample- or cell-annotation table.

    Parameters
    ----------
    path:
        Path to a metadata table (``.tsv`` or ``.csv``).
    index_col:
        Column to use as the row index (identifiers). Defaults to the first
        column.

    Returns
    -------
    pandas.DataFrame
        The metadata table with values unchanged from disk.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not point to an existing file.
    ValueError
        If the separator cannot be inferred, the table is empty, or the index
        contains duplicate identifiers.
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Metadata file not found: {resolved}")
    separator = _resolve_separator(resolved, sep=None)
    try:
        frame = pd.read_csv(resolved, sep=separator, index_col=index_col)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Metadata table is empty: {resolved}") from exc

    if frame.empty:
        raise ValueError(f"Metadata table has no rows or columns: {resolved}")
    if frame.index.has_duplicates:
        duplicates = sorted(set(frame.index[frame.index.duplicated()]))
        raise ValueError(
            f"Metadata table contains duplicate identifiers {duplicates}: "
            f"{resolved}. Identifiers must be unique."
        )
    return frame
