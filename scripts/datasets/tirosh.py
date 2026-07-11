"""Ingest the Tirosh et al. 2016 melanoma scRNA reference (GEO GSE72056).

Dataset-specific ingestion (lives in ``scripts/``, not the package). The
GSE72056 supplementary matrix is genes x cells, tab-separated, with three
annotation rows before the gene rows (verified against the real file)::

    Cell                                     <cell_1>  <cell_2>  ...
    tumor                                    <tumour id per cell>
    malignant(1=no,2=yes,0=unresolved)       <code per cell>
    non-malignant cell type (1=T,...,6=NK)   <code per cell>
    <GENE_SYMBOL>                            <log2(TPM/10 + 1) per cell>
    ...

Cell-type policy (ADR-0010, D1): the retained types are
``{malignant, T, B, Macrophage, Endothelial, CAF, NK}``; cells the source marks
unresolved are excluded by default.

Scale policy (ADR-0010, D2): values are ``log2(TPM/10 + 1)``. We reconstruct
linear TPM ``(2**x - 1) * 10`` and return an AnnData whose ``.X`` is natural
``log1p`` of the linear values and whose ``layers["counts"]`` holds the linear
values, so the existing M2/M3 signature pipeline applies unchanged.

The large gene x cell block is parsed with pandas' C parser (not by accumulating
Python floats), so the real matrix (~23k genes x ~4.6k cells) is read without
materialising ~100M boxed floats.
"""

from __future__ import annotations

import csv
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

from scdecon.logging_utils import get_logger
from scripts.datasets._io import open_text

__all__ = ["CELL_TYPE_CODES", "MALIGNANT_LABEL", "load_tirosh_reference"]

logger = get_logger("scripts.datasets.tirosh")

#: Label used for cells the source marks malignant.
MALIGNANT_LABEL = "malignant"

#: Encoding of the "non-malignant cell type" row. Code 0 (and any other value)
#: denotes an unresolved / unassigned cell.
CELL_TYPE_CODES = {
    1: "T",
    2: "B",
    3: "Macrophage",
    4: "Endothelial",
    5: "CAF",
    6: "NK",
}

_MALIGNANT_NO = 1
_MALIGNANT_YES = 2
_UNRESOLVED_LABEL = "unresolved"
_N_ANNOTATION_ROWS = 3


def load_tirosh_reference(
    path: str | Path, *, exclude_unresolved: bool = True
) -> anndata.AnnData:
    """Load GSE72056 into an analysis-ready AnnData (cells x genes).

    Parameters
    ----------
    path:
        Path to the GSE72056 matrix (optionally gzip-compressed).
    exclude_unresolved:
        If ``True`` (default), drop cells that are neither malignant nor a coded
        non-malignant type. If ``False``, keep them with cell type
        ``"unresolved"``.

    Returns
    -------
    anndata.AnnData
        Cells x genes. ``.X`` is natural ``log1p`` of reconstructed linear TPM;
        ``layers["counts"]`` holds the linear TPM. ``obs`` carries ``tumor`` and
        ``cell_type``; ``var_names`` are gene symbols.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file has fewer than the expected annotation rows, has no gene
        rows, has fewer expression columns than cells, or its header labels do
        not match the expected format.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tirosh matrix not found: {path}")

    header, tumor_row, malignant_row, celltype_row = _read_annotation_rows(path)
    _validate_header_labels(header[0], malignant_row[0], celltype_row[0])

    cell_names = header[1:]
    n_cells = len(cell_names)
    tumors = tumor_row[1 : n_cells + 1]
    malignant = [int(float(value)) for value in malignant_row[1 : n_cells + 1]]
    celltype = [int(float(value)) for value in celltype_row[1 : n_cells + 1]]

    matrix = _deduplicate_genes(_read_expression_matrix(path, n_cells))
    labels = _decode_cell_types(malignant, celltype)

    if exclude_unresolved:
        keep = [i for i, label in enumerate(labels) if label is not None]
    else:
        keep = list(range(n_cells))
        labels = [label if label is not None else _UNRESOLVED_LABEL for label in labels]
    if not keep:
        raise ValueError("No cells remain after excluding unresolved cells.")

    # Reconstruct linear TPM, then natural log1p (ADR-0010 D2). Columns of the
    # gene x cell matrix are cells in file order, so keep-indices select columns.
    keep_cols = np.asarray(keep, dtype=np.intp)
    log2_kept = matrix.to_numpy(dtype=np.float64)[:, keep_cols].T
    linear = (np.power(2.0, log2_kept) - 1.0) * 10.0
    x_log1p = np.log1p(linear)

    obs = pd.DataFrame(
        {
            "tumor": [tumors[i] for i in keep],
            "cell_type": [labels[i] for i in keep],
        },
        index=pd.Index([cell_names[i] for i in keep], name="cell"),
    )
    var = pd.DataFrame(index=matrix.index)
    adata = anndata.AnnData(X=x_log1p, obs=obs, var=var)
    adata.layers["counts"] = linear
    logger.info(
        "Loaded Tirosh reference: %d cells x %d genes (%d cell types)",
        adata.n_obs,
        adata.n_vars,
        obs["cell_type"].nunique(),
    )
    return adata


def _read_annotation_rows(
    path: Path,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Read the header plus the three fixed annotation rows, or fail loudly.

    Only the first four lines are consumed here (they carry quoted labels, so
    :mod:`csv` handles the quoting); the large gene block is read separately by
    :func:`_read_expression_matrix`.
    """
    rows: list[list[str]] = []
    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        for _ in range(_N_ANNOTATION_ROWS + 1):
            try:
                rows.append(next(reader))
            except StopIteration as exc:
                raise ValueError(
                    "Tirosh matrix has too few rows (expected a header plus "
                    f"{_N_ANNOTATION_ROWS} annotation rows before the gene rows)."
                ) from exc
    header, tumor_row, malignant_row, celltype_row = rows
    return header, tumor_row, malignant_row, celltype_row


def _read_expression_matrix(path: Path, n_cells: int) -> pd.DataFrame:
    """Read the gene x cell expression block (log2 scale) into a DataFrame.

    The four leading annotation lines are skipped by count; the gene symbol
    column becomes the index. Extra trailing columns (if any) are dropped to the
    ``n_cells`` declared in the header.
    """
    skiprows = 1 + _N_ANNOTATION_ROWS
    try:
        frame = pd.read_csv(path, sep="\t", skiprows=skiprows, header=None, index_col=0)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Tirosh matrix at {path} contains no gene rows.") from exc
    if frame.empty:
        raise ValueError(f"Tirosh matrix at {path} contains no gene rows.")
    if frame.shape[1] < n_cells:
        raise ValueError(
            f"Tirosh matrix at {path} has {frame.shape[1]} expression column(s) "
            f"but {n_cells} cells in the header."
        )
    frame = frame.iloc[:, :n_cells]
    frame.index = frame.index.astype(str)
    frame.index.name = "gene"
    return frame


def _validate_header_labels(
    cell_label: str, malignant_label: str, celltype_label: str
) -> None:
    """Guard against silent format drift in the annotation-row labels."""
    if cell_label.strip().lower() != "cell":
        raise ValueError(
            "Unexpected Tirosh header: first field should be 'Cell', got "
            f"{cell_label!r}."
        )
    if "malignant" not in malignant_label.lower():
        raise ValueError(f"Unexpected Tirosh malignant-row label: {malignant_label!r}.")
    if "cell type" not in celltype_label.lower():
        raise ValueError(f"Unexpected Tirosh cell-type-row label: {celltype_label!r}.")


def _decode_cell_types(malignant: list[int], celltype: list[int]) -> list[str | None]:
    """Decode per-cell (malignant, cell-type) codes into labels (None = unresolved)."""
    labels: list[str | None] = []
    for malignant_code, celltype_code in zip(malignant, celltype, strict=True):
        if malignant_code == _MALIGNANT_YES:
            labels.append(MALIGNANT_LABEL)
        elif malignant_code == _MALIGNANT_NO and celltype_code in CELL_TYPE_CODES:
            labels.append(CELL_TYPE_CODES[celltype_code])
        else:
            labels.append(None)
    return labels


def _deduplicate_genes(matrix: pd.DataFrame) -> pd.DataFrame:
    """Keep the first occurrence of each gene symbol, warning on duplicates."""
    if not matrix.index.has_duplicates:
        return matrix
    keep_mask = ~matrix.index.duplicated(keep="first")
    logger.warning(
        "Tirosh matrix has %d duplicate gene symbol(s); keeping the first "
        "occurrence of each.",
        int((~keep_mask).sum()),
    )
    return matrix.loc[keep_mask]
