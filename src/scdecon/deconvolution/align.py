"""Align a labelled signature matrix and bulk samples onto shared genes.

This module is the **adapter** between labelled (pandas) data and the
format-agnostic numerical solvers (:mod:`scdecon.deconvolution.base`): it
resolves the shared gene set, orders it by the signature's row order, and returns
plain NumPy arrays plus the labels needed to interpret them. Solvers themselves
never see gene labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from scdecon.logging_utils import get_logger

__all__ = ["AlignedInputs", "align_signature_and_bulk"]

#: Default minimum fraction of signature genes that must be present in the bulk
#: matrix before a low-overlap warning is emitted. Exposed as a parameter of
#: :func:`align_signature_and_bulk` so it is never hard-coded at call sites.
DEFAULT_MIN_OVERLAP = 0.5

logger = get_logger("deconvolution.align")


@dataclass(frozen=True, eq=False)
class AlignedInputs:
    """Gene-aligned deconvolution inputs, ready for a :class:`Solver`.

    Attributes
    ----------
    signature:
        Aligned signature matrix, shape ``(n_shared_genes, n_cell_types)``.
    bulk:
        Aligned bulk matrix, shape ``(n_shared_genes, n_samples)``.
    genes:
        Shared gene identifiers, in signature row order.
    cell_types:
        Cell-type labels (signature columns).
    sample_names:
        Bulk sample labels (bulk columns).
    """

    signature: NDArray[np.float64]
    bulk: NDArray[np.float64]
    genes: list[str]
    cell_types: list[str]
    sample_names: list[str]


def align_signature_and_bulk(
    signature: pd.DataFrame,
    bulk: pd.DataFrame,
    *,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
) -> AlignedInputs:
    """Restrict a signature and bulk matrix to their shared genes.

    Genes common to both are selected in **signature row order** (deterministic).
    A warning is emitted if the fraction of signature genes found in the bulk
    matrix falls below ``min_overlap``.

    Parameters
    ----------
    signature:
        Signature matrix (genes x cell types), gene-indexed.
    bulk:
        Bulk expression matrix (genes x samples), gene-indexed.
    min_overlap:
        Minimum fraction of signature genes that must be present in ``bulk``
        before a low-overlap warning is logged. In ``[0, 1]``.

    Returns
    -------
    AlignedInputs
        Aligned NumPy arrays plus gene / cell-type / sample labels.

    Raises
    ------
    ValueError
        If ``min_overlap`` is outside ``[0, 1]``, either matrix has duplicate
        gene identifiers, or the signature and bulk share no genes.
    """
    if not 0.0 <= min_overlap <= 1.0:
        raise ValueError(f"min_overlap must be in [0, 1], got {min_overlap}")
    if signature.index.has_duplicates:
        raise ValueError("Signature has duplicate gene identifiers.")
    if bulk.index.has_duplicates:
        raise ValueError("Bulk matrix has duplicate gene identifiers.")

    bulk_genes = set(bulk.index)
    shared = [str(gene) for gene in signature.index if gene in bulk_genes]
    if not shared:
        raise ValueError(
            "Signature and bulk share no genes. Check that both use the same "
            "gene identifiers (e.g. symbols vs Ensembl IDs) and the same "
            "orientation (genes as the index)."
        )

    n_signature_genes = signature.shape[0]
    overlap = len(shared) / n_signature_genes
    if overlap < min_overlap:
        logger.warning(
            "Low gene overlap: only %d of %d signature genes (%.1f%%) are present "
            "in the bulk matrix (below min_overlap=%.2f). Estimates may be "
            "unreliable; check gene identifiers.",
            len(shared),
            n_signature_genes,
            overlap * 100,
            min_overlap,
        )

    aligned_signature = signature.loc[shared].to_numpy(dtype=np.float64)
    aligned_bulk = bulk.loc[shared].to_numpy(dtype=np.float64)
    return AlignedInputs(
        signature=aligned_signature,
        bulk=aligned_bulk,
        genes=shared,
        cell_types=[str(column) for column in signature.columns],
        sample_names=[str(column) for column in bulk.columns],
    )
