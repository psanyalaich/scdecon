"""Orchestrate deconvolution of a bulk matrix against a signature.

This is a thin, pandas-aware adapter: it aligns genes (via
:func:`scdecon.deconvolution.align.align_signature_and_bulk`), then delegates all
numerical work to a :class:`~scdecon.deconvolution.base.Solver` for each sample,
and labels the result. No mathematics lives here — only label handling and
iteration. It is intentionally **not** part of the format-agnostic solver core.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scdecon.deconvolution.align import DEFAULT_MIN_OVERLAP, align_signature_and_bulk
from scdecon.deconvolution.base import Solver
from scdecon.deconvolution.nnls import NNLSSolver
from scdecon.logging_utils import get_logger

__all__ = ["deconvolve"]

logger = get_logger("deconvolution.deconvolve")


def deconvolve(
    signature: pd.DataFrame,
    bulk: pd.DataFrame,
    solver: Solver | None = None,
    *,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
) -> pd.DataFrame:
    """Estimate cell-type proportions for every bulk sample.

    Aligns ``signature`` and ``bulk`` onto their shared genes, then runs
    ``solver.fit`` per bulk sample.

    Parameters
    ----------
    signature:
        Signature matrix (genes x cell types), gene-indexed.
    bulk:
        Bulk expression matrix (genes x samples), gene-indexed.
    solver:
        Solver to use. Defaults to :class:`NNLSSolver`.
    min_overlap:
        Minimum fraction of signature genes that must be present in ``bulk``
        before a low-overlap warning is logged (see
        :func:`align_signature_and_bulk`).

    Returns
    -------
    pandas.DataFrame
        Estimated proportions, **cell types (index) x samples (columns)**. Each
        column is non-negative and sums to 1.

    Raises
    ------
    ValueError
        Propagated from alignment (e.g. no shared genes) or from the solver
        (e.g. a degenerate all-zero solution for some sample).
    """
    solver = solver or NNLSSolver()
    aligned = align_signature_and_bulk(signature, bulk, min_overlap=min_overlap)

    estimates = np.empty(
        (len(aligned.cell_types), len(aligned.sample_names)), dtype=np.float64
    )
    for sample_index in range(aligned.bulk.shape[1]):
        # aligned.bulk[:, sample_index] is a column view (no copy); the solver
        # treats its inputs as read-only.
        estimates[:, sample_index] = solver.fit(
            aligned.signature, aligned.bulk[:, sample_index]
        )

    result = pd.DataFrame(
        estimates, index=aligned.cell_types, columns=aligned.sample_names
    )
    result.index.name = "cell_type"
    logger.info(
        "Deconvolved %d samples over %d cell types using %s.",
        len(aligned.sample_names),
        len(aligned.cell_types),
        type(solver).__name__,
    )
    return result
