"""Normalisation of single-cell AnnData objects.

Transforming the expression matrix is the explicit purpose of :func:`normalize`,
so it mutates ``adata`` in place. Raw counts are preserved in
``adata.layers[config.counts_layer]`` before normalisation so the un-normalised
data is never lost.
"""

from __future__ import annotations

import anndata
import scanpy as sc

from scdecon.logging_utils import get_logger
from scdecon.preprocessing.params import PreprocessConfig

__all__ = ["normalize"]

logger = get_logger("preprocessing.normalize")


def normalize(adata: anndata.AnnData, config: PreprocessConfig) -> anndata.AnnData:
    """Library-size normalise and log-transform in place, preserving raw counts.

    Raw counts are copied to ``adata.layers[config.counts_layer]`` before
    ``sc.pp.normalize_total`` and ``sc.pp.log1p`` are applied to ``adata.X``.

    Parameters
    ----------
    adata:
        Data whose ``.X`` holds raw counts. Modified in place.
    config:
        Supplies ``target_sum`` for normalisation and ``counts_layer`` for the
        raw-count backup.

    Returns
    -------
    anndata.AnnData
        The same ``adata`` instance, normalised and log-transformed, with raw
        counts stored in ``.layers[config.counts_layer]``.

    Raises
    ------
    ValueError
        If ``config.counts_layer`` already exists in ``adata.layers``. Its
        presence indicates normalisation has already run; re-running would stash
        already-normalised values as "raw counts", so the operation is refused
        rather than silently overwriting or double-normalising.
    """
    if config.counts_layer in adata.layers:
        raise ValueError(
            f"Layer '{config.counts_layer}' already exists; normalisation appears "
            "to have already run. Refusing to overwrite raw counts or "
            "double-normalise. Start from raw counts without this layer."
        )
    adata.layers[config.counts_layer] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=config.target_sum)
    sc.pp.log1p(adata)
    logger.info(
        "Normalised %d cells to target_sum=%s and applied log1p; "
        "raw counts preserved in layer '%s'.",
        adata.n_obs,
        config.target_sum,
        config.counts_layer,
    )
    return adata
