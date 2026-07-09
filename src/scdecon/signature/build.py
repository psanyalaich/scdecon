"""Signature-matrix assembly from marker genes and a reference AnnData.

:func:`build_signature` computes each cell type's **linear-scale** mean expression
over the selected marker genes and returns a validated ``pandas.DataFrame``
(genes x cell types). Linear scale is essential: the deconvolution model
``b ~= S @ p`` is additive, which only holds in linear (not log) space -- so the
profile is the mean of ``expm1(X)``, even though marker *selection* ranks on the
log-normalised data.

The returned frame obeys a documented contract (see :func:`build_signature`),
enforced by :func:`_validate_signature_frame`. That validator is intentionally
private: it currently has a single consumer. When a second consumer appears
(e.g. reading a signature back from disk in a later milestone) it can be promoted
to a shared validation utility with the same contract; promoting it now would
expand the public API ahead of need.
"""

from __future__ import annotations

from typing import Any

import anndata
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas.api.types import is_numeric_dtype

from scdecon.logging_utils import get_logger
from scdecon.signature.markers import MarkerSet
from scdecon.signature.params import SignatureConfig

__all__ = ["build_signature"]

logger = get_logger("signature.build")


def build_signature(
    adata: anndata.AnnData, markers: MarkerSet, config: SignatureConfig
) -> pd.DataFrame:
    """Build a signature matrix of linear-scale mean profiles.

    The signature frame obeys this contract:

    - **index** = the marker genes, in **exactly** ``markers.genes()`` order.
      This ordering is part of the public contract and is reproducible.
    - **columns** = the reference cell types, sorted deterministically.
    - **values** = per-cell-type mean of ``expm1(adata.X)`` over the marker
      genes (linear scale); finite and non-negative.

    Parameters
    ----------
    adata:
        Log-normalised reference data (as produced by
        :mod:`scdecon.preprocessing`). Not modified.
    markers:
        Marker genes per cell type; ``markers.genes()`` sets the row order.
    config:
        Supplies ``cell_type_key`` (the ``obs`` column to group cells by).

    Returns
    -------
    pandas.DataFrame
        Genes (index, ``markers.genes()`` order) by cell types (sorted columns),
        holding linear-scale mean expression.

    Raises
    ------
    ValueError
        If ``cell_type_key`` is absent, ``adata.var_names`` has duplicates, a
        marker gene is missing from ``adata``, ``markers`` references a cell type
        absent from ``adata``, or the resulting matrix violates the frame
        contract.
    """
    genes = markers.genes()
    _validate_inputs(adata, markers, config, genes)

    cell_types = sorted({str(c) for c in adata.obs[config.cell_type_key].unique()})
    labels = adata.obs[config.cell_type_key].astype(str).to_numpy()
    linear = np.expm1(_dense(adata[:, genes].X))

    signature = pd.DataFrame(
        index=pd.Index(genes, name="gene"), columns=cell_types, dtype=float
    )
    for cell_type in cell_types:
        mask = labels == cell_type
        signature[cell_type] = linear[mask].mean(axis=0)

    _validate_signature_frame(signature)
    logger.info(
        "Built signature matrix: %d genes x %d cell types.",
        signature.shape[0],
        signature.shape[1],
    )
    return signature


def _dense(matrix: Any) -> NDArray[np.float64]:
    """Return a dense float array from a possibly-sparse AnnData ``.X``."""
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray(), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)


def _validate_inputs(
    adata: anndata.AnnData,
    markers: MarkerSet,
    config: SignatureConfig,
    genes: list[str],
) -> None:
    if config.cell_type_key not in adata.obs.columns:
        raise ValueError(
            f"cell_type_key '{config.cell_type_key}' not found in adata.obs."
        )
    if adata.var_names.has_duplicates:
        raise ValueError(
            "adata.var_names contains duplicates; gene identifiers must be unique."
        )
    known_genes = set(adata.var_names)
    missing = [gene for gene in genes if gene not in known_genes]
    if missing:
        raise ValueError(f"Marker genes not found in adata.var_names: {missing}.")
    adata_types = {str(c) for c in adata.obs[config.cell_type_key].unique()}
    unknown = sorted(set(markers.per_type) - adata_types)
    if unknown:
        raise ValueError(
            f"MarkerSet references cell types absent from adata: {unknown}."
        )


def _validate_signature_frame(frame: pd.DataFrame) -> None:
    """Enforce the signature-frame contract, failing loudly on violations."""
    if frame.empty:
        raise ValueError("Signature matrix is empty.")
    if frame.index.has_duplicates:
        duplicates = sorted(set(frame.index[frame.index.duplicated()]))
        raise ValueError(
            f"Signature matrix has duplicate gene identifiers: {duplicates}."
        )
    non_numeric = [str(c) for c in frame.columns if not is_numeric_dtype(frame[c])]
    if non_numeric:
        raise ValueError(f"Signature matrix has non-numeric columns: {non_numeric}.")
    values = frame.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Signature matrix contains non-finite values.")
    if bool((values < 0).any()):
        raise ValueError("Signature matrix contains negative values.")
