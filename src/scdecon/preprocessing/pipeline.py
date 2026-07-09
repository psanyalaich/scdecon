"""Orchestration of the single-cell preprocessing pipeline.

:func:`preprocess` chains the QC and normalisation steps into the single entry
point described by the project blueprint.
"""

from __future__ import annotations

import anndata

from scdecon.logging_utils import get_logger
from scdecon.preprocessing.normalize import normalize
from scdecon.preprocessing.params import PreprocessConfig
from scdecon.preprocessing.qc import compute_qc_metrics, filter_cells_and_genes

__all__ = ["QC_SUMMARY_KEY", "preprocess"]

#: ``.uns`` key under which the serialisable QC summary is stored. Namespaced to
#: avoid collisions with other tools' or future metadata.
QC_SUMMARY_KEY = "scdecon_qc_summary"

logger = get_logger("preprocessing.pipeline")


def preprocess(adata: anndata.AnnData, config: PreprocessConfig) -> anndata.AnnData:
    """Run the full preprocessing pipeline: QC metrics, filtering, normalisation.

    The supplied ``adata`` is annotated in place with QC metadata (matching
    Scanpy conventions and avoiding an unnecessary copy); callers who need the
    original untouched should pass ``adata.copy()``. Filtering then produces a
    new object, which is normalised and returned.

    QC metrics are always recomputed and overwritten deterministically -- if
    ``compute_qc_metrics`` has already been run on ``adata``, its columns are
    regenerated, not skipped -- so the result never depends on prior state.

    The QC summary is stored in serialisable form at
    ``result.uns[QC_SUMMARY_KEY]`` (a dict of integer counts) so it travels with
    the processed object (e.g. through ``write_h5ad``) without changing this
    function's return type.

    Parameters
    ----------
    adata:
        Raw-count single-cell data. Annotated in place with QC metadata; its
        expression values and dimensions are not modified.
    config:
        Thresholds and options for QC filtering and normalisation.

    Returns
    -------
    anndata.AnnData
        A new filtered and normalised dataset, with ``.uns[QC_SUMMARY_KEY]`` set
        and raw counts preserved in ``.layers[config.counts_layer]``.
    """
    compute_qc_metrics(adata, config)
    filtered, summary = filter_cells_and_genes(adata, config)
    normalize(filtered, config)
    filtered.uns[QC_SUMMARY_KEY] = summary.to_dict()
    logger.info(
        "Preprocessing complete: %d cells x %d genes.",
        filtered.n_obs,
        filtered.n_vars,
    )
    return filtered
