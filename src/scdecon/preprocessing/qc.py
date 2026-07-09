"""Quality-control metrics and filtering for single-cell AnnData objects.

Two responsibilities, kept separate:

- :func:`compute_qc_metrics` annotates QC metadata **in place** (``.obs``/``.var``
  only). It never touches the expression matrix, layers, names, or dimensions.
- :func:`filter_cells_and_genes` returns a **new, filtered** AnnData and leaves
  the input untouched, because changing dataset dimensions is a genuine
  transformation.

Every threshold and choice originates from
:class:`~scdecon.preprocessing.params.PreprocessConfig`; the only fixed values
are the explicitly-passed, non-biological Scanpy options.
"""

from __future__ import annotations

from dataclasses import dataclass

import anndata
import scanpy as sc

from scdecon.logging_utils import get_logger
from scdecon.preprocessing.params import PreprocessConfig

__all__ = ["QCSummary", "compute_qc_metrics", "filter_cells_and_genes"]

# Scanpy-standard metadata column names (structural identifiers, not thresholds).
_MITO_FLAG = "mt"
_PCT_MITO = "pct_counts_mt"

logger = get_logger("preprocessing.qc")


@dataclass(frozen=True)
class QCSummary:
    """Typed, immutable record of a QC filtering step.

    Attributes
    ----------
    n_cells_before, n_cells_after:
        Cell counts before and after filtering.
    n_genes_before, n_genes_after:
        Gene counts before and after filtering.
    n_cells_removed_by_min_genes:
        Cells dropped for expressing fewer than ``min_genes`` genes.
    n_cells_removed_by_max_pct_mito:
        Cells dropped for exceeding ``max_pct_mito`` mitochondrial percentage.
    n_genes_removed_by_min_cells:
        Genes dropped for being detected in fewer than ``min_cells`` cells.
    """

    n_cells_before: int
    n_cells_after: int
    n_genes_before: int
    n_genes_after: int
    n_cells_removed_by_min_genes: int
    n_cells_removed_by_max_pct_mito: int
    n_genes_removed_by_min_cells: int

    def __post_init__(self) -> None:
        """Validate that the counts reconcile, failing loudly otherwise."""
        counts = [
            self.n_cells_before,
            self.n_cells_after,
            self.n_genes_before,
            self.n_genes_after,
            self.n_cells_removed_by_min_genes,
            self.n_cells_removed_by_max_pct_mito,
            self.n_genes_removed_by_min_cells,
        ]
        if any(value < 0 for value in counts):
            raise ValueError(f"QCSummary counts must be non-negative, got {counts}")
        expected_cells = (
            self.n_cells_before
            - self.n_cells_removed_by_min_genes
            - self.n_cells_removed_by_max_pct_mito
        )
        if self.n_cells_after != expected_cells:
            raise ValueError(
                "QCSummary cell counts do not reconcile: "
                f"{self.n_cells_before} - {self.n_cells_removed_by_min_genes} - "
                f"{self.n_cells_removed_by_max_pct_mito} != {self.n_cells_after}"
            )
        expected_genes = self.n_genes_before - self.n_genes_removed_by_min_cells
        if self.n_genes_after != expected_genes:
            raise ValueError(
                "QCSummary gene counts do not reconcile: "
                f"{self.n_genes_before} - {self.n_genes_removed_by_min_cells} "
                f"!= {self.n_genes_after}"
            )

    @property
    def cells_removed(self) -> int:
        """Total cells removed across all cell filters."""
        return self.n_cells_before - self.n_cells_after

    @property
    def genes_removed(self) -> int:
        """Total genes removed."""
        return self.n_genes_before - self.n_genes_after

    def to_dict(self) -> dict[str, int]:
        """Return a plain-``int`` dict, suitable for serialisation into ``.uns``."""
        return {
            "n_cells_before": int(self.n_cells_before),
            "n_cells_after": int(self.n_cells_after),
            "n_genes_before": int(self.n_genes_before),
            "n_genes_after": int(self.n_genes_after),
            "n_cells_removed_by_min_genes": int(self.n_cells_removed_by_min_genes),
            "n_cells_removed_by_max_pct_mito": int(
                self.n_cells_removed_by_max_pct_mito
            ),
            "n_genes_removed_by_min_cells": int(self.n_genes_removed_by_min_cells),
        }

    def render(self) -> str:
        """Return a multi-line, human-readable summary."""
        return (
            "QC summary:\n"
            f"  cells: {self.n_cells_before} -> {self.n_cells_after} "
            f"(removed {self.cells_removed}: "
            f"{self.n_cells_removed_by_min_genes} by min_genes, "
            f"{self.n_cells_removed_by_max_pct_mito} by max_pct_mito)\n"
            f"  genes: {self.n_genes_before} -> {self.n_genes_after} "
            f"(removed {self.genes_removed} by min_cells)"
        )

    def __str__(self) -> str:
        return self.render()


def compute_qc_metrics(
    adata: anndata.AnnData, config: PreprocessConfig
) -> anndata.AnnData:
    """Annotate per-cell and per-gene QC metrics in place.

    Flags mitochondrial genes by ``config.mito_prefix`` and delegates metric
    computation to ``sc.pp.calculate_qc_metrics``. No filtering is performed.

    Parameters
    ----------
    adata:
        The dataset to annotate. Modified in place (``.obs``/``.var`` only).
    config:
        Supplies ``mito_prefix`` for identifying mitochondrial genes.

    Returns
    -------
    anndata.AnnData
        The same ``adata`` instance, with QC columns added. In particular
        ``.var["mt"]`` and ``.obs["pct_counts_mt"]``.
    """
    adata.var[_MITO_FLAG] = adata.var_names.str.startswith(config.mito_prefix)
    n_mito = int(adata.var[_MITO_FLAG].sum())
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=[_MITO_FLAG],
        percent_top=None,
        log1p=False,
        inplace=True,
    )
    logger.info(
        "Computed QC metrics for %d cells x %d genes "
        "(%d mitochondrial genes, prefix %r).",
        adata.n_obs,
        adata.n_vars,
        n_mito,
        config.mito_prefix,
    )
    return adata


def filter_cells_and_genes(
    adata: anndata.AnnData, config: PreprocessConfig
) -> tuple[anndata.AnnData, QCSummary]:
    """Filter low-quality cells and rarely-detected genes into a new AnnData.

    Applies, in this order: ``min_genes`` per cell, ``max_pct_mito`` per cell,
    then ``min_cells`` per gene (gene support counted after cell filtering).

    Parameters
    ----------
    adata:
        A dataset already annotated by :func:`compute_qc_metrics`. Not modified.
    config:
        Supplies ``min_genes``, ``max_pct_mito``, and ``min_cells``.

    Returns
    -------
    tuple[anndata.AnnData, QCSummary]
        The filtered copy and a summary of what was removed.

    Raises
    ------
    ValueError
        If QC metrics are absent (``obs['pct_counts_mt']`` missing).
    """
    if _PCT_MITO not in adata.obs.columns:
        raise ValueError(
            f"filter_cells_and_genes requires QC metrics; obs['{_PCT_MITO}'] is "
            "missing. Run compute_qc_metrics first."
        )

    work = adata.copy()
    n_cells_before = work.n_obs
    n_genes_before = work.n_vars

    sc.pp.filter_cells(work, min_genes=config.min_genes)
    removed_min_genes = n_cells_before - work.n_obs

    cells_before_mito = work.n_obs
    keep = work.obs[_PCT_MITO] <= config.max_pct_mito
    work = work[keep].copy()
    removed_max_pct_mito = cells_before_mito - work.n_obs

    sc.pp.filter_genes(work, min_cells=config.min_cells)
    removed_min_cells = n_genes_before - work.n_vars

    summary = QCSummary(
        n_cells_before=n_cells_before,
        n_cells_after=work.n_obs,
        n_genes_before=n_genes_before,
        n_genes_after=work.n_vars,
        n_cells_removed_by_min_genes=removed_min_genes,
        n_cells_removed_by_max_pct_mito=removed_max_pct_mito,
        n_genes_removed_by_min_cells=removed_min_cells,
    )
    logger.info("Filtered single-cell data.\n%s", summary.render())
    return work, summary
