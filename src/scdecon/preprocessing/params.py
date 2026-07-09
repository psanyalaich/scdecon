"""Parameters controlling single-cell preprocessing.

:class:`PreprocessConfig` is the single source of truth for every biological
threshold and choice used during QC and normalisation. The implementation reads
all such values from here, so there are no hidden constants in the preprocessing
code. It is a plain frozen dataclass; when the Pydantic settings layer arrives in
a later milestone it will construct or adapt to this object rather than replace
it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessConfig:
    """Immutable configuration for QC filtering and normalisation.

    Attributes
    ----------
    min_genes:
        Minimum number of genes a cell must express to be kept
        (``sc.pp.filter_cells(min_genes=...)``).
    min_cells:
        Minimum number of cells a gene must be detected in to be kept
        (``sc.pp.filter_genes(min_cells=...)``).
    max_pct_mito:
        Maximum percentage of counts from mitochondrial genes allowed per cell,
        in the range ``[0, 100]``. Cells above this are filtered out.
    mito_prefix:
        Gene-name prefix identifying mitochondrial genes (e.g. ``"MT-"`` for
        human symbols). Used to build the mito flag for QC metrics.
    target_sum:
        Per-cell total count after library-size normalisation
        (``sc.pp.normalize_total(target_sum=...)``). ``None`` uses Scanpy's
        median-count behaviour.
    counts_layer:
        Name of the ``adata.layers`` entry where raw counts are stored before
        normalisation, so the un-normalised data is never lost.
    """

    min_genes: int = 200
    min_cells: int = 3
    max_pct_mito: float = 20.0
    mito_prefix: str = "MT-"
    target_sum: float | None = 1e4
    counts_layer: str = "counts"

    def __post_init__(self) -> None:
        """Validate parameters, failing loudly on nonsensical values."""
        if self.min_genes < 0:
            raise ValueError(f"min_genes must be >= 0, got {self.min_genes}")
        if self.min_cells < 0:
            raise ValueError(f"min_cells must be >= 0, got {self.min_cells}")
        if not 0.0 <= self.max_pct_mito <= 100.0:
            raise ValueError(
                f"max_pct_mito must be in [0, 100], got {self.max_pct_mito}"
            )
        if self.target_sum is not None and self.target_sum <= 0:
            raise ValueError(f"target_sum must be > 0 or None, got {self.target_sum}")
        if not self.mito_prefix:
            raise ValueError("mito_prefix must be a non-empty string")
        if not self.counts_layer:
            raise ValueError("counts_layer must be a non-empty string")
