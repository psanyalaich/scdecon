"""Configuration and typed enums for signature construction.

:class:`SignatureConfig` is the single source of truth for marker-selection and
signature-building parameters, mirroring
:class:`~scdecon.preprocessing.params.PreprocessConfig`. The differential-
expression method is a :class:`RankMethod` enum rather than a free string, so
invalid methods are unrepresentable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RankMethod(StrEnum):
    """Differential-expression method for Scanpy ``rank_genes_groups``.

    ``StrEnum`` members are plain strings, so they pass directly to Scanpy and
    serialise as their value (e.g. ``"wilcoxon"``).
    """

    WILCOXON = "wilcoxon"
    T_TEST = "t-test"
    T_TEST_OVERESTIM_VAR = "t-test_overestim_var"
    LOGREG = "logreg"


@dataclass(frozen=True)
class SignatureConfig:
    """Immutable configuration for marker selection and signature construction.

    Attributes
    ----------
    cell_type_key:
        Column in ``adata.obs`` holding the cell-type annotation to group by.
    n_markers_per_type:
        Number of top-ranked marker genes to take per cell type before the
        cross-type specificity filter.
    method:
        Differential-expression ranking method (:class:`RankMethod`).
    min_cells_per_type:
        Minimum number of cells a cell type must have; below this, ranking is
        unreliable and selection fails loudly.
    """

    cell_type_key: str = "cell_type"
    n_markers_per_type: int = 25
    method: RankMethod = RankMethod.WILCOXON
    min_cells_per_type: int = 2

    def __post_init__(self) -> None:
        """Validate parameters, failing loudly on nonsensical values."""
        if self.n_markers_per_type < 1:
            raise ValueError(
                f"n_markers_per_type must be >= 1, got {self.n_markers_per_type}"
            )
        if self.min_cells_per_type < 1:
            raise ValueError(
                f"min_cells_per_type must be >= 1, got {self.min_cells_per_type}"
            )
        if not self.cell_type_key:
            raise ValueError("cell_type_key must be a non-empty string")
        if not isinstance(self.method, RankMethod):
            raise ValueError(
                f"method must be a RankMethod, got {type(self.method).__name__}"
            )
