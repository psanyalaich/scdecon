"""Configuration and typed enums for pseudobulk simulation.

:class:`SimulationConfig` is the single source of truth for simulation
parameters, mirroring the other layers' frozen-dataclass configs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ProportionPrior(StrEnum):
    """Prior distribution used to draw per-sample cell-type proportions."""

    DIRICHLET = "dirichlet"
    UNIFORM = "uniform"


@dataclass(frozen=True)
class SimulationConfig:
    """Immutable configuration for pseudobulk simulation.

    Attributes
    ----------
    n_samples:
        Number of pseudobulk samples to generate.
    n_cells_per_sample:
        Number of single cells summed into each pseudobulk sample.
    cell_type_key:
        Column in ``adata.obs`` holding the cell-type annotation.
    proportion_prior:
        Prior for drawing target proportions (:class:`ProportionPrior`).
    dirichlet_alpha:
        Symmetric Dirichlet concentration (only used for the Dirichlet prior).
        ``1.0`` is uniform over the simplex; ``< 1`` favours sparser mixtures.
    random_state:
        Seed for the simulation's random generator (full reproducibility).
    counts_layer:
        Name of the ``adata.layers`` entry holding raw counts to sum. ``None``
        uses ``adata.X`` (expected to be raw counts).
    """

    n_samples: int = 100
    n_cells_per_sample: int = 500
    cell_type_key: str = "cell_type"
    proportion_prior: ProportionPrior = ProportionPrior.DIRICHLET
    dirichlet_alpha: float = 1.0
    random_state: int = 0
    counts_layer: str | None = None

    def __post_init__(self) -> None:
        """Validate parameters, failing loudly on nonsensical values."""
        if self.n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {self.n_samples}")
        if self.n_cells_per_sample < 1:
            raise ValueError(
                f"n_cells_per_sample must be >= 1, got {self.n_cells_per_sample}"
            )
        if self.dirichlet_alpha <= 0:
            raise ValueError(f"dirichlet_alpha must be > 0, got {self.dirichlet_alpha}")
        if not self.cell_type_key:
            raise ValueError("cell_type_key must be a non-empty string")
        if not isinstance(self.proportion_prior, ProportionPrior):
            raise ValueError(
                "proportion_prior must be a ProportionPrior, got "
                f"{type(self.proportion_prior).__name__}"
            )
        if self.counts_layer is not None and not self.counts_layer:
            raise ValueError("counts_layer must be None or a non-empty string")
