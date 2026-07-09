"""Configuration objects and typed enums for the deconvolution solvers.

Frozen, validated dataclasses (no magic numbers), mirroring the other layers'
config pattern. They import only the standard library, so the solver-core modules
that consume them stay format-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RobustLoss(StrEnum):
    """Robust loss for :class:`~scdecon.deconvolution.robust.RobustSolver`.

    Values are the loss names accepted by ``scipy.optimize.least_squares``.
    """

    SOFT_L1 = "soft_l1"
    HUBER = "huber"


@dataclass(frozen=True)
class NuSVRConfig:
    """Configuration for :class:`~scdecon.deconvolution.nusvr.NuSVRSolver`.

    Attributes
    ----------
    nu:
        The ``nu`` parameter of nu-SVR, an upper bound on the fraction of margin
        errors and a lower bound on the fraction of support vectors, in
        ``(0, 1]``.
    """

    nu: float = 0.5

    def __post_init__(self) -> None:
        """Validate parameters, failing loudly on nonsensical values."""
        if not 0.0 < self.nu <= 1.0:
            raise ValueError(f"nu must be in (0, 1], got {self.nu}")


@dataclass(frozen=True)
class RobustConfig:
    """Configuration for :class:`~scdecon.deconvolution.robust.RobustSolver`.

    Attributes
    ----------
    loss:
        Robust loss function (:class:`RobustLoss`); ``soft_l1`` by default.
    f_scale:
        The soft-margin scale of the robust loss (``scipy.optimize.least_squares``
        ``f_scale``): residuals below this scale are treated as inliers. Must be
        positive.
    """

    loss: RobustLoss = RobustLoss.SOFT_L1
    f_scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate parameters, failing loudly on nonsensical values."""
        if not isinstance(self.loss, RobustLoss):
            raise ValueError(
                f"loss must be a RobustLoss, got {type(self.loss).__name__}"
            )
        if self.f_scale <= 0:
            raise ValueError(f"f_scale must be > 0, got {self.f_scale}")
