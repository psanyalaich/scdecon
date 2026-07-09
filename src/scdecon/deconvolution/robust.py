r"""Robust non-negative regression deconvolution solver.

Optimisation method
-------------------
This solver estimates cell-type proportions ``p`` by solving a **non-negative,
robustly-weighted least-squares** problem:

    minimise   sum_g rho( (S @ p - b)_g )   subject to   p >= 0

where ``rho`` is a robust loss (``soft_l1`` by default, ``huber`` optional) applied
to the per-gene residuals. Compared with ordinary (L2) least squares, the robust
loss down-weights large residuals, so a handful of outlier genes (e.g. poorly
modelled or noisy) influence the fit far less. The problem is solved with
``scipy.optimize.least_squares`` (Trust Region Reflective), which supports both
bound constraints and robust losses; the exact Jacobian ``S`` is supplied.

Assumptions
-----------
- Linear mixing: bulk is approximately a non-negative linear combination of the
  cell-type signature profiles.
- ``S`` and ``b`` are on a comparable (linear) scale.
- Residual outliers (rather than dense Gaussian noise) are the main error mode
  the robust loss is meant to tolerate.

Configurable parameters
-----------------------
Via :class:`~scdecon.deconvolution.params.RobustConfig`: ``loss`` (``soft_l1`` or
``huber``) and ``f_scale`` (the residual scale below which points are inliers).

Post-processing
---------------
Non-negativity is enforced directly by the solver's bounds (``p >= 0``); the
solution is then renormalised to ``sum(p) == 1``.

Input / output orientation
--------------------------
``signature`` is ``(n_genes, n_cell_types)`` and ``bulk`` is ``(n_genes,)``,
gene-aligned. The returned vector is ``(n_cell_types,)`` with ``p >= 0`` and
``sum(p) == 1``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from scdecon.deconvolution.base import Solver
from scdecon.deconvolution.params import RobustConfig

__all__ = ["RobustSolver"]


class RobustSolver(Solver):
    """Deconvolution via robust non-negative least squares (see module docstring)."""

    def __init__(self, config: RobustConfig | None = None) -> None:
        self._config = config or RobustConfig()

    def fit(
        self, signature: NDArray[np.float64], bulk: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Estimate cell-type proportions for one bulk sample.

        Solves ``min sum_g rho((S @ p - b)_g)`` subject to ``p >= 0`` with a robust
        loss, then renormalises to ``sum(p) == 1``.

        Parameters
        ----------
        signature:
            Signature matrix ``S`` of shape ``(n_genes, n_cell_types)``.
        bulk:
            Bulk expression vector ``b`` of shape ``(n_genes,)``, gene-aligned to
            ``signature``.

        Returns
        -------
        numpy.ndarray
            Proportion vector of shape ``(n_cell_types,)`` with ``p >= 0`` and
            ``sum(p) == 1``. Neither input is modified.

        Raises
        ------
        ValueError
            If the bulk has no positive expression to fit, or the solution is all
            zero (nothing to normalise). Unlike NNLS/nu-SVR -- whose solvers
            return exact zeros for a degenerate bulk -- the trust-region optimiser
            converges to a tiny non-zero interior point, so the degenerate case is
            caught at the input (a bulk with no positive signal) rather than only
            at the solution.
        """
        # A bulk with no positive expression carries no signal to deconvolve.
        # This is an exact, scale-free condition (not a tolerance): valid bulk
        # expression is non-negative and has at least one positive gene.
        if not bool(np.any(bulk > 0)):
            raise ValueError(
                "Robust regression received a bulk sample with no positive "
                "expression, so proportions cannot be estimated. Ensure the bulk "
                "is on the same (linear, non-negative) scale as the signature and "
                "is gene-aligned to it."
            )
        n_cell_types = signature.shape[1]
        initial_guess = np.full(n_cell_types, 1.0 / n_cell_types, dtype=np.float64)
        result = least_squares(
            fun=lambda proportions: signature @ proportions - bulk,
            x0=initial_guess,
            jac=lambda proportions: signature,
            bounds=(0.0, np.inf),
            loss=str(self._config.loss),
            f_scale=self._config.f_scale,
        )
        coefficients = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, None)
        total = float(coefficients.sum())
        if total <= 0:
            raise ValueError(
                "Robust regression produced an all-zero solution that cannot be "
                "normalised to proportions summing to 1. Check that the bulk is on "
                "the same (linear) scale as the signature and that gene "
                "identifiers match."
            )
        return coefficients / total
