r"""Non-negative least squares (NNLS) deconvolution solver.

Mathematics
-----------
For one bulk sample we model the expression vector ``b`` (over genes) as a
non-negative mixture of cell-type profiles::

    b ≈ S · p

where ``S`` is the signature matrix (genes × cell types) and ``p`` is the unknown
cell-type proportion vector. Proportions must satisfy two constraints:

    p ≥ 0            (a cell type cannot contribute negatively)
    Σ p = 1          (proportions of a sample sum to one)

This solver estimates ``p`` in two steps:

1. **Non-negative least squares.** Solve

       minimise ‖ S · x − b ‖₂   subject to   x ≥ 0

   This enforces the non-negativity constraint and finds the mixture whose
   predicted expression is closest (in Euclidean distance) to the observed bulk.

2. **Normalisation.** NNLS does not constrain the sum of ``x``, so we renormalise

       p = x / Σ x

   which imposes ``Σ p = 1`` while preserving the relative magnitudes. Because the
   generative model is linear and additive, ``x`` scales linearly with the bulk
   magnitude, so normalisation removes any overall scale and yields proportions.

About ``scipy.optimize.nnls``
-----------------------------
``scipy.optimize.nnls(A, b)`` solves ``argmin_x ‖A x − b‖₂`` subject to
``x ≥ 0`` using the Lawson–Hanson active-set algorithm, and returns the solution
``x`` together with the residual 2-norm. It is:

- **Constrained** exactly to ``x ≥ 0`` (no sum constraint — handled here).
- **Deterministic** and hyper-parameter-free (no learning rate, seed, or kernel),
  which makes results reproducible and easy to test against analytic ground
  truth.
- **Exact on noise-free, well-conditioned data**: if ``b = S · p`` for some
  ``p ≥ 0`` and ``S`` has full column rank, NNLS recovers ``p`` exactly.

Why NNLS is appropriate for this project
----------------------------------------
Cell-type deconvolution is precisely a non-negativity-constrained linear inverse
problem, and proportions are inherently non-negative. NNLS is the simplest
principled solver for it: no hyper-parameters to tune, deterministic, and it
provides an exact-recovery baseline against which the more elaborate solvers
(ν-SVR, robust regression in M6) can be benchmarked honestly.

Known limitations
-----------------
- NNLS minimises the **L2** residual and is therefore sensitive to outlier genes
  and heteroscedastic noise; robust/weighted variants (M6) address this.
- The sum-to-one constraint is imposed by **post-hoc normalisation**, not solved
  jointly; this assumes the un-normalised solution's scale is meaningful.
- Accuracy degrades if the signature columns are **collinear / ill-conditioned**
  (i.e. cell types are hard to distinguish); this is mitigated upstream by
  choosing specific marker genes (see the signature layer).
- ``S`` and ``b`` must be on a **comparable linear scale** (the signature is built
  as linear-scale means for exactly this reason).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls

from scdecon.deconvolution.base import Solver

__all__ = ["NNLSSolver"]


class NNLSSolver(Solver):
    """Deconvolution via non-negative least squares (see module docstring)."""

    def fit(
        self, signature: NDArray[np.float64], bulk: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Estimate cell-type proportions for one bulk sample.

        Solves ``min ‖S x − b‖₂`` subject to ``x ≥ 0`` (``scipy.optimize.nnls``),
        then normalises to ``Σ p = 1``.

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
            ``sum(p) == 1``. Freshly allocated by the normalisation step; neither
            ``signature`` nor ``bulk`` is copied or modified.

        Raises
        ------
        ValueError
            If NNLS returns the all-zero solution (its coefficients sum to zero),
            which cannot be normalised into proportions. This typically means the
            bulk sample has ~zero expression on the signature genes, the signature
            and bulk are on incompatible scales, or gene alignment left almost no
            usable signal. Check that the bulk was preprocessed on the same scale
            as the signature and that gene identifiers match.
        """
        raw_coefficients, _residual = nnls(signature, bulk)
        # np.asarray with a matching dtype is a no-op view (no copy) and fixes the
        # static type, since scipy.optimize.nnls is untyped.
        coefficients = np.asarray(raw_coefficients, dtype=np.float64)
        # NNLS guarantees coefficients >= 0, so a non-positive total can only be
        # the all-zero solution: degenerate and impossible to turn into
        # proportions. This is an exact condition, not a tolerance/epsilon.
        total = float(coefficients.sum())
        if total <= 0:
            raise ValueError(
                "NNLS returned an all-zero solution that cannot be normalised to "
                "proportions summing to 1. Likely causes: the bulk sample has "
                "near-zero expression across the signature genes, the signature "
                "and bulk are on incompatible scales, or gene alignment left too "
                "little signal. Check that the bulk is preprocessed on the same "
                "(linear) scale as the signature and that gene identifiers match."
            )
        return coefficients / total
