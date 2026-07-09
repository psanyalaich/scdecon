r"""nu-SVR deconvolution solver (CIBERSORT-style).

Optimisation method
-------------------
This solver estimates the cell-type proportions ``p`` for a single bulk sample by
**nu support-vector regression** of the bulk expression on the signature columns,
following the CIBERSORT approach. Treating each gene as a training example with
features given by the signature (genes x cell types) and target given by the bulk
vector, a linear nu-SVR learns weights ``w`` (one per cell type) that best predict

    b_g ~= sum_c w_c * S[g, c]

using an epsilon-insensitive, ``nu``-controlled support-vector objective. The
learned coefficients ``w`` are the (signed) unnormalised proportions.

Assumptions
-----------
- Linear mixing: bulk is approximately a non-negative linear combination of the
  cell-type signature profiles.
- ``S`` and ``b`` are on a comparable (linear) scale.
- The linear-kernel SVR coefficients are meaningful proportion weights; the SVR's
  robustness to outlier genes is the intended advantage over plain NNLS.

Configurable parameters
-----------------------
- ``nu`` (via :class:`~scdecon.deconvolution.params.NuSVRConfig`). The kernel is
  fixed to ``linear`` and ``C`` to ``1.0`` (standard, non-biological settings).

Post-processing
---------------
nu-SVR does not constrain the coefficients to be non-negative or to sum to one, so
we enforce the proportion constraints afterwards:

    p = clip(w, 0, inf);   p = p / sum(p)

Input / output orientation
--------------------------
``signature`` is ``(n_genes, n_cell_types)`` and ``bulk`` is ``(n_genes,)``,
gene-aligned. The returned vector is ``(n_cell_types,)`` with ``p >= 0`` and
``sum(p) == 1``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.svm import NuSVR

from scdecon.deconvolution.base import Solver
from scdecon.deconvolution.params import NuSVRConfig

__all__ = ["NuSVRSolver"]


class NuSVRSolver(Solver):
    """Deconvolution via linear nu support-vector regression (see module docstring)."""

    def __init__(self, config: NuSVRConfig | None = None) -> None:
        self._config = config or NuSVRConfig()

    def fit(
        self, signature: NDArray[np.float64], bulk: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Estimate cell-type proportions for one bulk sample.

        Fits a linear nu-SVR of ``bulk`` on ``signature``'s columns, then clips
        negative coefficients to zero and renormalises to ``sum(p) == 1``.

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
            If every coefficient is non-positive (nothing to normalise). Likely
            causes mirror :class:`~scdecon.deconvolution.nnls.NNLSSolver`:
            near-zero bulk on the signature genes, incompatible scales, or too
            little usable signal.
        """
        model = NuSVR(nu=self._config.nu, kernel="linear", C=1.0)
        model.fit(signature, bulk)
        coefficients = np.asarray(model.coef_, dtype=np.float64).ravel()
        clipped = np.clip(coefficients, 0.0, None)
        total = float(clipped.sum())
        if total <= 0:
            raise ValueError(
                "nu-SVR produced no positive coefficients, so the result cannot "
                "be normalised to proportions summing to 1. Check that the bulk "
                "is on the same (linear) scale as the signature and that gene "
                "identifiers match."
            )
        return clipped / total
