"""Abstract solver interface for cell-type deconvolution.

Solvers are deliberately **format-agnostic**: they operate purely on NumPy arrays
and know nothing about gene labels, pandas, AnnData, Scanpy, plotting, or I/O.
Aligning a labelled signature matrix and a bulk sample onto shared genes is a
separate concern handled by :mod:`scdecon.deconvolution.align`; labelling the
estimated proportions is the caller's/orchestrator's job.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

__all__ = ["Solver"]


class Solver(ABC):
    """Abstract base class for deconvolution solvers.

    A solver estimates the cell-type proportion vector ``p`` for a single bulk
    sample by (approximately) solving ``b ~= S @ p`` subject to ``p >= 0`` and
    ``sum(p) == 1``. Implementations must enforce both constraints as part of
    their contract.
    """

    @abstractmethod
    def fit(
        self, signature: NDArray[np.float64], bulk: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Estimate cell-type proportions for one bulk sample.

        Parameters
        ----------
        signature:
            Signature matrix of shape ``(n_genes, n_cell_types)``. Rows are genes
            already aligned to ``bulk``; columns are cell types.
        bulk:
            Bulk expression vector of shape ``(n_genes,)``, gene-aligned to
            ``signature``.

        Returns
        -------
        numpy.ndarray
            Proportion vector of shape ``(n_cell_types,)`` with ``p >= 0`` and
            ``sum(p) == 1``.
        """
        raise NotImplementedError
