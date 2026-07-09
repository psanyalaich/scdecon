"""Deconvolution: estimate cell-type proportions from bulk expression.

The solvers (:class:`~scdecon.deconvolution.base.Solver`, e.g.
:class:`~scdecon.deconvolution.nnls.NNLSSolver`) are format-agnostic and operate
purely on NumPy arrays. Aligning a labelled signature and bulk matrix onto shared
genes (:mod:`scdecon.deconvolution.align`) and orchestrating per-sample solving
into a labelled result (:func:`~scdecon.deconvolution.deconvolve.deconvolve`) are
separate, pandas-aware concerns.
"""

from scdecon.deconvolution.align import AlignedInputs, align_signature_and_bulk
from scdecon.deconvolution.base import Solver
from scdecon.deconvolution.deconvolve import deconvolve
from scdecon.deconvolution.nnls import NNLSSolver

__all__ = [
    "AlignedInputs",
    "NNLSSolver",
    "Solver",
    "align_signature_and_bulk",
    "deconvolve",
]
