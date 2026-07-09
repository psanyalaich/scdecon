"""Deconvolution: estimate cell-type proportions from bulk expression.

The solvers (:class:`~scdecon.deconvolution.base.Solver`, e.g.
:class:`~scdecon.deconvolution.nnls.NNLSSolver`,
:class:`~scdecon.deconvolution.nusvr.NuSVRSolver`,
:class:`~scdecon.deconvolution.robust.RobustSolver`) are format-agnostic and
operate purely on NumPy arrays. Aligning a labelled signature and bulk matrix onto
shared genes (:mod:`scdecon.deconvolution.align`) and orchestrating per-sample
solving into a labelled result
(:func:`~scdecon.deconvolution.deconvolve.deconvolve`) are separate, pandas-aware
concerns.
"""

from scdecon.deconvolution.align import AlignedInputs, align_signature_and_bulk
from scdecon.deconvolution.base import Solver
from scdecon.deconvolution.benchmark import BenchmarkResult, run_benchmark
from scdecon.deconvolution.deconvolve import deconvolve
from scdecon.deconvolution.nnls import NNLSSolver
from scdecon.deconvolution.nusvr import NuSVRSolver
from scdecon.deconvolution.params import NuSVRConfig, RobustConfig, RobustLoss
from scdecon.deconvolution.robust import RobustSolver

__all__ = [
    "AlignedInputs",
    "BenchmarkResult",
    "NNLSSolver",
    "NuSVRConfig",
    "NuSVRSolver",
    "RobustConfig",
    "RobustLoss",
    "RobustSolver",
    "Solver",
    "align_signature_and_bulk",
    "deconvolve",
    "run_benchmark",
]
