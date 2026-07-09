"""Validation: accuracy metrics for deconvolution against known proportions.

A pure computation layer (NumPy / pandas / scipy.stats) operating on
proportion ``DataFrame`` objects oriented as cell types x samples. It imports no
I/O or plotting.
"""

from scdecon.validation.metrics import ValidationReport, align_proportions, evaluate

__all__ = ["ValidationReport", "align_proportions", "evaluate"]
