"""Single-cell preprocessing: QC and normalisation of in-memory AnnData objects.

This layer operates purely on AnnData objects already loaded by
:mod:`scdecon.io`; it performs no file I/O of its own.
"""

from scdecon.preprocessing.normalize import normalize
from scdecon.preprocessing.params import PreprocessConfig
from scdecon.preprocessing.pipeline import preprocess
from scdecon.preprocessing.qc import (
    QCSummary,
    compute_qc_metrics,
    filter_cells_and_genes,
)

__all__ = [
    "PreprocessConfig",
    "QCSummary",
    "compute_qc_metrics",
    "filter_cells_and_genes",
    "normalize",
    "preprocess",
]
