"""Fair, solver-agnostic benchmarking of deconvolution solvers.

Runs every supplied solver over a **single, once-aligned** pseudobulk set and
scores each with the **same** metrics, so comparisons are fair by construction:

- the signature and bulk are aligned **exactly once**; every solver receives the
  byte-identical aligned arrays (no per-solver preprocessing);
- the same ground truth and the same :func:`scdecon.validation.evaluate` are used
  for all solvers;
- the harness depends only on the abstract
  :class:`~scdecon.deconvolution.base.Solver` interface -- it never imports,
  references, or branches on any concrete solver. Adding a solver requires only
  passing another ``Solver`` instance in the mapping, with no change here.

Solver identities are the caller's mapping keys and are preserved **exactly**
(never renamed, normalised, or inferred) throughout ``reports``, ``runtimes``,
:meth:`BenchmarkResult.to_frame`, :meth:`BenchmarkResult.render`, and
:meth:`BenchmarkResult.best`.

.. note::
   Runtimes are single ``time.perf_counter`` measurements of the per-sample
   solving loop (shared alignment is excluded). They are **informational only**
   and must not be interpreted as rigorous performance benchmarks: they vary with
   hardware, operating system, and load.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from scdecon.deconvolution.align import DEFAULT_MIN_OVERLAP, align_signature_and_bulk
from scdecon.deconvolution.base import Solver
from scdecon.logging_utils import get_logger
from scdecon.validation import ValidationReport, evaluate

__all__ = ["BenchmarkResult", "run_benchmark"]

logger = get_logger("deconvolution.benchmark")

#: Metric column -> whether a higher value is better (used by ``best``).
_METRIC_HIGHER_IS_BETTER = {
    "overall_rmse": False,
    "mean_pearson": True,
    "mean_spearman": True,
    "runtime_s": False,
}
_METRIC_COLUMNS = ["overall_rmse", "mean_pearson", "mean_spearman", "runtime_s"]


@dataclass(frozen=True, eq=False)
class BenchmarkResult:
    """Comparison of solvers over one shared pseudobulk set.

    Composes the M5 :class:`~scdecon.validation.ValidationReport` per solver plus
    a runtime; introduces no second reporting abstraction. Solver names are
    exactly the caller-supplied mapping keys.

    Attributes
    ----------
    reports:
        Solver name -> :class:`~scdecon.validation.ValidationReport`.
    runtimes:
        Solver name -> wall-clock seconds for the solving loop (informational).
    """

    reports: Mapping[str, ValidationReport]
    runtimes: Mapping[str, float]

    def to_frame(self) -> pd.DataFrame:
        """Return the comparison table (index = solver names, exactly as given).

        Columns: ``overall_rmse``, ``mean_pearson``, ``mean_spearman``,
        ``runtime_s``.
        """
        names = list(self.reports)
        rows = [
            {
                "overall_rmse": self.reports[name].overall_rmse,
                "mean_pearson": self.reports[name].mean_pearson,
                "mean_spearman": self.reports[name].mean_spearman,
                "runtime_s": self.runtimes[name],
            }
            for name in names
        ]
        frame = pd.DataFrame(rows, index=names, columns=_METRIC_COLUMNS)
        frame.index.name = "solver"
        return frame

    def best(self, by: str = "overall_rmse") -> str:
        """Return the name of the best solver by a metric.

        Lower is better for ``overall_rmse`` and ``runtime_s``; higher is better
        for ``mean_pearson`` and ``mean_spearman``.

        Raises
        ------
        ValueError
            If ``by`` is not a known metric.
        """
        if by not in _METRIC_HIGHER_IS_BETTER:
            raise ValueError(
                f"Unknown metric '{by}'. Choose one of "
                f"{sorted(_METRIC_HIGHER_IS_BETTER)}."
            )
        column = self.to_frame()[by]
        if _METRIC_HIGHER_IS_BETTER[by]:
            return str(column.idxmax())
        return str(column.idxmin())

    def render(self) -> str:
        """Return a human-readable comparison table."""
        return "Benchmark results:\n" + self.to_frame().to_string()

    def __str__(self) -> str:
        return self.render()


def run_benchmark(
    signature: pd.DataFrame,
    bulk: pd.DataFrame,
    truth: pd.DataFrame,
    solvers: Mapping[str, Solver],
    *,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
) -> BenchmarkResult:
    """Benchmark ``solvers`` over one shared, once-aligned pseudobulk set.

    Parameters
    ----------
    signature:
        Signature matrix (genes x cell types), gene-indexed.
    bulk:
        Bulk expression matrix (genes x samples), gene-indexed.
    truth:
        Ground-truth proportions (cell types x samples).
    solvers:
        Mapping of caller-chosen name -> :class:`Solver`. Required; the harness
        never constructs solvers itself. Names are preserved exactly.
    min_overlap:
        Passed to :func:`~scdecon.deconvolution.align.align_signature_and_bulk`.

    Returns
    -------
    BenchmarkResult
        Per-solver validation report and runtime.

    Raises
    ------
    ValueError
        If ``solvers`` is empty, or the inputs cannot be aligned / evaluated.
    """
    if not solvers:
        raise ValueError("run_benchmark requires at least one solver.")

    # Align exactly once so every solver receives identical inputs (fairness).
    aligned = align_signature_and_bulk(signature, bulk, min_overlap=min_overlap)

    reports: dict[str, ValidationReport] = {}
    runtimes: dict[str, float] = {}
    for name, solver in solvers.items():
        start = time.perf_counter()
        estimates = np.empty(
            (len(aligned.cell_types), len(aligned.sample_names)), dtype=np.float64
        )
        for sample_index in range(aligned.bulk.shape[1]):
            estimates[:, sample_index] = solver.fit(
                aligned.signature, aligned.bulk[:, sample_index]
            )
        runtimes[name] = time.perf_counter() - start

        prediction = pd.DataFrame(
            estimates, index=aligned.cell_types, columns=aligned.sample_names
        )
        prediction.index.name = "cell_type"
        reports[name] = evaluate(truth, prediction)

    logger.info(
        "Benchmarked %d solvers over %d samples.",
        len(solvers),
        len(aligned.sample_names),
    )
    return BenchmarkResult(reports=reports, runtimes=runtimes)
