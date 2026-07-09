r"""Accuracy metrics for deconvolution against known proportions.

Compares an estimated proportion matrix against ground truth, both oriented as
**cell types (index) x samples (columns)** — the orientation produced by
:func:`scdecon.deconvolution.deconvolve` and
:class:`scdecon.simulation.PseudobulkDataset`.

Mathematics
-----------
For aligned truth ``T`` and prediction ``P`` (both ``C`` cell types x ``S``
samples), with residual ``R = P - T``:

- overall RMSE          = sqrt( mean over all (c, s) of R[c, s]^2 )
- per-cell-type RMSE_c  = sqrt( mean over samples of R[c, s]^2 )
- per-cell-type Pearson = linear correlation of T[c, :] and P[c, :] across samples
- per-cell-type Spearman = rank correlation of T[c, :] and P[c, :]

Correlations require at least two samples and non-constant vectors; otherwise the
metric is ``nan`` (reported, not raised).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from scdecon.logging_utils import get_logger

__all__ = ["ValidationReport", "align_proportions", "evaluate"]

logger = get_logger("validation.metrics")


@dataclass(frozen=True, eq=False)
class ValidationReport:
    """Typed, lightweight result of a validation run.

    Attributes
    ----------
    overall_rmse:
        Root-mean-square error across all cell-type x sample entries.
    per_type:
        One row per cell type (index), with columns ``rmse``, ``pearson``,
        ``spearman`` (the last two computed across samples; ``nan`` when a
        correlation is undefined).
    """

    overall_rmse: float
    per_type: pd.DataFrame

    def to_frame(self) -> pd.DataFrame:
        """Return the per-cell-type metric table (a copy), for serialisation."""
        return self.per_type.copy()

    @property
    def mean_pearson(self) -> float:
        """nan-aware mean of the per-cell-type Pearson correlations."""
        return float(np.nanmean(self.per_type["pearson"].to_numpy()))

    @property
    def mean_spearman(self) -> float:
        """nan-aware mean of the per-cell-type Spearman correlations."""
        return float(np.nanmean(self.per_type["spearman"].to_numpy()))

    def render(self) -> str:
        """Return a one-line human-readable summary."""
        return (
            f"Validation: overall RMSE={self.overall_rmse:.4f}, "
            f"mean Pearson={self.mean_pearson:.4f}, "
            f"mean Spearman={self.mean_spearman:.4f} "
            f"across {len(self.per_type)} cell types."
        )

    def __str__(self) -> str:
        return self.render()


def align_proportions(
    truth: pd.DataFrame, prediction: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check that truth and prediction describe the same data and align them.

    Both must be **cell types (index) x samples (columns)** with identical
    cell-type and sample label sets (order-independent). The prediction is
    reindexed to the truth's cell-type and sample order.

    Parameters
    ----------
    truth:
        Ground-truth proportions (cell types x samples).
    prediction:
        Estimated proportions (cell types x samples).

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        ``(truth, aligned_prediction)`` with matching index and columns.

    Raises
    ------
    ValueError
        If either frame is empty, has duplicate cell-type or sample labels, or
        the cell-type / sample label sets differ.
    """
    if truth.empty or prediction.empty:
        raise ValueError(
            "truth and prediction must be non-empty (cell types x samples)."
        )
    if truth.index.has_duplicates or prediction.index.has_duplicates:
        raise ValueError("Duplicate cell-type labels in truth or prediction index.")
    if truth.columns.has_duplicates or prediction.columns.has_duplicates:
        raise ValueError("Duplicate sample labels in truth or prediction columns.")

    if set(truth.index) != set(prediction.index):
        only_truth = sorted(set(truth.index) - set(prediction.index))
        only_prediction = sorted(set(prediction.index) - set(truth.index))
        raise ValueError(
            "truth and prediction cell types differ: only in truth "
            f"{only_truth}, only in prediction {only_prediction}."
        )
    if set(truth.columns) != set(prediction.columns):
        only_truth = sorted(set(truth.columns) - set(prediction.columns))
        only_prediction = sorted(set(prediction.columns) - set(truth.columns))
        raise ValueError(
            "truth and prediction samples differ: only in truth "
            f"{only_truth}, only in prediction {only_prediction}."
        )

    aligned_prediction = prediction.reindex(index=truth.index, columns=truth.columns)
    return truth, aligned_prediction


def evaluate(truth: pd.DataFrame, prediction: pd.DataFrame) -> ValidationReport:
    """Score estimated proportions against ground truth.

    Parameters
    ----------
    truth:
        Ground-truth proportions, **cell types (index) x samples (columns)**.
    prediction:
        Estimated proportions, same orientation and labels as ``truth``.

    Returns
    -------
    ValidationReport
        Overall RMSE and per-cell-type RMSE / Pearson / Spearman.

    Raises
    ------
    ValueError
        If the inputs cannot be aligned (see :func:`align_proportions`).
    """
    truth_aligned, prediction_aligned = align_proportions(truth, prediction)
    residual = prediction_aligned.to_numpy() - truth_aligned.to_numpy()
    overall_rmse = float(np.sqrt(np.mean(np.square(residual))))

    records: list[tuple[str, float, float, float]] = []
    for cell_type in truth_aligned.index:
        true_row = truth_aligned.loc[cell_type].to_numpy()
        predicted_row = prediction_aligned.loc[cell_type].to_numpy()
        rmse = float(np.sqrt(np.mean(np.square(predicted_row - true_row))))
        pearson = _safe_correlation(true_row, predicted_row, spearman=False)
        spearman = _safe_correlation(true_row, predicted_row, spearman=True)
        records.append((str(cell_type), rmse, pearson, spearman))

    per_type = pd.DataFrame(
        records, columns=["cell_type", "rmse", "pearson", "spearman"]
    ).set_index("cell_type")
    logger.info(
        "Validated %d cell types x %d samples: overall RMSE=%.4f.",
        per_type.shape[0],
        truth_aligned.shape[1],
        overall_rmse,
    )
    return ValidationReport(overall_rmse=overall_rmse, per_type=per_type)


def _safe_correlation(
    true_row: np.ndarray, predicted_row: np.ndarray, *, spearman: bool
) -> float:
    """Correlation across samples; ``nan`` when undefined (constant / < 2 points)."""
    if len(true_row) < 2 or np.ptp(true_row) == 0 or np.ptp(predicted_row) == 0:
        return float("nan")
    if spearman:
        return float(spearmanr(true_row, predicted_row)[0])
    return float(pearsonr(true_row, predicted_row)[0])
