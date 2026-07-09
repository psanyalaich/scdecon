"""Figures rendered from scdecon results.

Plotting consumes the computational core's outputs (e.g. a signature
``DataFrame``) but the core never depends on plotting: matplotlib and seaborn
are imported only here. Figures are drawn with matplotlib's object-oriented
``Figure`` API (no global ``pyplot`` state), so rendering is headless-safe.
"""

from __future__ import annotations

from math import ceil
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from scdecon.deconvolution import BenchmarkResult
from scdecon.logging_utils import get_logger
from scdecon.validation import align_proportions

__all__ = [
    "plot_benchmark",
    "plot_signature_heatmap",
    "plot_truth_vs_prediction",
]

logger = get_logger("plotting.figures")


def plot_signature_heatmap(
    signature: pd.DataFrame, path: str | Path, *, title: str | None = None
) -> Path:
    """Render a genes-by-cell-types signature heatmap and save it to ``path``.

    Parameters
    ----------
    signature:
        Signature matrix (genes x cell types), as returned by
        :func:`scdecon.signature.build_signature`. Not modified.
    path:
        Destination image path (e.g. ``.png``). Missing parent directories are
        created.
    title:
        Optional figure title.

    Returns
    -------
    pathlib.Path
        The path that was written.

    Raises
    ------
    ValueError
        If ``signature`` is empty.
    """
    if signature.empty:
        raise ValueError("Cannot plot an empty signature matrix.")

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    n_genes, n_cell_types = signature.shape
    figure = Figure(figsize=(max(4.0, n_cell_types * 1.2), max(4.0, n_genes * 0.3)))
    axes = figure.subplots()
    sns.heatmap(
        signature,
        ax=axes,
        cmap="viridis",
        cbar_kws={"label": "mean expression"},
    )
    axes.set_xlabel("cell type")
    axes.set_ylabel("gene")
    if title is not None:
        axes.set_title(title)
    figure.tight_layout()
    figure.savefig(resolved, dpi=150)

    logger.info("Wrote signature heatmap to %s", resolved)
    return resolved


def plot_truth_vs_prediction(
    truth: pd.DataFrame,
    prediction: pd.DataFrame,
    path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Render a truth-vs-prediction scatter grid (one panel per cell type).

    Each panel plots true (x) against predicted (y) proportions across samples,
    with a ``y = x`` reference line and ``[0, 1]`` axes.

    Parameters
    ----------
    truth:
        Ground-truth proportions, **cell types (index) x samples (columns)**.
    prediction:
        Estimated proportions, same orientation and labels as ``truth``.
    path:
        Destination image path. Missing parent directories are created.
    title:
        Optional overall figure title.

    Returns
    -------
    pathlib.Path
        The path that was written.

    Raises
    ------
    ValueError
        If the inputs cannot be aligned (see
        :func:`scdecon.validation.align_proportions`).
    """
    truth_aligned, prediction_aligned = align_proportions(truth, prediction)
    cell_types = truth_aligned.index.tolist()

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    n_types = len(cell_types)
    n_cols = min(3, n_types)
    n_rows = ceil(n_types / n_cols)
    figure = Figure(figsize=(n_cols * 3.2, n_rows * 3.2))
    axes = figure.subplots(n_rows, n_cols, squeeze=False)

    for position, cell_type in enumerate(cell_types):
        row, col = divmod(position, n_cols)
        panel = axes[row][col]
        panel.scatter(
            truth_aligned.loc[cell_type].to_numpy(),
            prediction_aligned.loc[cell_type].to_numpy(),
            s=12,
        )
        panel.plot([0.0, 1.0], [0.0, 1.0], color="grey", linestyle="--", linewidth=1)
        panel.set_xlim(0.0, 1.0)
        panel.set_ylim(0.0, 1.0)
        panel.set_title(str(cell_type))
        panel.set_xlabel("true proportion")
        panel.set_ylabel("predicted proportion")

    for position in range(n_types, n_rows * n_cols):
        row, col = divmod(position, n_cols)
        axes[row][col].set_axis_off()

    if title is not None:
        figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(resolved, dpi=150)

    logger.info("Wrote truth-vs-prediction plot to %s", resolved)
    return resolved


def plot_benchmark(
    result: BenchmarkResult,
    path: str | Path,
    *,
    metric: str = "overall_rmse",
    title: str | None = None,
) -> Path:
    """Render a bar chart comparing solvers on a benchmark metric.

    Parameters
    ----------
    result:
        The :class:`~scdecon.deconvolution.BenchmarkResult` to plot. Solver names
        are used exactly as recorded.
    path:
        Destination image path. Missing parent directories are created.
    metric:
        Column of ``result.to_frame()`` to plot (e.g. ``"overall_rmse"``,
        ``"mean_pearson"``, ``"runtime_s"``).
    title:
        Optional figure title.

    Returns
    -------
    pathlib.Path
        The path that was written.

    Raises
    ------
    ValueError
        If the result is empty or ``metric`` is not a benchmark column.
    """
    frame = result.to_frame()
    if frame.empty:
        raise ValueError("Cannot plot an empty benchmark result.")
    if metric not in frame.columns:
        raise ValueError(
            f"metric '{metric}' is not a benchmark column {list(frame.columns)}."
        )

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    figure = Figure(figsize=(max(4.0, len(frame) * 1.2), 4.0))
    axes = figure.subplots()
    axes.bar([str(name) for name in frame.index], frame[metric].to_numpy())
    axes.set_xlabel("solver")
    axes.set_ylabel(metric)
    if title is not None:
        axes.set_title(title)
    figure.tight_layout()
    figure.savefig(resolved, dpi=150)

    logger.info("Wrote benchmark plot to %s", resolved)
    return resolved
