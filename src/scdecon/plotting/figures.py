"""Figures rendered from scdecon results.

Plotting consumes the computational core's outputs (e.g. a signature
``DataFrame``) but the core never depends on plotting: matplotlib and seaborn
are imported only here. Figures are drawn with matplotlib's object-oriented
``Figure`` API (no global ``pyplot`` state), so rendering is headless-safe.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from scdecon.logging_utils import get_logger

__all__ = ["plot_signature_heatmap"]

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
