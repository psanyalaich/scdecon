"""Plotting: figures rendered from scdecon results.

This layer depends on the computational core (it consumes core outputs such as a
signature ``DataFrame``), but the core never depends on plotting. matplotlib and
seaborn live only here.
"""

from scdecon.plotting.figures import (
    plot_signature_heatmap,
    plot_truth_vs_prediction,
)

__all__ = ["plot_signature_heatmap", "plot_truth_vs_prediction"]
