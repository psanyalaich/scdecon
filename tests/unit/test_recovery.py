"""End-to-end recovery test for the deconvolution pipeline.

Verifies that the full pipeline recovers known composition on clean, deterministic
synthetic data:

    split_reference -> normalize -> select_markers -> build_signature
        -> simulate_pseudobulk (held-out) -> deconvolve -> evaluate

The thresholds below verify *successful recovery* on synthetic data; they are not
a benchmark of absolute performance. The dataset is intentionally simple (equal
per-cell library sizes, disjoint marker blocks) so recovery is near-exact and the
test is robust and reproducible.
"""

from __future__ import annotations

import anndata
import numpy as np
import pandas as pd

from scdecon.deconvolution import deconvolve
from scdecon.preprocessing import PreprocessConfig, normalize
from scdecon.signature import SignatureConfig, build_signature, select_markers
from scdecon.simulation import (
    SimulationConfig,
    simulate_pseudobulk,
    split_reference,
)
from scdecon.validation import evaluate


def _recovery_adata() -> anndata.AnnData:
    """Four cell types with disjoint 2-gene marker blocks and equal library size.

    Every cell of a type expresses only that type's two genes (~50 each, so total
    counts ~100 for every cell across all types). Small deterministic jitter
    avoids zero within-group variance for marker ranking.
    """
    rng = np.random.default_rng(42)
    cell_types = ["A", "B", "C", "D"]
    blocks = {"A": (0, 1), "B": (2, 3), "C": (4, 5), "D": (6, 7)}
    n_genes = 8
    n_per_type = 20
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for cell_type in cell_types:
        first, second = blocks[cell_type]
        for _ in range(n_per_type):
            vector = np.zeros(n_genes)
            vector[first] = 50.0 + rng.normal(0.0, 1.0)
            vector[second] = 50.0 + rng.normal(0.0, 1.0)
            rows.append(np.clip(vector, 0.0, None))
            labels.append(cell_type)
    counts = np.asarray(rows, dtype=np.float32)
    obs = pd.DataFrame(
        {"cell_type": pd.Categorical(labels)},
        index=[f"cell{i}" for i in range(len(labels))],
    )
    var = pd.DataFrame(index=[f"g{g}" for g in range(n_genes)])
    return anndata.AnnData(X=counts, obs=obs, var=var)


def test_pipeline_recovers_known_proportions() -> None:
    raw = _recovery_adata()
    signature_part, heldout_part = split_reference(
        raw, signature_fraction=0.5, random_state=0
    )

    # Build the signature from the signature partition (normalise only; the QC
    # thresholds are meant for real data, not this tiny synthetic fixture).
    signature_adata = signature_part.copy()
    normalize(signature_adata, PreprocessConfig())
    signature_config = SignatureConfig(n_markers_per_type=2, min_cells_per_type=2)
    markers = select_markers(signature_adata, signature_config)
    signature = build_signature(signature_adata, markers, signature_config)

    # Simulate pseudobulk from the held-out partition (raw counts).
    dataset = simulate_pseudobulk(
        heldout_part,
        SimulationConfig(n_samples=25, n_cells_per_sample=50, random_state=1),
    )

    prediction = deconvolve(signature, dataset.bulk)
    report = evaluate(dataset.proportions, prediction)

    assert report.overall_rmse < 0.05
    assert report.mean_pearson > 0.9
