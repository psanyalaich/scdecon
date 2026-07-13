"""Regenerate the documentation figures under ``docs/assets/`` (deterministic).

Run from the repository root::

    python docs/generate_figures.py

Uses only the public ``scdecon`` API on seeded **synthetic** data (no real
datasets), so the figures are small and fully reproducible. It mirrors the
end-to-end validation flow: split a reference, build a signature, simulate
held-out pseudobulk with known proportions, deconvolve, and score. Two figures
are produced:

- ``recovery.png``  -- true vs. predicted proportions (pipeline recovers known
  composition on clean synthetic data).
- ``benchmark.png`` -- per-solver accuracy (NNLS / nu-SVR / robust) on one shared
  pseudobulk set.
"""

from __future__ import annotations

from pathlib import Path

import anndata
import numpy as np
import pandas as pd

from scdecon.deconvolution import (
    NNLSSolver,
    NuSVRSolver,
    RobustSolver,
    deconvolve,
    run_benchmark,
)
from scdecon.plotting import plot_benchmark, plot_truth_vs_prediction
from scdecon.preprocessing import PreprocessConfig, normalize
from scdecon.signature import (
    SignatureConfig,
    build_signature,
    select_markers,
)
from scdecon.simulation import (
    SimulationConfig,
    simulate_pseudobulk,
    split_reference,
)

# The scdecon plotting layer uses matplotlib's object-oriented Figure API and
# never calls pyplot.show(), so it is headless-safe; matplotlib auto-selects a
# non-interactive backend when no display is available (e.g. in CI).

ASSETS = Path(__file__).parent / "assets"


def _benchmark_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A controlled, same-scale linear mixture for a fair solver comparison.

    ``bulk = signature @ proportions`` plus mild noise, so every solver is
    well-posed (bulk and signature share a measurement scale). Mild noise makes
    the solvers differ, which is what a benchmark should show.
    """
    rng = np.random.default_rng(7)
    cell_types = ["T cell", "B cell", "NK cell", "Macrophage", "Malignant"]
    genes = [f"g{i}" for i in range(30)]
    samples = [f"s{i}" for i in range(40)]
    signature_values = rng.uniform(0.5, 5.0, size=(len(genes), len(cell_types)))
    proportions = rng.dirichlet(np.ones(len(cell_types)), size=len(samples)).T
    bulk_values = signature_values @ proportions
    bulk_values += rng.normal(0.0, 0.02 * float(bulk_values.mean()), bulk_values.shape)
    bulk_values = np.clip(bulk_values, 0.0, None)
    signature = pd.DataFrame(signature_values, index=genes, columns=cell_types)
    bulk = pd.DataFrame(bulk_values, index=genes, columns=samples)
    truth = pd.DataFrame(proportions, index=cell_types, columns=samples)
    return signature, bulk, truth


def _synthetic_reference() -> anndata.AnnData:
    """Five cell types, each with two dominant marker genes over a low background.

    Every gene has a small baseline in every cell (so the signature is dense and
    well-conditioned for all solvers), with each type's two marker genes strongly
    elevated. Small deterministic jitter avoids zero within-group variance.
    """
    rng = np.random.default_rng(42)
    cell_types = ["T cell", "B cell", "NK cell", "Macrophage", "Malignant"]
    n_genes = 2 * len(cell_types)
    n_per_type = 30
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for index, cell_type in enumerate(cell_types):
        first, second = 2 * index, 2 * index + 1
        for _ in range(n_per_type):
            vector = rng.uniform(2.0, 6.0, size=n_genes)  # background in every gene
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


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    raw = _synthetic_reference()

    signature_part, heldout_part = split_reference(
        raw, signature_fraction=0.5, random_state=0
    )
    signature_adata = signature_part.copy()
    normalize(signature_adata, PreprocessConfig())
    signature_config = SignatureConfig(n_markers_per_type=2, min_cells_per_type=2)
    markers = select_markers(signature_adata, signature_config)
    signature = build_signature(signature_adata, markers, signature_config)

    dataset = simulate_pseudobulk(
        heldout_part,
        SimulationConfig(n_samples=40, n_cells_per_sample=50, random_state=1),
    )

    prediction = deconvolve(signature, dataset.bulk)
    plot_truth_vs_prediction(
        dataset.proportions,
        prediction,
        ASSETS / "recovery.png",
        title="Recovery of known proportions (synthetic)",
    )

    bench_signature, bench_bulk, bench_truth = _benchmark_inputs()
    result = run_benchmark(
        bench_signature,
        bench_bulk,
        bench_truth,
        {"NNLS": NNLSSolver(), "nu-SVR": NuSVRSolver(), "robust": RobustSolver()},
    )
    plot_benchmark(
        result,
        ASSETS / "benchmark.png",
        metric="overall_rmse",
        title="Solver accuracy (lower is better)",
    )
    print(f"wrote figures to {ASSETS}")


if __name__ == "__main__":
    main()
