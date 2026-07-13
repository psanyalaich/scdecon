"""Tests for the Typer CLI composition root (``scdecon.cli``).

These tests verify the *wiring* and the *exit-code policy*, not the science:
the numerical steps are already covered by each module's own tests. We check
that commands load config, construct the right objects, call the library, write
the expected outputs, and translate failures into the documented exit codes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from scdecon import __version__
from scdecon.cli import ExitCode, _build_solver, app
from scdecon.config import SolverSettings
from scdecon.deconvolution import NNLSSolver, NuSVRSolver, RobustSolver
from scdecon.io import read_bulk, write_h5ad, write_table

runner = CliRunner()


def _write_config(tmp_path: Path, mapping: dict[str, Any]) -> Path:
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    return path


@pytest.fixture
def signature_bulk_truth(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write a tiny, well-posed signature/bulk/truth trio to disk as TSV.

    Two samples so that per-cell-type correlations in the benchmark are defined
    (a single sample yields ``nan`` correlations by design).
    """
    signature = pd.DataFrame(
        {"A": [10.0, 8.0, 0.0, 1.0], "B": [0.0, 1.0, 10.0, 8.0]},
        index=["G1", "G2", "G3", "G4"],
    )
    bulk = pd.DataFrame(
        {"S1": [6.0, 4.9, 4.0, 3.8], "S2": [3.0, 3.1, 7.0, 5.9]},
        index=["G1", "G2", "G3", "G4"],
    )
    truth = pd.DataFrame({"S1": [0.6, 0.4], "S2": [0.3, 0.7]}, index=["A", "B"])
    sig_path = write_table(signature, tmp_path / "signature.tsv")
    bulk_path = write_table(bulk, tmp_path / "bulk.tsv")
    truth_path = write_table(truth, tmp_path / "truth.tsv")
    return sig_path, bulk_path, truth_path


# --------------------------------------------------------------------------- #
# Trivial commands / discovery
# --------------------------------------------------------------------------- #
def test_version_prints_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("build-signature", "simulate", "deconvolve", "benchmark"):
        assert command in result.stdout


# --------------------------------------------------------------------------- #
# Solver factory (composition root)
# --------------------------------------------------------------------------- #
def test_build_solver_returns_configured_types() -> None:
    assert isinstance(_build_solver(SolverSettings(name="nnls")), NNLSSolver)
    nusvr = _build_solver(SolverSettings(name="nusvr", nu=0.25))
    assert isinstance(nusvr, NuSVRSolver)
    robust = _build_solver(SolverSettings(name="robust", f_scale=2.0))
    assert isinstance(robust, RobustSolver)


# --------------------------------------------------------------------------- #
# Happy paths (light -- prove wiring, not the science)
# --------------------------------------------------------------------------- #
def test_deconvolve_writes_normalised_proportions(
    tmp_path: Path, signature_bulk_truth: tuple[Path, Path, Path]
) -> None:
    sig_path, bulk_path, _ = signature_bulk_truth
    out = tmp_path / "proportions.tsv"
    config = _write_config(
        tmp_path,
        {
            "paths": {
                "signature": str(sig_path),
                "bulk": str(bulk_path),
                "proportions": str(out),
            },
            "solver": {"name": "nnls"},
        },
    )
    result = runner.invoke(app, ["deconvolve", "--config", str(config)])
    assert result.exit_code == 0, result.output
    proportions = read_bulk(out)
    assert proportions.shape == (2, 2)
    assert proportions.sum(axis=0).to_numpy() == pytest.approx(1.0)


def test_benchmark_writes_metrics(
    tmp_path: Path, signature_bulk_truth: tuple[Path, Path, Path]
) -> None:
    sig_path, bulk_path, truth_path = signature_bulk_truth
    metrics = tmp_path / "metrics.tsv"
    plot = tmp_path / "benchmark.png"
    config = _write_config(
        tmp_path,
        {
            "paths": {
                "signature": str(sig_path),
                "bulk": str(bulk_path),
                "truth": str(truth_path),
                "metrics": str(metrics),
                "benchmark_plot": str(plot),
            },
            "benchmark": {"solvers": ["nnls", "robust"], "metric": "overall_rmse"},
        },
    )
    result = runner.invoke(app, ["benchmark", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert plot.exists()
    frame = read_bulk(metrics)
    assert {"nnls", "robust"}.issubset(set(frame.columns) | set(frame.index))


def test_build_signature_writes_signature_and_heatmap(
    tmp_path: Path, raw_reference_adata: anndata.AnnData
) -> None:
    reference = tmp_path / "reference.h5ad"
    write_h5ad(raw_reference_adata, reference)
    signature_out = tmp_path / "signature.tsv"
    heatmap_out = tmp_path / "signature.png"
    config = _write_config(
        tmp_path,
        {
            "paths": {
                "reference": str(reference),
                "signature": str(signature_out),
                "heatmap": str(heatmap_out),
            },
            "preprocessing": {"min_genes": 0, "min_cells": 0, "max_pct_mito": 100.0},
            "markers": {"n_markers_per_type": 2, "min_cells_per_type": 2},
        },
    )
    result = runner.invoke(app, ["build-signature", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert signature_out.exists() and heatmap_out.exists()
    signature = read_bulk(signature_out)
    # One column per cell type; non-negative linear-scale values.
    assert set(signature.columns) == {"A", "B", "C", "D"}
    assert signature.to_numpy().min() >= 0.0


def test_simulate_writes_bulk_and_truth(
    tmp_path: Path, tiny_adata: anndata.AnnData
) -> None:
    reference = tmp_path / "reference.h5ad"
    write_h5ad(tiny_adata, reference)
    bulk_out = tmp_path / "bulk.tsv"
    truth_out = tmp_path / "truth.tsv"
    config = _write_config(
        tmp_path,
        {
            "paths": {
                "reference": str(reference),
                "bulk": str(bulk_out),
                "truth": str(truth_out),
            },
            "simulation": {
                "n_samples": 3,
                "n_cells_per_sample": 10,
                "cell_type_key": "cell_type",
            },
        },
    )
    result = runner.invoke(app, ["simulate", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert bulk_out.exists() and truth_out.exists()
    truth = read_bulk(truth_out)
    # Each simulated sample's proportions sum to 1.
    assert truth.sum(axis=0).to_numpy() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Exit-code policy
# --------------------------------------------------------------------------- #
def test_missing_config_option_is_usage_error() -> None:
    result = runner.invoke(app, ["build-signature"])
    assert result.exit_code == ExitCode.USAGE  # Typer/Click missing-option -> 2


def test_missing_required_path_is_config_error(tmp_path: Path) -> None:
    # Valid config but no paths.reference for build-signature.
    config = _write_config(tmp_path, {"markers": {"n_markers_per_type": 5}})
    result = runner.invoke(app, ["build-signature", "--config", str(config)])
    assert result.exit_code == ExitCode.CONFIG_ERROR


def test_unknown_key_is_config_error(tmp_path: Path) -> None:
    config = _write_config(tmp_path, {"preprocessing": {"bogus": 1}})
    result = runner.invoke(app, ["build-signature", "--config", str(config)])
    assert result.exit_code == ExitCode.CONFIG_ERROR


def test_out_of_range_value_is_config_error(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        {
            "paths": {"reference": "r.h5ad", "signature": "s.tsv"},
            "preprocessing": {"min_genes": -1},
        },
    )
    result = runner.invoke(app, ["build-signature", "--config", str(config)])
    assert result.exit_code == ExitCode.CONFIG_ERROR


def test_missing_input_file_is_input_error(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        {
            "paths": {
                "signature": str(tmp_path / "missing_signature.tsv"),
                "bulk": str(tmp_path / "missing_bulk.tsv"),
                "proportions": str(tmp_path / "out.tsv"),
            }
        },
    )
    result = runner.invoke(app, ["deconvolve", "--config", str(config)])
    assert result.exit_code == ExitCode.INPUT_ERROR


def test_no_shared_genes_is_runtime_error(tmp_path: Path) -> None:
    signature = pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 1.0]}, index=["G1", "G2"])
    bulk = pd.DataFrame({"S1": [1.0, 1.0]}, index=["X1", "X2"])  # disjoint genes
    sig_path = write_table(signature, tmp_path / "signature.tsv")
    bulk_path = write_table(bulk, tmp_path / "bulk.tsv")
    config = _write_config(
        tmp_path,
        {
            "paths": {
                "signature": str(sig_path),
                "bulk": str(bulk_path),
                "proportions": str(tmp_path / "out.tsv"),
            }
        },
    )
    result = runner.invoke(app, ["deconvolve", "--config", str(config)])
    assert result.exit_code == ExitCode.RUNTIME_ERROR
