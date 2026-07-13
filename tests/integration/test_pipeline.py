"""End-to-end integration tests for the M8 interface layer.

These exercise the whole pipeline on a tiny fixture, at two layers that must both
hold (they validate different things):

- ``test_cli_pipeline_end_to_end`` drives the real ``scdecon`` CLI as a
  subprocess through ``build-signature -> simulate -> deconvolve -> benchmark``.
  It proves a fresh clone can run the package's public interface end to end and
  runs in every environment.
- ``test_snakemake_pipeline_end_to_end`` runs the actual Snakemake workflow
  (``--cores 1``) over the same fixture, proving the orchestration layer wires
  and executes the CLI correctly. Skipped unless Snakemake is installed (the
  optional ``pipeline`` extra).

This is a *wiring* test, not a scientific-accuracy test: signature and pseudobulk
are built from the same reference (no leakage split), so we assert well-formed
outputs, not recovery quality (recovery is validated on split data in
``tests/unit/test_recovery.py``).
"""

from __future__ import annotations

import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import anndata
import pytest
import yaml

from scdecon.io import read_bulk, write_h5ad

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAKEFILE = REPO_ROOT / "workflow" / "Snakefile"
_HAS_SNAKEMAKE = find_spec("snakemake") is not None
_CLI = [sys.executable, "-m", "scdecon.cli"]


def _write_pipeline_config(
    tmp_path: Path, reference: Path
) -> tuple[Path, dict[str, str]]:
    """Write a complete run-config (tiny, lenient) and return it plus the paths."""
    results = tmp_path / "results"
    paths: dict[str, str] = {
        "reference": str(reference),
        "signature": str(results / "signature.tsv"),
        "heatmap": str(results / "signature_heatmap.png"),
        "bulk": str(results / "pseudobulk.tsv"),
        "truth": str(results / "truth_proportions.tsv"),
        "proportions": str(results / "estimated_proportions.tsv"),
        "metrics": str(results / "benchmark_metrics.tsv"),
        "benchmark_plot": str(results / "benchmark.png"),
    }
    cfg: dict[str, Any] = {
        "paths": paths,
        "preprocessing": {"min_genes": 0, "min_cells": 0, "max_pct_mito": 100.0},
        "markers": {"n_markers_per_type": 2, "min_cells_per_type": 2},
        "simulation": {
            "n_samples": 6,
            "n_cells_per_sample": 30,
            "cell_type_key": "cell_type",
            "random_state": 0,
        },
        "solver": {"name": "nnls"},
        "benchmark": {"solvers": ["nnls", "nusvr", "robust"], "metric": "overall_rmse"},
    }
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return config_path, paths


def _assert_outputs_well_formed(paths: dict[str, str]) -> None:
    """Shared assertions on the artefacts produced by a full pipeline run."""
    for key in ("signature", "bulk", "truth", "proportions", "metrics"):
        assert Path(paths[key]).exists(), f"missing output: {key}"
    for key in ("heatmap", "benchmark_plot"):
        assert Path(paths[key]).exists(), f"missing figure: {key}"

    # Estimated proportions: cell types x samples, each column sums to 1.
    proportions = read_bulk(paths["proportions"])
    assert proportions.shape[1] == 6
    assert proportions.to_numpy().min() >= 0.0
    assert proportions.sum(axis=0).to_numpy() == pytest.approx(1.0)

    # Benchmark metrics: one row per solver.
    metrics = read_bulk(paths["metrics"])
    labels = set(metrics.index) | set(metrics.columns)
    assert {"nnls", "nusvr", "robust"}.issubset(labels)


def test_cli_pipeline_end_to_end(
    tmp_path: Path, raw_reference_adata: anndata.AnnData
) -> None:
    reference = tmp_path / "reference.h5ad"
    write_h5ad(raw_reference_adata, reference)
    config, paths = _write_pipeline_config(tmp_path, reference)

    for command in ("build-signature", "simulate", "deconvolve", "benchmark"):
        result = subprocess.run(
            [*_CLI, command, "--config", str(config)],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
        assert result.returncode == 0, f"{command} failed:\n{result.stderr}"

    _assert_outputs_well_formed(paths)


@pytest.mark.skipif(
    not _HAS_SNAKEMAKE, reason="snakemake not installed (pipeline extra)"
)
def test_snakemake_pipeline_end_to_end(
    tmp_path: Path, raw_reference_adata: anndata.AnnData
) -> None:
    reference = tmp_path / "reference.h5ad"
    write_h5ad(raw_reference_adata, reference)
    config, paths = _write_pipeline_config(tmp_path, reference)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "snakemake",
            "--snakefile",
            str(SNAKEFILE),
            "--config",
            f"run_config={config}",
            f"scdecon_command={sys.executable} -m scdecon.cli",
            "--cores",
            "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 0, result.stderr
    _assert_outputs_well_formed(paths)
