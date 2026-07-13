"""Tests for the Snakemake workflow and its example configuration.

Two levels of validation:

- ``test_example_config_is_valid_run_config`` runs everywhere and proves the
  YAML the workflow hands to the CLI is a valid ``scdecon`` run configuration
  (the workflow and the CLI share this file, so it must parse as a ``RunConfig``).
- ``test_workflow_dry_run`` invokes ``snakemake -n`` to confirm the Snakefile is
  valid and the rule DAG connects build-signature -> simulate -> deconvolve ->
  benchmark. It is skipped unless Snakemake is installed (the optional
  ``pipeline`` extra); it never executes the CLI (dry run only).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import anndata
import pytest
import yaml

from scdecon.config import RunConfig
from scdecon.io import write_h5ad
from scdecon.preprocessing.params import PreprocessConfig
from scdecon.signature.params import SignatureConfig
from scdecon.simulation.params import SimulationConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAKEFILE = REPO_ROOT / "workflow" / "Snakefile"
EXAMPLE_CONFIG = REPO_ROOT / "config" / "example_run.yaml"

_HAS_SNAKEMAKE = importlib.util.find_spec("snakemake") is not None

# Every path key the workflow's rules reference; the run-config must provide all.
_REQUIRED_PATH_KEYS = {
    "reference",
    "signature",
    "heatmap",
    "bulk",
    "truth",
    "proportions",
    "metrics",
    "benchmark_plot",
}


def test_example_config_is_valid_run_config() -> None:
    """The shipped example config parses and builds the frozen configs."""
    cfg = RunConfig.load(EXAMPLE_CONFIG)
    assert isinstance(cfg.to_preprocess_config(), PreprocessConfig)
    assert isinstance(cfg.to_signature_config(), SignatureConfig)
    assert isinstance(cfg.to_simulation_config(), SimulationConfig)
    assert cfg.solver.name == "nnls"
    assert cfg.benchmark.solvers == ["nnls", "nusvr", "robust"]


def test_example_config_declares_all_workflow_paths() -> None:
    """The example config must define every path the Snakefile wires."""
    with EXAMPLE_CONFIG.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    assert _REQUIRED_PATH_KEYS.issubset(raw["paths"].keys())


def _write_run_config(tmp_path: Path, reference: Path) -> Path:
    """A complete run-config over tmp paths with tiny, fast parameters."""
    results = tmp_path / "results"
    paths: dict[str, str] = {
        "reference": str(reference),
        "signature": str(results / "signature.tsv"),
        "heatmap": str(results / "heatmap.png"),
        "bulk": str(results / "bulk.tsv"),
        "truth": str(results / "truth.tsv"),
        "proportions": str(results / "proportions.tsv"),
        "metrics": str(results / "metrics.tsv"),
        "benchmark_plot": str(results / "benchmark.png"),
    }
    cfg: dict[str, Any] = {
        "paths": paths,
        "preprocessing": {"min_genes": 0, "min_cells": 0, "max_pct_mito": 100.0},
        "markers": {"n_markers_per_type": 2, "min_cells_per_type": 2},
        "simulation": {
            "n_samples": 3,
            "n_cells_per_sample": 10,
            "cell_type_key": "cell_type",
        },
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.mark.skipif(
    not _HAS_SNAKEMAKE, reason="snakemake not installed (pipeline extra)"
)
def test_workflow_dry_run(tmp_path: Path, tiny_adata: anndata.AnnData) -> None:
    """`snakemake -n` resolves the DAG and plans all four rules."""
    reference = tmp_path / "reference.h5ad"
    write_h5ad(tiny_adata, reference)
    run_config = _write_run_config(tmp_path, reference)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "snakemake",
            "--snakefile",
            str(SNAKEFILE),
            "--config",
            f"run_config={run_config}",
            "--cores",
            "1",
            "-n",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 0, result.stderr
    plan = result.stdout + result.stderr
    for rule in ("build_signature", "simulate", "deconvolve", "benchmark"):
        assert rule in plan
