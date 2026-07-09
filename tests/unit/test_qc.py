"""Unit tests for scdecon.preprocessing.qc."""

from __future__ import annotations

import anndata
import numpy as np
import pytest

from scdecon.preprocessing import (
    PreprocessConfig,
    QCSummary,
    compute_qc_metrics,
    filter_cells_and_genes,
)


def _config() -> PreprocessConfig:
    return PreprocessConfig(
        min_genes=2, min_cells=2, max_pct_mito=50.0, mito_prefix="MT-"
    )


# --- compute_qc_metrics ----------------------------------------------------


def test_compute_qc_metrics_annotates_in_place(
    raw_counts_adata: anndata.AnnData,
) -> None:
    result = compute_qc_metrics(raw_counts_adata, _config())
    assert result is raw_counts_adata  # same object, annotated in place
    assert "pct_counts_mt" in result.obs.columns
    assert "mt" in result.var.columns
    assert result.var["mt"].tolist() == [False] * 5 + [True] + [False]


def test_compute_qc_metrics_preserves_expression(
    raw_counts_adata: anndata.AnnData,
) -> None:
    before = raw_counts_adata.X.copy()
    compute_qc_metrics(raw_counts_adata, _config())
    np.testing.assert_array_equal(raw_counts_adata.X, before)
    assert raw_counts_adata.shape == (6, 7)


def test_compute_qc_metrics_mito_percentage(
    raw_counts_adata: anndata.AnnData,
) -> None:
    compute_qc_metrics(raw_counts_adata, _config())
    obs = raw_counts_adata.obs
    assert obs.loc["cell3", "pct_counts_mt"] == pytest.approx(90.0)  # 9 / 10
    assert obs.loc["cell0", "pct_counts_mt"] == pytest.approx(0.0)


# --- filter_cells_and_genes ------------------------------------------------


def test_filter_requires_qc_metrics(raw_counts_adata: anndata.AnnData) -> None:
    with pytest.raises(ValueError, match="compute_qc_metrics"):
        filter_cells_and_genes(raw_counts_adata, _config())


def test_filter_shapes_and_summary(raw_counts_adata: anndata.AnnData) -> None:
    config = _config()
    compute_qc_metrics(raw_counts_adata, config)
    filtered, summary = filter_cells_and_genes(raw_counts_adata, config)

    assert filtered.shape == (4, 6)
    assert filtered.obs_names.tolist() == ["cell0", "cell1", "cell4", "cell5"]
    assert "RARE1" not in filtered.var_names
    assert "MT-CO1" in filtered.var_names

    assert summary.n_cells_before == 6
    assert summary.n_cells_after == 4
    assert summary.n_genes_before == 7
    assert summary.n_genes_after == 6
    assert summary.n_cells_removed_by_min_genes == 1
    assert summary.n_cells_removed_by_max_pct_mito == 1
    assert summary.n_genes_removed_by_min_cells == 1


def test_filter_is_non_destructive(raw_counts_adata: anndata.AnnData) -> None:
    config = _config()
    compute_qc_metrics(raw_counts_adata, config)
    filtered, _ = filter_cells_and_genes(raw_counts_adata, config)
    assert filtered is not raw_counts_adata
    assert raw_counts_adata.shape == (6, 7)  # input untouched


def test_filter_retained_cells_satisfy_thresholds(
    raw_counts_adata: anndata.AnnData,
) -> None:
    config = _config()
    compute_qc_metrics(raw_counts_adata, config)
    filtered, _ = filter_cells_and_genes(raw_counts_adata, config)
    assert bool((filtered.obs["pct_counts_mt"] <= config.max_pct_mito).all())


# --- QCSummary -------------------------------------------------------------


def test_qcsummary_properties_and_render() -> None:
    summary = QCSummary(6, 4, 7, 6, 1, 1, 1)
    assert summary.cells_removed == 2
    assert summary.genes_removed == 1
    assert "cells: 6 -> 4" in summary.render()
    assert str(summary) == summary.render()


def test_qcsummary_rejects_inconsistent_cells() -> None:
    with pytest.raises(ValueError, match="cell counts do not reconcile"):
        QCSummary(6, 5, 7, 6, 1, 1, 1)  # 6 - 1 - 1 != 5


def test_qcsummary_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        QCSummary(6, 7, 7, 6, -1, 0, 1)
