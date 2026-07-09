"""Unit tests for scdecon.preprocessing.normalize and the preprocess pipeline."""

from __future__ import annotations

import anndata
import numpy as np
import pytest

from scdecon.preprocessing import PreprocessConfig, normalize, preprocess
from scdecon.preprocessing.pipeline import QC_SUMMARY_KEY


def _filter_config() -> PreprocessConfig:
    return PreprocessConfig(
        min_genes=2, min_cells=2, max_pct_mito=50.0, mito_prefix="MT-"
    )


# --- normalize -------------------------------------------------------------


def test_normalize_preserves_raw_counts(raw_counts_adata: anndata.AnnData) -> None:
    config = PreprocessConfig()
    before = raw_counts_adata.X.copy()
    result = normalize(raw_counts_adata, config)

    assert result is raw_counts_adata  # in place, same object
    np.testing.assert_array_equal(result.layers[config.counts_layer], before)
    assert not np.array_equal(np.asarray(result.X), before)  # X transformed


def test_normalize_hits_target_sum(raw_counts_adata: anndata.AnnData) -> None:
    config = PreprocessConfig(target_sum=1e4)
    normalize(raw_counts_adata, config)
    # log1p is invertible via expm1; per-cell totals should equal target_sum.
    recovered = np.expm1(np.asarray(raw_counts_adata.X)).sum(axis=1)
    np.testing.assert_allclose(recovered, 1e4, rtol=1e-3)


def test_normalize_refuses_existing_counts_layer(
    raw_counts_adata: anndata.AnnData,
) -> None:
    config = PreprocessConfig()
    normalize(raw_counts_adata, config)
    with pytest.raises(ValueError, match="already exists"):
        normalize(raw_counts_adata, config)


# --- preprocess ------------------------------------------------------------


def test_preprocess_end_to_end(raw_counts_adata: anndata.AnnData) -> None:
    config = _filter_config()
    raw_before = raw_counts_adata.X.copy()
    result = preprocess(raw_counts_adata, config)

    assert isinstance(result, anndata.AnnData)  # not a tuple
    assert result.shape == (4, 6)
    assert config.counts_layer in result.layers

    # raw counts in the layer match the surviving cell/gene subset of the input
    expected = raw_before[np.ix_([0, 1, 4, 5], [0, 1, 2, 3, 4, 5])]
    np.testing.assert_array_equal(result.layers[config.counts_layer], expected)
    assert not np.array_equal(np.asarray(result.X), expected)  # X normalised


def test_preprocess_stores_serialisable_qc_summary(
    raw_counts_adata: anndata.AnnData,
) -> None:
    result = preprocess(raw_counts_adata, _filter_config())
    summary = result.uns[QC_SUMMARY_KEY]
    assert isinstance(summary, dict)
    assert summary["n_cells_before"] == 6
    assert summary["n_cells_after"] == 4
    assert summary["n_genes_after"] == 6
    assert all(isinstance(v, int) for v in summary.values())


def test_preprocess_mutates_input_metadata_but_not_dims(
    raw_counts_adata: anndata.AnnData,
) -> None:
    preprocess(raw_counts_adata, _filter_config())
    # input annotated in place (QC metadata) but expression/dimensions untouched
    assert "pct_counts_mt" in raw_counts_adata.obs.columns
    assert raw_counts_adata.shape == (6, 7)


def test_preprocess_recomputes_qc_deterministically(
    raw_counts_adata: anndata.AnnData,
) -> None:
    from scdecon.preprocessing import compute_qc_metrics

    config = _filter_config()
    # Pre-run QC and then corrupt the mito column; preprocess must recompute it.
    compute_qc_metrics(raw_counts_adata, config)
    raw_counts_adata.obs["pct_counts_mt"] = 999.0
    result = preprocess(raw_counts_adata, config)
    # If recomputed, corrupted values are overwritten and filtering is normal.
    assert result.shape == (4, 6)
