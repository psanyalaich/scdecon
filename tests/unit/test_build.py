"""Unit tests for scdecon.signature.build."""

from __future__ import annotations

from typing import cast

import anndata
import numpy as np
import pandas as pd
import pytest

from scdecon.signature import (
    MarkerSet,
    SignatureConfig,
    build_signature,
    select_markers,
)
from scdecon.signature.build import _dense, _validate_signature_frame


def _config() -> SignatureConfig:
    return SignatureConfig(cell_type_key="cell_type", n_markers_per_type=3)


@pytest.fixture
def markers(synthetic_signature_adata: anndata.AnnData) -> MarkerSet:
    return select_markers(synthetic_signature_adata, _config())


# --- build_signature -------------------------------------------------------


def test_dimensions_and_labels(
    synthetic_signature_adata: anndata.AnnData, markers: MarkerSet
) -> None:
    signature = build_signature(synthetic_signature_adata, markers, _config())
    assert signature.shape == (len(markers.genes()), 4)
    assert signature.columns.tolist() == ["A", "B", "C", "D"]


def test_row_order_is_exactly_markers_genes(
    synthetic_signature_adata: anndata.AnnData, markers: MarkerSet
) -> None:
    signature = build_signature(synthetic_signature_adata, markers, _config())
    assert signature.index.tolist() == markers.genes()


def test_block_structure(
    synthetic_signature_adata: anndata.AnnData, markers: MarkerSet
) -> None:
    signature = build_signature(synthetic_signature_adata, markers, _config())
    for cell_type, genes in markers.per_type.items():
        for gene in genes:
            assert signature.loc[gene].idxmax() == cell_type


def test_values_are_linear_scale(
    synthetic_signature_adata: anndata.AnnData, markers: MarkerSet
) -> None:
    signature = build_signature(synthetic_signature_adata, markers, _config())
    genes = markers.genes()
    dense = _dense(synthetic_signature_adata[:, genes].X)
    linear = np.expm1(dense)
    labels = synthetic_signature_adata.obs["cell_type"].astype(str).to_numpy()
    for cell_type in ["A", "B", "C", "D"]:
        expected = linear[labels == cell_type].mean(axis=0)
        np.testing.assert_allclose(signature[cell_type].to_numpy(), expected, rtol=1e-5)
    # A marker on the log scale is ~5; on the linear scale expm1(5) ~= 147.
    a_marker = markers.per_type["A"][0]
    assert cast(float, signature.loc[a_marker, "A"]) > 100


def test_values_non_negative_and_finite(
    synthetic_signature_adata: anndata.AnnData, markers: MarkerSet
) -> None:
    signature = build_signature(synthetic_signature_adata, markers, _config())
    values = signature.to_numpy()
    assert (values >= 0).all()
    assert np.isfinite(values).all()


def test_deterministic(
    synthetic_signature_adata: anndata.AnnData, markers: MarkerSet
) -> None:
    first = build_signature(synthetic_signature_adata, markers, _config())
    second = build_signature(synthetic_signature_adata, markers, _config())
    pd.testing.assert_frame_equal(first, second)


# --- validation errors -----------------------------------------------------


def test_missing_marker_gene_raises(
    synthetic_signature_adata: anndata.AnnData,
) -> None:
    bad = MarkerSet({"A": ("NOT_A_GENE",), "B": ("GB1",)})
    with pytest.raises(ValueError, match="not found"):
        build_signature(synthetic_signature_adata, bad, _config())


def test_unknown_cell_type_raises(
    synthetic_signature_adata: anndata.AnnData,
) -> None:
    bad = MarkerSet({"A": ("GA1",), "ZZZ": ("GB1",)})
    with pytest.raises(ValueError, match="absent from adata"):
        build_signature(synthetic_signature_adata, bad, _config())


def test_missing_cell_type_key_raises(
    synthetic_signature_adata: anndata.AnnData, markers: MarkerSet
) -> None:
    config = SignatureConfig(cell_type_key="missing")
    with pytest.raises(ValueError, match="cell_type_key"):
        build_signature(synthetic_signature_adata, markers, config)


def test_duplicate_var_names_raises() -> None:
    adata = anndata.AnnData(
        X=np.ones((4, 3), dtype=np.float32),
        obs=pd.DataFrame(
            {"cell_type": pd.Categorical(["A", "A", "B", "B"])},
            index=[f"c{i}" for i in range(4)],
        ),
        var=pd.DataFrame(index=["g1", "g1", "g2"]),
    )
    markers = MarkerSet({"A": ("g2",), "B": ("g2",)})
    with pytest.raises(ValueError, match="duplicates"):
        build_signature(adata, markers, SignatureConfig())


# --- _validate_signature_frame (pure helper) -------------------------------


def test_validator_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        _validate_signature_frame(pd.DataFrame())


def test_validator_rejects_duplicate_index() -> None:
    frame = pd.DataFrame({"A": [1.0, 2.0]}, index=["g1", "g1"])
    with pytest.raises(ValueError, match="duplicate"):
        _validate_signature_frame(frame)


def test_validator_rejects_negative() -> None:
    frame = pd.DataFrame({"A": [1.0, -1.0]}, index=["g1", "g2"])
    with pytest.raises(ValueError, match="negative"):
        _validate_signature_frame(frame)


def test_validator_rejects_non_finite() -> None:
    frame = pd.DataFrame({"A": [1.0, np.inf]}, index=["g1", "g2"])
    with pytest.raises(ValueError, match="non-finite"):
        _validate_signature_frame(frame)
