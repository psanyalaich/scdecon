"""Unit tests for scdecon.signature.markers."""

from __future__ import annotations

import anndata
import pytest

from scdecon.signature import (
    MarkerSet,
    RankGenesGroupsSelector,
    SignatureConfig,
    select_markers,
)
from scdecon.signature.markers import _drop_shared


def _config(**overrides: object) -> SignatureConfig:
    params: dict[str, object] = {
        "cell_type_key": "cell_type",
        "n_markers_per_type": 3,
        "min_cells_per_type": 2,
    }
    params.update(overrides)
    return SignatureConfig(**params)  # type: ignore[arg-type]


# --- specificity filter (pure helper) --------------------------------------


def test_drop_shared_removes_cross_type_genes() -> None:
    ranked = {"A": ["g1", "shared"], "B": ["g2", "shared"], "C": ["g3"]}
    specific = _drop_shared(ranked)
    assert specific == {"A": ("g1",), "B": ("g2",), "C": ("g3",)}


# --- select_markers --------------------------------------------------------


def test_select_markers_finds_type_specific(
    synthetic_signature_adata: anndata.AnnData,
) -> None:
    markers = select_markers(synthetic_signature_adata, _config())
    assert {"GA1", "GA2"}.issubset(set(markers.per_type["A"]))
    assert {"GB1", "GB2"}.issubset(set(markers.per_type["B"]))
    assert {"GC1", "GC2"}.issubset(set(markers.per_type["C"]))


def test_select_markers_drops_shared_gene(
    synthetic_signature_adata: anndata.AnnData,
) -> None:
    markers = select_markers(synthetic_signature_adata, _config())
    assert "SHARED" not in markers.genes()
    assert "SHARED" not in markers.per_type["A"]
    assert "SHARED" not in markers.per_type["B"]


def test_markerset_genes_deduplicated(
    synthetic_signature_adata: anndata.AnnData,
) -> None:
    markers = select_markers(synthetic_signature_adata, _config())
    genes = markers.genes()
    assert len(genes) == len(set(genes))


def test_markerset_to_frame(
    synthetic_signature_adata: anndata.AnnData,
) -> None:
    markers = select_markers(synthetic_signature_adata, _config())
    frame = markers.to_frame()
    assert list(frame.columns) == ["cell_type", "gene", "rank"]
    expected_rows = sum(len(v) for v in markers.per_type.values())
    assert len(frame) == expected_rows


def test_select_markers_respects_n(
    synthetic_signature_adata: anndata.AnnData,
) -> None:
    markers = select_markers(synthetic_signature_adata, _config(n_markers_per_type=2))
    for genes in markers.per_type.values():
        assert len(genes) <= 2


def test_select_markers_deterministic(
    synthetic_signature_adata: anndata.AnnData,
) -> None:
    first = select_markers(synthetic_signature_adata.copy(), _config())
    second = select_markers(synthetic_signature_adata.copy(), _config())
    assert dict(first.per_type) == dict(second.per_type)


def test_select_markers_annotates_uns_only(
    synthetic_signature_adata: anndata.AnnData,
) -> None:
    before = synthetic_signature_adata.X.copy()
    select_markers(synthetic_signature_adata, _config())
    assert "rank_genes_groups" in synthetic_signature_adata.uns
    assert synthetic_signature_adata.shape == (16, 11)
    assert (synthetic_signature_adata.X == before).all()


def test_selector_can_be_used_directly(
    synthetic_signature_adata: anndata.AnnData,
) -> None:
    markers = RankGenesGroupsSelector().select(synthetic_signature_adata, _config())
    assert isinstance(markers, MarkerSet)


# --- validation errors -----------------------------------------------------


def test_missing_cell_type_key_raises(
    synthetic_signature_adata: anndata.AnnData,
) -> None:
    with pytest.raises(ValueError, match="cell_type_key"):
        select_markers(synthetic_signature_adata, _config(cell_type_key="missing"))


def test_single_cell_type_raises(
    synthetic_signature_adata: anndata.AnnData,
) -> None:
    synthetic_signature_adata.obs["cell_type"] = "A"
    with pytest.raises(ValueError, match="at least 2 cell types"):
        select_markers(synthetic_signature_adata, _config())


def test_too_few_cells_per_type_raises(
    synthetic_signature_adata: anndata.AnnData,
) -> None:
    with pytest.raises(ValueError, match="fewer than"):
        select_markers(synthetic_signature_adata, _config(min_cells_per_type=100))
