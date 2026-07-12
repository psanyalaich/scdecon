"""Unit tests for scdecon.io.readers."""

from __future__ import annotations

from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest

from scdecon.io import read_bulk, read_h5ad, read_metadata

# --- read_bulk -------------------------------------------------------------


def test_read_bulk_shape_and_orientation(toy_bulk_path: Path) -> None:
    frame = read_bulk(toy_bulk_path)
    assert frame.shape == (6, 4)
    assert frame.index.tolist() == ["CD3D", "CD8A", "MS4A1", "NKG7", "LYZ", "FCGR3A"]
    assert frame.columns.tolist() == ["S1", "S2", "S3", "S4"]


def test_read_bulk_preserves_values(toy_bulk_path: Path) -> None:
    frame = read_bulk(toy_bulk_path)
    assert frame.loc["CD3D", "S1"] == 12
    assert frame.loc["LYZ", "S4"] == 15
    assert all(pd.api.types.is_numeric_dtype(frame[c]) for c in frame.columns)


def test_read_bulk_accepts_str_path(toy_bulk_path: Path) -> None:
    frame = read_bulk(str(toy_bulk_path))
    assert frame.shape == (6, 4)


def test_read_bulk_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_bulk(tmp_path / "does_not_exist.tsv")


def test_read_bulk_duplicate_genes(tmp_path: Path) -> None:
    path = tmp_path / "dup.tsv"
    path.write_text("gene\tS1\nG1\t1\nG1\t2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate gene identifiers"):
        read_bulk(path)


def test_read_bulk_non_numeric(tmp_path: Path) -> None:
    path = tmp_path / "text.tsv"
    path.write_text("gene\tS1\nG1\thello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-numeric"):
        read_bulk(path)


def test_read_bulk_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.tsv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        read_bulk(path)


def test_read_bulk_unknown_suffix(tmp_path: Path) -> None:
    path = tmp_path / "matrix.dat"
    path.write_text("gene\tS1\nG1\t1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="separator"):
        read_bulk(path)


def test_read_bulk_explicit_separator(tmp_path: Path) -> None:
    path = tmp_path / "matrix.dat"
    path.write_text("gene\tS1\nG1\t1\n", encoding="utf-8")
    frame = read_bulk(path, sep="\t")
    assert frame.loc["G1", "S1"] == 1


# --- read_metadata ---------------------------------------------------------


def test_read_metadata_shape_and_columns(toy_metadata_path: Path) -> None:
    frame = read_metadata(toy_metadata_path)
    assert frame.index.tolist() == ["S1", "S2", "S3", "S4"]
    assert frame.columns.tolist() == ["condition", "batch"]
    assert frame.loc["S1", "condition"] == "tumor"


def test_read_metadata_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_metadata(tmp_path / "nope.csv")


def test_read_metadata_duplicate_index(tmp_path: Path) -> None:
    path = tmp_path / "dup.csv"
    path.write_text("sample,cond\nS1,a\nS1,b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate identifiers"):
        read_metadata(path)


def test_read_metadata_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        read_metadata(path)


# --- read_h5ad -------------------------------------------------------------


def test_read_h5ad_roundtrip(tmp_path: Path, tiny_adata: anndata.AnnData) -> None:
    path = tmp_path / "tiny.h5ad"
    tiny_adata.write_h5ad(path)

    loaded = read_h5ad(path)

    assert loaded.shape == tiny_adata.shape
    assert loaded.obs["cell_type"].tolist() == tiny_adata.obs["cell_type"].tolist()
    assert loaded.var_names.tolist() == tiny_adata.var_names.tolist()
    np.testing.assert_allclose(loaded.X, tiny_adata.X)


def test_read_h5ad_accepts_str_path(
    tmp_path: Path, tiny_adata: anndata.AnnData
) -> None:
    path = tmp_path / "tiny.h5ad"
    tiny_adata.write_h5ad(path)
    loaded = read_h5ad(str(path))
    assert loaded.shape == tiny_adata.shape


def test_read_h5ad_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_h5ad(tmp_path / "missing.h5ad")
