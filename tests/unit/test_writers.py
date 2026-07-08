"""Unit tests for scdecon.io.writers."""

from __future__ import annotations

from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest

from scdecon.io import read_bulk, read_h5ad, write_h5ad, write_table


def _unordered_frame() -> pd.DataFrame:
    """A small numeric frame whose gene index is deliberately unsorted."""
    return pd.DataFrame(
        {"S1": [1, 2, 3], "S2": [4, 5, 6]},
        index=pd.Index(["G3", "G1", "G2"], name="gene"),
    )


# --- write_table -----------------------------------------------------------


def test_write_table_roundtrip_tsv(tmp_path: Path) -> None:
    original = _unordered_frame()
    path = write_table(original, tmp_path / "out.tsv")
    loaded = read_bulk(path)
    pd.testing.assert_frame_equal(loaded, original)


def test_write_table_roundtrip_csv(tmp_path: Path) -> None:
    original = _unordered_frame()
    path = write_table(original, tmp_path / "out.csv")
    loaded = read_bulk(path)
    pd.testing.assert_frame_equal(loaded, original)


def test_write_table_preserves_row_order(tmp_path: Path) -> None:
    original = _unordered_frame()
    path = write_table(original, tmp_path / "out.tsv")
    loaded = read_bulk(path)
    assert loaded.index.tolist() == ["G3", "G1", "G2"]


def test_write_table_returns_path(tmp_path: Path) -> None:
    target = tmp_path / "out.tsv"
    result = write_table(_unordered_frame(), target)
    assert isinstance(result, Path)
    assert result == target


def test_write_table_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deeper" / "out.tsv"
    write_table(_unordered_frame(), target)
    assert target.is_file()


def test_write_table_unknown_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="separator"):
        write_table(_unordered_frame(), tmp_path / "out.dat")


def test_write_table_explicit_separator(tmp_path: Path) -> None:
    original = _unordered_frame()
    path = write_table(original, tmp_path / "out.dat", sep="\t")
    loaded = read_bulk(path, sep="\t")
    pd.testing.assert_frame_equal(loaded, original)


# --- write_h5ad ------------------------------------------------------------


def test_write_h5ad_roundtrip(tmp_path: Path, tiny_adata: anndata.AnnData) -> None:
    path = write_h5ad(tiny_adata, tmp_path / "out.h5ad")
    loaded = read_h5ad(path)
    assert loaded.shape == tiny_adata.shape
    assert loaded.obs["cell_type"].tolist() == tiny_adata.obs["cell_type"].tolist()
    np.testing.assert_allclose(loaded.X, tiny_adata.X)


def test_write_h5ad_returns_path(tmp_path: Path, tiny_adata: anndata.AnnData) -> None:
    target = tmp_path / "out.h5ad"
    result = write_h5ad(tiny_adata, target)
    assert isinstance(result, Path)
    assert result == target


def test_write_h5ad_creates_parent_dirs(
    tmp_path: Path, tiny_adata: anndata.AnnData
) -> None:
    target = tmp_path / "nested" / "out.h5ad"
    write_h5ad(tiny_adata, target)
    assert target.is_file()
