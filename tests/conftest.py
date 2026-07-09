"""Shared pytest fixtures for the scdecon test suite."""

from __future__ import annotations

from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def fixture_dir() -> Path:
    """Directory holding the committed text fixtures."""
    return Path(__file__).parent / "data"


@pytest.fixture
def toy_bulk_path(fixture_dir: Path) -> Path:
    """Path to the committed toy bulk matrix (genes x samples, TSV)."""
    return fixture_dir / "toy_bulk.tsv"


@pytest.fixture
def toy_metadata_path(fixture_dir: Path) -> Path:
    """Path to the committed toy sample-metadata table (CSV)."""
    return fixture_dir / "toy_metadata.csv"


@pytest.fixture
def tiny_adata() -> anndata.AnnData:
    """A minimal in-memory AnnData for round-tripping .h5ad in tests."""
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 10, size=(6, 4)).astype(np.float32)
    obs = pd.DataFrame(
        {"cell_type": ["T", "B", "T", "NK", "B", "T"]},
        index=[f"cell{i}" for i in range(6)],
    )
    var = pd.DataFrame(index=[f"gene{j}" for j in range(4)])
    return anndata.AnnData(X=counts, obs=obs, var=var)


@pytest.fixture
def raw_counts_adata() -> anndata.AnnData:
    """Raw-count AnnData with deterministic QC edge cases.

    Designed so that, with a config of ``min_genes=2, min_cells=2,
    max_pct_mito=50, mito_prefix="MT-"``:

    - ``cell2`` expresses a single gene  -> removed by ``min_genes``
    - ``cell3`` is 90% mitochondrial      -> removed by ``max_pct_mito``
    - ``RARE1`` is detected in one cell    -> removed by ``min_cells``
    - ``MT-CO1`` is the mitochondrial gene

    Surviving cells: cell0, cell1, cell4, cell5; surviving genes: all but RARE1.
    """
    genes = ["CD3D", "CD8A", "MS4A1", "NKG7", "LYZ", "MT-CO1", "RARE1"]
    counts = np.array(
        [
            [5, 4, 3, 2, 1, 0, 7],  # cell0: keep; only cell expressing RARE1
            [4, 0, 5, 3, 2, 1, 0],  # cell1: keep
            [0, 0, 0, 3, 0, 0, 0],  # cell2: 1 gene -> min_genes
            [1, 0, 0, 0, 0, 9, 0],  # cell3: 90% mito -> max_pct_mito
            [3, 2, 0, 4, 1, 0, 0],  # cell4: keep
            [2, 3, 4, 0, 5, 2, 0],  # cell5: keep
        ],
        dtype=np.float32,
    )
    obs = pd.DataFrame(index=[f"cell{i}" for i in range(6)])
    var = pd.DataFrame(index=genes)
    return anndata.AnnData(X=counts, obs=obs, var=var)
