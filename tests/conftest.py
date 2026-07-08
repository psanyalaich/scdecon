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
