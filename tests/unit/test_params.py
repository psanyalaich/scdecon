"""Unit tests for scdecon.preprocessing.params.PreprocessConfig."""

from __future__ import annotations

import dataclasses

import pytest

from scdecon.preprocessing import PreprocessConfig


def test_defaults() -> None:
    config = PreprocessConfig()
    assert config.min_genes == 200
    assert config.min_cells == 3
    assert config.max_pct_mito == 20.0
    assert config.mito_prefix == "MT-"
    assert config.target_sum == 1e4
    assert config.counts_layer == "counts"


def test_custom_values() -> None:
    config = PreprocessConfig(min_genes=100, max_pct_mito=10.0, target_sum=None)
    assert config.min_genes == 100
    assert config.max_pct_mito == 10.0
    assert config.target_sum is None


def test_is_frozen() -> None:
    config = PreprocessConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.min_genes = 5  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_genes": -1},
        {"min_cells": -1},
        {"max_pct_mito": -0.1},
        {"max_pct_mito": 100.1},
        {"target_sum": 0},
        {"mito_prefix": ""},
        {"counts_layer": ""},
    ],
)
def test_invalid_values_raise(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PreprocessConfig(**kwargs)  # type: ignore[arg-type]
