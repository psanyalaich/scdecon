"""Unit tests for scdecon.simulation.params."""

from __future__ import annotations

import pytest

from scdecon.simulation import ProportionPrior, SimulationConfig


def test_defaults() -> None:
    config = SimulationConfig()
    assert config.n_samples == 100
    assert config.n_cells_per_sample == 500
    assert config.cell_type_key == "cell_type"
    assert config.proportion_prior is ProportionPrior.DIRICHLET
    assert config.dirichlet_alpha == 1.0
    assert config.random_state == 0
    assert config.counts_layer is None


def test_proportion_prior_enum() -> None:
    assert ProportionPrior.DIRICHLET.value == "dirichlet"
    assert ProportionPrior("uniform") is ProportionPrior.UNIFORM


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_samples": 0},
        {"n_cells_per_sample": 0},
        {"dirichlet_alpha": 0.0},
        {"cell_type_key": ""},
        {"counts_layer": ""},
    ],
)
def test_invalid_values_raise(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SimulationConfig(**kwargs)  # type: ignore[arg-type]


def test_proportion_prior_must_be_enum() -> None:
    with pytest.raises(ValueError, match="proportion_prior must be"):
        SimulationConfig(proportion_prior="dirichlet")  # type: ignore[arg-type]
