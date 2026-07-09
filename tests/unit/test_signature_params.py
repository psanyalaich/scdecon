"""Unit tests for scdecon.signature.params."""

from __future__ import annotations

import pytest

from scdecon.signature import RankMethod, SignatureConfig


def test_defaults() -> None:
    config = SignatureConfig()
    assert config.cell_type_key == "cell_type"
    assert config.n_markers_per_type == 25
    assert config.method is RankMethod.WILCOXON
    assert config.min_cells_per_type == 2


def test_rankmethod_is_str_compatible() -> None:
    assert RankMethod.WILCOXON.value == "wilcoxon"
    assert str(RankMethod.WILCOXON) == "wilcoxon"
    assert RankMethod("t-test") is RankMethod.T_TEST


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_markers_per_type": 0},
        {"min_cells_per_type": 0},
        {"cell_type_key": ""},
    ],
)
def test_invalid_values_raise(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SignatureConfig(**kwargs)  # type: ignore[arg-type]


def test_method_must_be_rankmethod() -> None:
    with pytest.raises(ValueError, match="method must be a RankMethod"):
        SignatureConfig(method="wilcoxon")  # type: ignore[arg-type]
