"""Unit tests for scdecon.deconvolution.params."""

from __future__ import annotations

import pytest

from scdecon.deconvolution import NuSVRConfig, RobustConfig, RobustLoss


def test_nusvr_defaults() -> None:
    assert NuSVRConfig().nu == 0.5


@pytest.mark.parametrize("bad_nu", [0.0, -0.1, 1.5])
def test_nusvr_invalid_nu_raises(bad_nu: float) -> None:
    with pytest.raises(ValueError, match="nu must be in"):
        NuSVRConfig(nu=bad_nu)


def test_robust_defaults() -> None:
    config = RobustConfig()
    assert config.loss is RobustLoss.SOFT_L1
    assert config.f_scale == 1.0


def test_robust_loss_enum() -> None:
    assert RobustLoss.SOFT_L1.value == "soft_l1"
    assert RobustLoss("huber") is RobustLoss.HUBER


@pytest.mark.parametrize("bad_scale", [0.0, -1.0])
def test_robust_invalid_f_scale_raises(bad_scale: float) -> None:
    with pytest.raises(ValueError, match="f_scale must be"):
        RobustConfig(f_scale=bad_scale)


def test_robust_loss_must_be_enum() -> None:
    with pytest.raises(ValueError, match="loss must be a RobustLoss"):
        RobustConfig(loss="soft_l1")  # type: ignore[arg-type]
