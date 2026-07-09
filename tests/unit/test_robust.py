"""Unit tests for scdecon.deconvolution.robust.RobustSolver.

Permanent regression tests mirroring the NNLS suite: recovery (exact and under
outliers), non-negativity, sum-to-one, determinism, and the degenerate case.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scdecon.deconvolution import RobustConfig, RobustLoss, RobustSolver, Solver


def _signature(seed: int = 0) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 5.0, size=(20, 3))


def _proportions() -> NDArray[np.float64]:
    return np.array([0.2, 0.3, 0.5], dtype=np.float64)


def test_is_a_solver() -> None:
    assert isinstance(RobustSolver(), Solver)


def test_recovery_on_clean_data() -> None:
    signature = _signature()
    p_true = _proportions()
    bulk = signature @ p_true
    p_hat = RobustSolver().fit(signature, bulk)
    np.testing.assert_allclose(p_hat, p_true, atol=1e-3)


def test_robust_to_outlier_gene() -> None:
    signature = _signature()
    p_true = _proportions()
    bulk = signature @ p_true
    corrupted = bulk.copy()
    corrupted[0] += 1000.0  # a single gross outlier gene
    p_hat = RobustSolver().fit(signature, corrupted)
    # The robust loss should keep the estimate close despite the outlier.
    np.testing.assert_allclose(p_hat, p_true, atol=0.1)


def test_non_negativity_and_sum_to_one() -> None:
    signature = _signature()
    bulk = signature @ _proportions()
    p_hat = RobustSolver().fit(signature, bulk)
    assert (p_hat >= 0).all()
    assert p_hat.sum() == pytest.approx(1.0)


def test_determinism() -> None:
    signature = _signature()
    bulk = signature @ _proportions()
    first = RobustSolver().fit(signature, bulk)
    second = RobustSolver().fit(signature, bulk)
    np.testing.assert_array_equal(first, second)


def test_huber_loss_configurable() -> None:
    signature = _signature()
    bulk = signature @ _proportions()
    p_hat = RobustSolver(RobustConfig(loss=RobustLoss.HUBER)).fit(signature, bulk)
    assert p_hat.sum() == pytest.approx(1.0)


def test_degenerate_zero_bulk_raises() -> None:
    signature = _signature()
    bulk = np.zeros(signature.shape[0], dtype=np.float64)
    with pytest.raises(ValueError, match="no positive expression"):
        RobustSolver().fit(signature, bulk)


def test_does_not_mutate_inputs() -> None:
    signature = _signature()
    bulk = signature @ _proportions()
    signature_copy = signature.copy()
    bulk_copy = bulk.copy()
    RobustSolver().fit(signature, bulk)
    np.testing.assert_array_equal(signature, signature_copy)
    np.testing.assert_array_equal(bulk, bulk_copy)
