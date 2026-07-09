"""Unit tests for scdecon.deconvolution.nnls.NNLSSolver.

These are permanent regression tests for the core solver:
exact recovery (canonical), scale invariance, non-negativity, sum-to-one,
determinism, small-noise recovery, and the degenerate all-zero case.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scdecon.deconvolution import NNLSSolver


def _signature(
    seed: int = 0, n_genes: int = 8, n_cell_types: int = 3
) -> NDArray[np.float64]:
    """A well-conditioned, non-negative signature matrix (full column rank)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 5.0, size=(n_genes, n_cell_types))


def _proportions() -> NDArray[np.float64]:
    return np.array([0.2, 0.3, 0.5], dtype=np.float64)


def test_exact_recovery() -> None:
    """Canonical reference test: b = S @ p_true must recover p_true exactly."""
    signature = _signature()
    p_true = _proportions()
    bulk = signature @ p_true
    p_hat = NNLSSolver().fit(signature, bulk)
    np.testing.assert_allclose(p_hat, p_true, rtol=1e-6, atol=1e-9)


def test_scale_invariance() -> None:
    """Scaling the bulk by a positive constant leaves normalised p unchanged."""
    signature = _signature()
    bulk = signature @ _proportions()
    solver = NNLSSolver()
    p_hat = solver.fit(signature, bulk)
    p_hat_scaled = solver.fit(signature, 7.5 * bulk)
    np.testing.assert_allclose(p_hat_scaled, p_hat, rtol=1e-9, atol=1e-12)


def test_non_negativity() -> None:
    signature = _signature()
    bulk = signature @ _proportions()
    p_hat = NNLSSolver().fit(signature, bulk)
    assert (p_hat >= 0).all()


def test_sum_to_one() -> None:
    signature = _signature()
    bulk = signature @ _proportions()
    p_hat = NNLSSolver().fit(signature, bulk)
    assert p_hat.sum() == pytest.approx(1.0)


def test_determinism() -> None:
    signature = _signature()
    bulk = signature @ _proportions()
    first = NNLSSolver().fit(signature, bulk)
    second = NNLSSolver().fit(signature, bulk)
    np.testing.assert_array_equal(first, second)


def test_small_noise_recovery() -> None:
    signature = _signature()
    p_true = _proportions()
    rng = np.random.default_rng(1)
    bulk = signature @ p_true + rng.normal(0.0, 0.01, size=signature.shape[0])
    p_hat = NNLSSolver().fit(signature, bulk)
    assert p_hat.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(p_hat, p_true, atol=0.05)


def test_degenerate_all_zero_raises() -> None:
    signature = _signature()
    bulk = np.zeros(signature.shape[0], dtype=np.float64)
    with pytest.raises(ValueError, match="all-zero solution"):
        NNLSSolver().fit(signature, bulk)


def test_does_not_mutate_inputs() -> None:
    signature = _signature()
    bulk = signature @ _proportions()
    signature_copy = signature.copy()
    bulk_copy = bulk.copy()
    NNLSSolver().fit(signature, bulk)
    np.testing.assert_array_equal(signature, signature_copy)
    np.testing.assert_array_equal(bulk, bulk_copy)
