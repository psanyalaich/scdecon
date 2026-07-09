"""Unit tests for scdecon.deconvolution.nusvr.NuSVRSolver.

Permanent regression tests mirroring the NNLS suite: (approximate) recovery,
non-negativity, sum-to-one, determinism, and the degenerate case.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scdecon.deconvolution import NuSVRConfig, NuSVRSolver, Solver


def _signature(seed: int = 0) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 5.0, size=(40, 3))


def _proportions() -> NDArray[np.float64]:
    return np.array([0.2, 0.3, 0.5], dtype=np.float64)


def test_is_a_solver() -> None:
    assert isinstance(NuSVRSolver(), Solver)


def test_approximate_recovery() -> None:
    signature = _signature()
    p_true = _proportions()
    bulk = signature @ p_true
    p_hat = NuSVRSolver().fit(signature, bulk)
    np.testing.assert_allclose(p_hat, p_true, atol=0.1)


def test_non_negativity_and_sum_to_one() -> None:
    signature = _signature()
    bulk = signature @ _proportions()
    p_hat = NuSVRSolver().fit(signature, bulk)
    assert (p_hat >= 0).all()
    assert p_hat.sum() == pytest.approx(1.0)


def test_determinism() -> None:
    signature = _signature()
    bulk = signature @ _proportions()
    first = NuSVRSolver().fit(signature, bulk)
    second = NuSVRSolver().fit(signature, bulk)
    np.testing.assert_array_equal(first, second)


def test_respects_configured_nu() -> None:
    signature = _signature()
    bulk = signature @ _proportions()
    # A different nu should still yield valid proportions.
    p_hat = NuSVRSolver(NuSVRConfig(nu=0.25)).fit(signature, bulk)
    assert p_hat.sum() == pytest.approx(1.0)


def test_degenerate_zero_bulk_raises() -> None:
    signature = _signature()
    bulk = np.zeros(signature.shape[0], dtype=np.float64)
    with pytest.raises(ValueError, match="no positive coefficients"):
        NuSVRSolver().fit(signature, bulk)


def test_does_not_mutate_inputs() -> None:
    signature = _signature()
    bulk = signature @ _proportions()
    signature_copy = signature.copy()
    bulk_copy = bulk.copy()
    NuSVRSolver().fit(signature, bulk)
    np.testing.assert_array_equal(signature, signature_copy)
    np.testing.assert_array_equal(bulk, bulk_copy)
