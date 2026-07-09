"""Unit tests for scdecon.deconvolution.base.Solver (abstract interface)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scdecon.deconvolution import Solver


class _UniformSolver(Solver):
    """Minimal concrete solver returning a uniform proportion vector."""

    def fit(
        self, signature: NDArray[np.float64], bulk: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        n_cell_types = signature.shape[1]
        return np.full(n_cell_types, 1.0 / n_cell_types)


def test_solver_is_abstract() -> None:
    with pytest.raises(TypeError):
        Solver()  # type: ignore[abstract]


def test_concrete_solver_can_be_used() -> None:
    signature = np.eye(3, dtype=np.float64)
    bulk = np.array([1.0, 2.0, 3.0])
    proportions = _UniformSolver().fit(signature, bulk)
    assert proportions.shape == (3,)
    assert proportions.sum() == pytest.approx(1.0)
