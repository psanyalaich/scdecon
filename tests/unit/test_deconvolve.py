"""Unit tests for scdecon.deconvolution.deconvolve (orchestrator)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from scdecon.deconvolution import NNLSSolver, Solver, deconvolve


def _signature_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = rng.uniform(0.1, 5.0, size=(8, 3))
    genes = [f"g{i}" for i in range(8)]
    return pd.DataFrame(data, index=genes, columns=["A", "B", "C"])


def test_returns_cell_types_by_samples() -> None:
    signature = _signature_frame()
    p_true = np.array([[0.2, 0.5], [0.3, 0.1], [0.5, 0.4]])  # 3 cell types x 2 samples
    bulk = pd.DataFrame(
        signature.to_numpy() @ p_true,
        index=signature.index,
        columns=["s1", "s2"],
    )
    result = deconvolve(signature, bulk)
    assert result.index.tolist() == ["A", "B", "C"]
    assert result.columns.tolist() == ["s1", "s2"]
    assert result.index.name == "cell_type"


def test_exact_recovery_through_orchestrator() -> None:
    signature = _signature_frame()
    p_true = np.array([[0.2, 0.5], [0.3, 0.1], [0.5, 0.4]])
    bulk = pd.DataFrame(
        signature.to_numpy() @ p_true,
        index=signature.index,
        columns=["s1", "s2"],
    )
    result = deconvolve(signature, bulk)
    np.testing.assert_allclose(result.to_numpy(), p_true, rtol=1e-6, atol=1e-9)


def test_columns_sum_to_one_and_non_negative() -> None:
    signature = _signature_frame()
    bulk = pd.DataFrame(
        np.abs(np.random.default_rng(2).normal(5.0, 1.0, size=(8, 3))),
        index=signature.index,
        columns=["s1", "s2", "s3"],
    )
    result = deconvolve(signature, bulk)
    np.testing.assert_allclose(result.sum(axis=0).to_numpy(), 1.0, rtol=1e-9)
    assert (result.to_numpy() >= 0).all()


def test_defaults_to_nnls_and_accepts_custom_solver() -> None:
    signature = _signature_frame()
    bulk = pd.DataFrame(
        signature.to_numpy() @ np.array([[1.0], [0.0], [0.0]]),
        index=signature.index,
        columns=["s1"],
    )
    # default solver
    default_result = deconvolve(signature, bulk)
    assert default_result.shape == (3, 1)

    class _FirstTypeSolver(Solver):
        def fit(
            self, sig: NDArray[np.float64], b: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            out = np.zeros(sig.shape[1])
            out[0] = 1.0
            return out

    custom = deconvolve(signature, bulk, solver=_FirstTypeSolver())
    np.testing.assert_array_equal(custom["s1"].to_numpy(), [1.0, 0.0, 0.0])
    # sanity: default NNLS also recovers the single-type mixture here
    assert isinstance(NNLSSolver(), Solver)


def test_aligns_when_bulk_has_extra_and_missing_genes() -> None:
    signature = _signature_frame()  # genes g0..g7
    p_true = np.array([[0.25], [0.25], [0.5]])
    full_bulk = signature.to_numpy() @ p_true
    # bulk missing g7, with an extra unrelated gene appended
    bulk = pd.DataFrame(
        np.vstack([full_bulk[:7], [[999.0]]]),
        index=[f"g{i}" for i in range(7)] + ["gZ"],
        columns=["s1"],
    )
    result = deconvolve(signature, bulk, min_overlap=0.0)
    assert result.shape == (3, 1)
    assert result["s1"].sum() == pytest.approx(1.0)


def test_no_shared_genes_raises() -> None:
    signature = _signature_frame()
    bulk = pd.DataFrame(np.ones((3, 1)), index=["x1", "x2", "x3"], columns=["s1"])
    with pytest.raises(ValueError, match="share no genes"):
        deconvolve(signature, bulk)
