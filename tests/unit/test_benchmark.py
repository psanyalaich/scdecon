"""Unit tests for scdecon.deconvolution.benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from scdecon.deconvolution import (
    BenchmarkResult,
    NNLSSolver,
    RobustSolver,
    Solver,
    run_benchmark,
)
from scdecon.validation import ValidationReport


def _benchmark_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    genes = [f"g{i}" for i in range(6)]
    cell_types = ["A", "B", "C"]
    samples = ["s1", "s2", "s3", "s4"]
    signature_values = rng.uniform(0.1, 5.0, size=(6, 3))
    proportions = rng.dirichlet(np.ones(3), size=4).T  # (3 types, 4 samples)
    bulk_values = signature_values @ proportions
    signature = pd.DataFrame(signature_values, index=genes, columns=cell_types)
    bulk = pd.DataFrame(bulk_values, index=genes, columns=samples)
    truth = pd.DataFrame(proportions, index=cell_types, columns=samples)
    return signature, bulk, truth


class _RecordingSolver(Solver):
    """Records (copies of) every input it receives, for fairness testing."""

    def __init__(self) -> None:
        self.calls: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []

    def fit(
        self, signature: NDArray[np.float64], bulk: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        self.calls.append((signature.copy(), bulk.copy()))
        n_cell_types = signature.shape[1]
        return np.full(n_cell_types, 1.0 / n_cell_types)


# --- fairness (first-class) ------------------------------------------------


def test_all_solvers_receive_identical_inputs() -> None:
    signature, bulk, truth = _benchmark_inputs()
    first, second = _RecordingSolver(), _RecordingSolver()
    run_benchmark(signature, bulk, truth, {"first": first, "second": second})

    assert len(first.calls) == len(second.calls) > 0
    for (sig_a, bulk_a), (sig_b, bulk_b) in zip(first.calls, second.calls, strict=True):
        np.testing.assert_array_equal(sig_a, sig_b)
        np.testing.assert_array_equal(bulk_a, bulk_b)


# --- correctness & result object -------------------------------------------


def test_run_benchmark_with_real_solvers() -> None:
    signature, bulk, truth = _benchmark_inputs()
    solvers = {"nnls": NNLSSolver(), "robust": RobustSolver()}
    result = run_benchmark(signature, bulk, truth, solvers)

    assert isinstance(result, BenchmarkResult)
    frame = result.to_frame()
    assert list(frame.index) == ["nnls", "robust"]
    assert list(frame.columns) == [
        "overall_rmse",
        "mean_pearson",
        "mean_spearman",
        "runtime_s",
    ]
    metric_values = frame[["overall_rmse", "mean_pearson", "mean_spearman"]].to_numpy()
    assert np.isfinite(metric_values).all()
    assert result.best() in solvers


def test_solver_names_preserved_exactly() -> None:
    signature, bulk, truth = _benchmark_inputs()
    names = {"My Custom Solver!": NNLSSolver(), "solver-2 (v2)": RobustSolver()}
    result = run_benchmark(signature, bulk, truth, names)
    assert set(result.reports) == set(names)
    assert set(result.runtimes) == set(names)
    assert list(result.to_frame().index) == list(names)
    assert result.best() in names


def test_metrics_are_deterministic() -> None:
    signature, bulk, truth = _benchmark_inputs()
    metric_cols = ["overall_rmse", "mean_pearson", "mean_spearman"]
    first = run_benchmark(signature, bulk, truth, {"nnls": NNLSSolver()})
    second = run_benchmark(signature, bulk, truth, {"nnls": NNLSSolver()})
    pd.testing.assert_frame_equal(
        first.to_frame()[metric_cols], second.to_frame()[metric_cols]
    )


# --- best() direction ------------------------------------------------------


def _report(rmse: float, pearson: float) -> ValidationReport:
    per_type = pd.DataFrame(
        {"rmse": [rmse], "pearson": [pearson], "spearman": [pearson]},
        index=pd.Index(["A"], name="cell_type"),
    )
    return ValidationReport(overall_rmse=rmse, per_type=per_type)


def test_best_selects_by_metric_direction() -> None:
    result = BenchmarkResult(
        reports={"low_rmse": _report(0.1, 0.5), "high_pearson": _report(0.3, 0.9)},
        runtimes={"low_rmse": 1.0, "high_pearson": 2.0},
    )
    assert result.best("overall_rmse") == "low_rmse"
    assert result.best("mean_pearson") == "high_pearson"
    assert result.best("runtime_s") == "low_rmse"


def test_best_unknown_metric_raises() -> None:
    result = BenchmarkResult(reports={"a": _report(0.1, 0.5)}, runtimes={"a": 1.0})
    with pytest.raises(ValueError, match="Unknown metric"):
        result.best("nonsense")


def test_render_and_str() -> None:
    result = BenchmarkResult(reports={"a": _report(0.1, 0.5)}, runtimes={"a": 1.0})
    assert "Benchmark results" in result.render()
    assert str(result) == result.render()


# --- errors ----------------------------------------------------------------


def test_empty_solvers_raises() -> None:
    signature, bulk, truth = _benchmark_inputs()
    with pytest.raises(ValueError, match="at least one solver"):
        run_benchmark(signature, bulk, truth, {})


def test_no_shared_genes_raises() -> None:
    signature, bulk, truth = _benchmark_inputs()
    relabelled = bulk.rename(index={gene: f"X{gene}" for gene in bulk.index})
    with pytest.raises(ValueError, match="share no genes"):
        run_benchmark(signature, relabelled, truth, {"nnls": NNLSSolver()})
