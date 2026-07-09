"""Unit tests for scdecon.plotting.figures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scdecon.deconvolution import BenchmarkResult
from scdecon.plotting import (
    plot_benchmark,
    plot_signature_heatmap,
    plot_truth_vs_prediction,
)
from scdecon.validation import ValidationReport


def _signature() -> pd.DataFrame:
    return pd.DataFrame(
        {"A": [10.0, 0.1, 0.0], "B": [0.1, 12.0, 0.2], "C": [0.0, 0.1, 9.0]},
        index=["GA1", "GB1", "GC1"],
    )


def _proportions() -> pd.DataFrame:
    return pd.DataFrame(
        [[0.1, 0.2, 0.3], [0.6, 0.5, 0.4], [0.3, 0.3, 0.3]],
        index=["A", "B", "C"],
        columns=["s1", "s2", "s3"],
    )


def test_plot_creates_file(tmp_path: Path) -> None:
    path = plot_signature_heatmap(_signature(), tmp_path / "heatmap.png")
    assert isinstance(path, Path)
    assert path.is_file()
    assert path.stat().st_size > 0


def test_plot_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "figs" / "sub" / "heatmap.png"
    plot_signature_heatmap(_signature(), target)
    assert target.is_file()


def test_plot_accepts_title(tmp_path: Path) -> None:
    path = plot_signature_heatmap(
        _signature(), tmp_path / "titled.png", title="Signature"
    )
    assert path.is_file()


def test_plot_empty_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        plot_signature_heatmap(pd.DataFrame(), tmp_path / "empty.png")


def test_truth_vs_prediction_creates_file(tmp_path: Path) -> None:
    truth = _proportions()
    path = plot_truth_vs_prediction(truth, truth * 0.9, tmp_path / "scatter.png")
    assert isinstance(path, Path)
    assert path.is_file()
    assert path.stat().st_size > 0


def test_truth_vs_prediction_creates_parent_dirs(tmp_path: Path) -> None:
    truth = _proportions()
    target = tmp_path / "figs" / "scatter.png"
    plot_truth_vs_prediction(truth, truth.copy(), target, title="Recovery")
    assert target.is_file()


def test_truth_vs_prediction_mismatch_raises(tmp_path: Path) -> None:
    truth = _proportions()
    prediction = truth.rename(index={"C": "D"})
    with pytest.raises(ValueError, match="cell types differ"):
        plot_truth_vs_prediction(truth, prediction, tmp_path / "bad.png")


def _benchmark_result() -> BenchmarkResult:
    per_type = pd.DataFrame(
        {"rmse": [0.1], "pearson": [0.9], "spearman": [0.9]},
        index=pd.Index(["A"], name="cell_type"),
    )
    report = ValidationReport(overall_rmse=0.1, per_type=per_type)
    return BenchmarkResult(
        reports={"nnls": report, "robust": report},
        runtimes={"nnls": 0.5, "robust": 0.7},
    )


def test_plot_benchmark_creates_file(tmp_path: Path) -> None:
    path = plot_benchmark(_benchmark_result(), tmp_path / "bench.png")
    assert path.is_file()
    assert path.stat().st_size > 0


def test_plot_benchmark_bad_metric_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="metric"):
        plot_benchmark(_benchmark_result(), tmp_path / "b.png", metric="nope")


def test_plot_benchmark_empty_raises(tmp_path: Path) -> None:
    empty = BenchmarkResult(reports={}, runtimes={})
    with pytest.raises(ValueError, match="empty"):
        plot_benchmark(empty, tmp_path / "empty.png")
