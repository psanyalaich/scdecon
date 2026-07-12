"""Unit tests for scdecon.validation.metrics."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from scdecon.validation import ValidationReport, align_proportions, evaluate


def _frame(values: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(
        values,
        index=["A", "B", "C"],
        columns=["s1", "s2", "s3", "s4"],
    )


def _truth() -> pd.DataFrame:
    return _frame(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.6, 0.5, 0.4, 0.3],
            [0.3, 0.3, 0.3, 0.3],
        ]
    )


# --- evaluate --------------------------------------------------------------


def test_perfect_prediction() -> None:
    truth = _truth()
    report = evaluate(truth, truth.copy())
    assert report.overall_rmse == pytest.approx(0.0)
    # A and B vary across samples -> Pearson/Spearman == 1; C is constant -> nan.
    assert report.per_type.loc["A", "pearson"] == pytest.approx(1.0)
    assert report.per_type.loc["B", "spearman"] == pytest.approx(1.0)
    assert np.isnan(cast(float, report.per_type.loc["C", "pearson"]))


def test_overall_rmse_hand_computed() -> None:
    truth = _frame([[0.0, 0.0, 0.0, 0.0]] * 3)
    prediction = _frame([[0.2] * 4] * 3)  # every residual is 0.2
    report = evaluate(truth, prediction)
    assert report.overall_rmse == pytest.approx(0.2)


def test_pearson_one_for_positive_linear_map() -> None:
    truth = _truth()
    prediction = truth * 0.5 + 0.1  # positive affine -> Pearson 1
    report = evaluate(truth, prediction)
    assert report.per_type.loc["A", "pearson"] == pytest.approx(1.0)


def test_report_frame_and_aggregates() -> None:
    report = evaluate(_truth(), _truth().copy())
    frame = report.to_frame()
    assert list(frame.columns) == ["rmse", "pearson", "spearman"]
    assert frame.index.name == "cell_type"
    assert report.mean_pearson == pytest.approx(1.0)  # nan (C) is ignored
    assert "overall RMSE" in report.render()
    assert str(report) == report.render()


def test_reindexes_prediction_to_truth_order() -> None:
    truth = _truth()
    shuffled = truth.loc[["C", "A", "B"], ["s4", "s1", "s3", "s2"]]
    report = evaluate(truth, shuffled)
    assert report.overall_rmse == pytest.approx(0.0)


def test_is_deterministic() -> None:
    truth = _truth()
    prediction = truth * 0.9
    first = evaluate(truth, prediction)
    second = evaluate(truth, prediction)
    pd.testing.assert_frame_equal(first.per_type, second.per_type)


# --- align_proportions errors ----------------------------------------------


def test_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        align_proportions(pd.DataFrame(), _truth())


def test_cell_type_mismatch_raises() -> None:
    truth = _truth()
    prediction = truth.rename(index={"C": "D"})
    with pytest.raises(ValueError, match="cell types differ"):
        evaluate(truth, prediction)


def test_sample_mismatch_raises() -> None:
    truth = _truth()
    prediction = truth.rename(columns={"s4": "s9"})
    with pytest.raises(ValueError, match="samples differ"):
        evaluate(truth, prediction)


def test_duplicate_labels_raise() -> None:
    truth = _truth()
    dup = pd.concat([truth, truth.loc[["A"]]])  # duplicate cell-type index
    with pytest.raises(ValueError, match="Duplicate cell-type"):
        align_proportions(dup, truth)


def test_duplicate_sample_labels_raise() -> None:
    truth = _truth()
    dup = pd.concat([truth, truth.iloc[:, [0]]], axis=1)  # duplicate sample column
    with pytest.raises(ValueError, match="Duplicate sample"):
        align_proportions(dup, truth)


def test_report_is_lightweight_and_typed() -> None:
    report = evaluate(_truth(), _truth().copy())
    assert isinstance(report, ValidationReport)
    assert isinstance(report.overall_rmse, float)
