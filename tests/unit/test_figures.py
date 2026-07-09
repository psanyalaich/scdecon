"""Unit tests for scdecon.plotting.figures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scdecon.plotting import plot_signature_heatmap


def _signature() -> pd.DataFrame:
    return pd.DataFrame(
        {"A": [10.0, 0.1, 0.0], "B": [0.1, 12.0, 0.2], "C": [0.0, 0.1, 9.0]},
        index=["GA1", "GB1", "GC1"],
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
