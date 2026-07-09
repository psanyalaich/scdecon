"""Unit tests for scdecon.deconvolution.align."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from scdecon.deconvolution import AlignedInputs, align_signature_and_bulk

_ALIGN_LOGGER = "scdecon.deconvolution.align"


def _signature(genes: list[str]) -> pd.DataFrame:
    data = np.arange(len(genes) * 2, dtype=float).reshape(len(genes), 2)
    return pd.DataFrame(data, index=genes, columns=["A", "B"])


def _bulk(genes: list[str], samples: list[str]) -> pd.DataFrame:
    data = np.ones((len(genes), len(samples)), dtype=float)
    return pd.DataFrame(data, index=genes, columns=samples)


def test_intersects_and_preserves_signature_order() -> None:
    signature = _signature(["g3", "g1", "g2"])  # deliberately unsorted
    bulk = _bulk(["g1", "g2", "g3", "gX"], ["s1", "s2"])
    aligned = align_signature_and_bulk(signature, bulk)
    assert isinstance(aligned, AlignedInputs)
    assert aligned.genes == ["g3", "g1", "g2"]  # signature row order preserved
    assert aligned.signature.shape == (3, 2)
    assert aligned.bulk.shape == (3, 2)
    assert aligned.cell_types == ["A", "B"]
    assert aligned.sample_names == ["s1", "s2"]


def test_only_shared_genes_kept() -> None:
    signature = _signature(["g1", "g2", "g3"])
    bulk = _bulk(["g2", "g3", "g9"], ["s1"])
    aligned = align_signature_and_bulk(signature, bulk, min_overlap=0.0)
    assert aligned.genes == ["g2", "g3"]


def test_deterministic() -> None:
    signature = _signature(["g3", "g1", "g2"])
    bulk = _bulk(["g1", "g2", "g3"], ["s1"])
    first = align_signature_and_bulk(signature, bulk)
    second = align_signature_and_bulk(signature, bulk)
    assert first.genes == second.genes
    np.testing.assert_array_equal(first.signature, second.signature)
    np.testing.assert_array_equal(first.bulk, second.bulk)


def test_no_shared_genes_raises() -> None:
    signature = _signature(["g1", "g2"])
    bulk = _bulk(["x1", "x2"], ["s1"])
    with pytest.raises(ValueError, match="share no genes"):
        align_signature_and_bulk(signature, bulk)


def test_low_overlap_warns(caplog: pytest.LogCaptureFixture) -> None:
    signature = _signature(["g1", "g2", "g3", "g4"])
    bulk = _bulk(["g1", "gX", "gY", "gZ"], ["s1"])  # only 1/4 overlap
    with caplog.at_level(logging.WARNING, logger=_ALIGN_LOGGER):
        align_signature_and_bulk(signature, bulk, min_overlap=0.5)
    assert any("overlap" in record.message.lower() for record in caplog.records)


def test_high_overlap_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    signature = _signature(["g1", "g2", "g3", "g4"])
    bulk = _bulk(["g1", "g2", "g3", "g4"], ["s1"])
    with caplog.at_level(logging.WARNING, logger=_ALIGN_LOGGER):
        align_signature_and_bulk(signature, bulk, min_overlap=0.5)
    assert not any("overlap" in record.message.lower() for record in caplog.records)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_invalid_min_overlap_raises(bad: float) -> None:
    signature = _signature(["g1", "g2"])
    bulk = _bulk(["g1", "g2"], ["s1"])
    with pytest.raises(ValueError, match="min_overlap"):
        align_signature_and_bulk(signature, bulk, min_overlap=bad)


def test_duplicate_signature_genes_raises() -> None:
    signature = _signature(["g1", "g1", "g2"])
    bulk = _bulk(["g1", "g2"], ["s1"])
    with pytest.raises(ValueError, match="Signature has duplicate"):
        align_signature_and_bulk(signature, bulk)


def test_duplicate_bulk_genes_raises() -> None:
    signature = _signature(["g1", "g2"])
    bulk = _bulk(["g1", "g1", "g2"], ["s1"])
    with pytest.raises(ValueError, match="Bulk matrix has duplicate"):
        align_signature_and_bulk(signature, bulk)
