"""Unit tests for scdecon.genes (generic gene-ID harmonisation)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from scdecon.genes import (
    GeneAggregation,
    GeneMappingCoverage,
    compute_mapping_coverage,
    relabel_gene_index,
    strip_ensembl_version,
)

_GENES_LOGGER = "scdecon.genes"


@pytest.mark.parametrize(
    ("gene_id", "expected"),
    [
        ("ENSG00000141510.16", "ENSG00000141510"),
        ("ENSG00000141510.1", "ENSG00000141510"),
        ("ENSG00000141510", "ENSG00000141510"),  # no version -> unchanged
        ("TP53", "TP53"),  # bare symbol -> unchanged
        ("HLA-DRB1", "HLA-DRB1"),  # hyphen, no dot -> unchanged
        ("ENSG00000141510.16_PAR_Y", "ENSG00000141510.16_PAR_Y"),  # non-digit tail
        ("gene.name.2", "gene.name"),  # only the final dot-int is stripped
    ],
)
def test_strip_ensembl_version(gene_id: str, expected: str) -> None:
    assert strip_ensembl_version(gene_id) == expected


def _frame(genes: list[str], samples: list[str]) -> pd.DataFrame:
    data = np.arange(len(genes) * len(samples), dtype=float).reshape(
        len(genes), len(samples)
    )
    return pd.DataFrame(data, index=pd.Index(genes, name="gene_id"), columns=samples)


def test_relabel_simple_one_to_one() -> None:
    frame = _frame(["e1", "e2", "e3"], ["s1", "s2"])
    mapping = {"e1": "A", "e2": "B", "e3": "C"}
    result = relabel_gene_index(frame, mapping)
    assert list(result.index) == ["A", "B", "C"]
    assert list(result.columns) == ["s1", "s2"]
    assert result.index.name == "gene_id"
    np.testing.assert_array_equal(result.to_numpy(), frame.to_numpy())


def test_relabel_preserves_first_appearance_order() -> None:
    # Targets should appear in the order their first contributing row appears,
    # not sorted alphabetically.
    frame = _frame(["e3", "e1", "e2"], ["s1"])
    mapping = {"e3": "Z", "e1": "Y", "e2": "X"}
    result = relabel_gene_index(frame, mapping)
    assert list(result.index) == ["Z", "Y", "X"]


def test_relabel_collapses_collisions_by_sum() -> None:
    frame = _frame(["e1", "e2", "e3"], ["s1", "s2"])  # rows [0,1],[2,3],[4,5]
    mapping = {"e1": "A", "e2": "A", "e3": "B"}  # e1+e2 -> A
    result = relabel_gene_index(frame, mapping, aggregate=GeneAggregation.SUM)
    assert list(result.index) == ["A", "B"]
    np.testing.assert_array_equal(result.loc["A"].to_numpy(), [0 + 2, 1 + 3])
    np.testing.assert_array_equal(result.loc["B"].to_numpy(), [4, 5])


def test_relabel_collapses_collisions_by_mean() -> None:
    frame = _frame(["e1", "e2", "e3"], ["s1", "s2"])
    mapping = {"e1": "A", "e2": "A", "e3": "B"}
    result = relabel_gene_index(frame, mapping, aggregate=GeneAggregation.MEAN)
    np.testing.assert_array_equal(result.loc["A"].to_numpy(), [1.0, 2.0])  # means
    np.testing.assert_array_equal(result.loc["B"].to_numpy(), [4, 5])


def test_relabel_drops_unmapped_by_default() -> None:
    frame = _frame(["e1", "e2", "e3"], ["s1"])
    mapping = {"e1": "A", "e3": "C"}  # e2 unmapped
    result = relabel_gene_index(frame, mapping)
    assert list(result.index) == ["A", "C"]


def test_relabel_unmapped_raises_when_not_dropping() -> None:
    frame = _frame(["e1", "e2"], ["s1"])
    mapping = {"e1": "A"}  # e2 unmapped
    with pytest.raises(ValueError, match="absent from the mapping"):
        relabel_gene_index(frame, mapping, drop_unmapped=False)


def test_relabel_no_mappable_genes_raises() -> None:
    frame = _frame(["e1", "e2"], ["s1"])
    mapping = {"x1": "A"}
    with pytest.raises(ValueError, match="No gene identifiers could be mapped"):
        relabel_gene_index(frame, mapping)


def test_relabel_empty_frame_raises() -> None:
    frame = pd.DataFrame(index=pd.Index([], name="gene_id"), columns=["s1"])
    with pytest.raises(ValueError, match="empty frame"):
        relabel_gene_index(frame, {"e1": "A"})


def test_relabel_duplicate_index_raises() -> None:
    frame = _frame(["e1", "e1", "e2"], ["s1"])
    with pytest.raises(ValueError, match="duplicate gene identifiers"):
        relabel_gene_index(frame, {"e1": "A", "e2": "B"})


def test_relabel_is_deterministic() -> None:
    frame = _frame(["e1", "e2", "e3"], ["s1", "s2"])
    mapping = {"e1": "A", "e2": "A", "e3": "B"}
    first = relabel_gene_index(frame, mapping)
    second = relabel_gene_index(frame, mapping)
    assert list(first.index) == list(second.index)
    np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())


def test_relabel_does_not_mutate_input() -> None:
    frame = _frame(["e1", "e2"], ["s1"])
    before = frame.copy()
    relabel_gene_index(frame, {"e1": "A", "e2": "B"})
    pd.testing.assert_frame_equal(frame, before)


def test_relabel_composes_with_version_stripping() -> None:
    # The intended M7 usage: strip Ensembl versions, then map to symbols.
    frame = _frame(["ENSG1.5", "ENSG2.3"], ["s1"])
    frame.index = frame.index.map(strip_ensembl_version)
    mapping = {"ENSG1": "TP53", "ENSG2": "EGFR"}
    result = relabel_gene_index(frame, mapping)
    assert list(result.index) == ["TP53", "EGFR"]


def test_relabel_rejects_non_numeric_columns() -> None:
    frame = pd.DataFrame(
        {"s1": [1.0, 2.0], "label": ["x", "y"]},
        index=pd.Index(["e1", "e2"], name="gene_id"),
    )
    with pytest.raises(ValueError, match="numeric"):
        relabel_gene_index(frame, {"e1": "A", "e2": "B"})


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_relabel_invalid_min_coverage_raises(bad: float) -> None:
    frame = _frame(["e1"], ["s1"])
    with pytest.raises(ValueError, match="min_coverage"):
        relabel_gene_index(frame, {"e1": "A"}, min_coverage=bad)


def test_relabel_warns_on_low_coverage(caplog: pytest.LogCaptureFixture) -> None:
    frame = _frame(["e1", "e2", "e3", "e4"], ["s1"])
    mapping = {"e1": "A"}  # 1/4 = 25% coverage
    with caplog.at_level(logging.WARNING, logger=_GENES_LOGGER):
        relabel_gene_index(frame, mapping, min_coverage=0.5)
    assert any("coverage" in record.message.lower() for record in caplog.records)


def test_relabel_does_not_warn_on_full_coverage(
    caplog: pytest.LogCaptureFixture,
) -> None:
    frame = _frame(["e1", "e2"], ["s1"])
    mapping = {"e1": "A", "e2": "B"}
    with caplog.at_level(logging.WARNING, logger=_GENES_LOGGER):
        relabel_gene_index(frame, mapping, min_coverage=0.5)
    assert not any("coverage" in record.message.lower() for record in caplog.records)


# --- Mapping-coverage QC metric -------------------------------------------


def test_mapping_coverage_counts_and_derived() -> None:
    coverage = compute_mapping_coverage(
        ["e1", "e2", "e3", "e4"], {"e1": "A", "e3": "C"}
    )
    assert coverage.n_total == 4
    assert coverage.n_mapped == 2
    assert coverage.n_unmapped == 2
    assert coverage.fraction_mapped == 0.5
    assert coverage.percent_mapped == 50.0


def test_mapping_coverage_to_dict_and_render() -> None:
    coverage = compute_mapping_coverage(["e1", "e2"], {"e1": "A"})
    assert coverage.to_dict() == {
        "n_total": 2,
        "n_mapped": 1,
        "n_unmapped": 1,
        "fraction_mapped": 0.5,
    }
    assert "1/2" in coverage.render()
    assert str(coverage) == coverage.render()


def test_mapping_coverage_empty_is_zero() -> None:
    coverage = compute_mapping_coverage([], {"e1": "A"})
    assert coverage.n_total == 0
    assert coverage.n_mapped == 0
    assert coverage.fraction_mapped == 0.0


@pytest.mark.parametrize(
    ("n_total", "n_mapped"),
    [(2, 3), (-1, 0), (5, -1)],
)
def test_mapping_coverage_validates(n_total: int, n_mapped: int) -> None:
    with pytest.raises(ValueError):
        GeneMappingCoverage(n_total=n_total, n_mapped=n_mapped)
