"""Generic, dataset-agnostic gene-identifier harmonisation utilities.

Real bulk and single-cell datasets frequently label genes in different
namespaces -- for example version-suffixed Ensembl IDs (``ENSG00000141510.16``)
in one and HGNC symbols (``TP53``) in the other. Deconvolution requires a single,
consistent gene index, so callers harmonise the namespaces **before** aligning a
signature and bulk matrix (:func:`scdecon.deconvolution.align_signature_and_bulk`).

This module provides the reusable building blocks for that step. It is
deliberately **generic**: it operates on plain strings and gene-indexed
``DataFrame`` objects plus an already-parsed ``{old_id -> new_id}`` mapping, and
it knows nothing about any particular dataset, file format, or study. Producing
the mapping table (e.g. by parsing an annotation GTF) is a dataset-ingestion
concern that lives outside the package.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd
from pandas.api.types import is_numeric_dtype

from scdecon.logging_utils import get_logger

__all__ = [
    "GeneAggregation",
    "GeneMappingCoverage",
    "compute_mapping_coverage",
    "relabel_gene_index",
    "strip_ensembl_version",
]

#: Default minimum fraction of a matrix's gene identifiers that must be found in
#: the mapping before :func:`relabel_gene_index` logs a low-coverage warning.
#: Exposed as a parameter so the threshold is never hard-coded at call sites.
DEFAULT_MIN_COVERAGE = 0.5

logger = get_logger("genes")


class GeneAggregation(StrEnum):
    """How to combine rows whose identifiers collapse onto the same target label.

    When several source identifiers map to one target (a many-to-one mapping, e.g.
    two Ensembl IDs sharing a symbol), the corresponding rows must be combined.

    Attributes
    ----------
    SUM:
        Add the rows -- appropriate for additive, count-like expression values.
    MEAN:
        Average the rows -- appropriate for already-normalised expression values.
    """

    SUM = "sum"
    MEAN = "mean"


def strip_ensembl_version(gene_id: str) -> str:
    """Remove a trailing ``.<version>`` suffix from an Ensembl gene identifier.

    Ensembl / GENCODE identifiers are often version-suffixed (e.g.
    ``ENSG00000141510.16``); the unversioned stem (``ENSG00000141510``) is what
    maps stably across annotation releases. Only a trailing dot followed by an
    all-digit suffix is stripped; any other identifier (including bare symbols
    such as ``HLA-DRB1``) is returned unchanged.

    Parameters
    ----------
    gene_id:
        A gene identifier.

    Returns
    -------
    str
        ``gene_id`` without its trailing ``.<digits>`` version, if present;
        otherwise ``gene_id`` unchanged.

    Notes
    -----
    Identifiers whose trailing segment is not purely numeric are returned
    unchanged **by design**. This includes GENCODE pseudoautosomal identifiers
    such as ``ENSG00000182162.10_PAR_Y``: the ``_PAR_Y`` suffix makes the trailing
    segment non-numeric, so the version is not stripped and the identifier is
    returned verbatim. This is harmless as long as *both* sides of a harmonisation
    apply this same function: such identifiers then remain byte-identical on each
    side and still match (the chrX and chrY pseudoautosomal copies collapse onto
    one symbol via :func:`relabel_gene_index`). Only a mapping built with a
    *different* normalisation (e.g. fully base identifiers) would fail to match
    them.
    """
    stem, separator, version = gene_id.rpartition(".")
    if separator and version.isdigit():
        return stem
    return gene_id


@dataclass(frozen=True)
class GeneMappingCoverage:
    """Coverage of a gene-identifier mapping over a set of identifiers.

    A lightweight, first-class QC metric for gene-ID harmonisation: it records
    how many identifiers a mapping covers so low overlap (a classic cause of
    unreliable deconvolution) cannot pass unnoticed. Mirrors the summary-object
    style of :class:`~scdecon.preprocessing.QCSummary`.

    Attributes
    ----------
    n_total:
        Number of identifiers considered.
    n_mapped:
        Number of those identifiers present in the mapping.
    """

    n_total: int
    n_mapped: int

    def __post_init__(self) -> None:
        """Validate the counts, failing loudly on impossible values."""
        if self.n_total < 0:
            raise ValueError(f"n_total must be >= 0, got {self.n_total}")
        if not 0 <= self.n_mapped <= self.n_total:
            raise ValueError(
                f"n_mapped must be in [0, n_total={self.n_total}], got {self.n_mapped}"
            )

    @property
    def n_unmapped(self) -> int:
        """Number of identifiers absent from the mapping."""
        return self.n_total - self.n_mapped

    @property
    def fraction_mapped(self) -> float:
        """Fraction of identifiers covered by the mapping (0.0 if none)."""
        if self.n_total == 0:
            return 0.0
        return self.n_mapped / self.n_total

    @property
    def percent_mapped(self) -> float:
        """Percentage of identifiers covered by the mapping."""
        return self.fraction_mapped * 100.0

    def to_dict(self) -> dict[str, int | float]:
        """Return a serialisable summary of the coverage."""
        return {
            "n_total": self.n_total,
            "n_mapped": self.n_mapped,
            "n_unmapped": self.n_unmapped,
            "fraction_mapped": self.fraction_mapped,
        }

    def render(self) -> str:
        """Return a human-readable one-line summary."""
        return (
            f"{self.n_mapped}/{self.n_total} gene identifiers mapped "
            f"({self.percent_mapped:.1f}%), {self.n_unmapped} unmapped"
        )

    def __str__(self) -> str:
        return self.render()


def compute_mapping_coverage(
    identifiers: Iterable[str], mapping: Mapping[str, str]
) -> GeneMappingCoverage:
    """Report how many gene identifiers are covered by a mapping.

    Parameters
    ----------
    identifiers:
        Gene identifiers to check (e.g. a bulk matrix's gene index).
    mapping:
        Mapping keyed by gene identifier.

    Returns
    -------
    GeneMappingCoverage
        Counts of total vs mapped identifiers, with derived fractions.
    """
    ids = [str(identifier) for identifier in identifiers]
    n_mapped = sum(1 for identifier in ids if identifier in mapping)
    return GeneMappingCoverage(n_total=len(ids), n_mapped=n_mapped)


def relabel_gene_index(
    frame: pd.DataFrame,
    mapping: Mapping[str, str],
    *,
    aggregate: GeneAggregation = GeneAggregation.SUM,
    drop_unmapped: bool = True,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> pd.DataFrame:
    """Relabel a gene-indexed matrix from one identifier namespace to another.

    Each row of ``frame`` (indexed by gene identifier) is relabelled through
    ``mapping`` (``{current_id -> target_id}``). Identifiers absent from
    ``mapping`` are unmapped. Because a mapping may be many-to-one (several
    current identifiers sharing a target), rows that collapse onto the same target
    are combined per ``aggregate``. Target labels appear in the order their first
    contributing row appears in ``frame`` (deterministic).

    Parameters
    ----------
    frame:
        Gene-indexed matrix (genes x columns), e.g. a bulk expression matrix.
    mapping:
        Mapping from a current gene identifier to its target identifier.
    aggregate:
        How to combine rows that collapse onto the same target label
        (:class:`GeneAggregation`); ``SUM`` by default.
    drop_unmapped:
        If ``True`` (default), silently discard rows whose identifier is not in
        ``mapping``. If ``False``, raise when any identifier is unmapped.
    min_coverage:
        Minimum fraction of ``frame``'s identifiers that must be found in
        ``mapping`` before a low-coverage warning is logged. In ``[0, 1]``. The
        coverage is always computable via :func:`compute_mapping_coverage`.

    Returns
    -------
    pandas.DataFrame
        A new matrix indexed by target identifier (columns unchanged). The index
        name is preserved from ``frame``.

    Raises
    ------
    ValueError
        If ``min_coverage`` is outside ``[0, 1]``, ``frame`` is empty, its index
        has duplicate identifiers, any column is non-numeric, ``drop_unmapped`` is
        ``False`` and some identifier is unmapped, or no identifier could be
        mapped at all.
    """
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError(f"min_coverage must be in [0, 1], got {min_coverage}")
    if frame.empty:
        raise ValueError("Cannot relabel an empty frame.")
    if frame.index.has_duplicates:
        raise ValueError("Frame has duplicate gene identifiers in its index.")
    non_numeric = [
        str(col) for col in frame.columns if not is_numeric_dtype(frame[col])
    ]
    if non_numeric:
        raise ValueError(
            "relabel_gene_index requires all columns to be numeric expression "
            f"values; found non-numeric column(s): {non_numeric[:5]}."
        )

    current_ids = [str(gene) for gene in frame.index]
    coverage = compute_mapping_coverage(current_ids, mapping)
    if coverage.fraction_mapped < min_coverage:
        logger.warning(
            "Low gene-ID mapping coverage: %s (below min_coverage=%.2f). Estimates "
            "may be unreliable; check that identifier namespaces match (e.g. strip "
            "Ensembl version suffixes first).",
            coverage.render(),
            min_coverage,
        )

    mapped = [
        (pos, mapping[gene]) for pos, gene in enumerate(current_ids) if gene in mapping
    ]

    if coverage.n_unmapped and not drop_unmapped:
        missing = [gene for gene in current_ids if gene not in mapping]
        raise ValueError(
            f"{len(missing)} gene identifier(s) are absent from the mapping "
            f"(e.g. {missing[:5]}). Pass drop_unmapped=True to discard them, or "
            f"supply a mapping that covers them."
        )
    if not mapped:
        raise ValueError(
            "No gene identifiers could be mapped to the target namespace. Check "
            "that the mapping and the frame use compatible identifiers (e.g. strip "
            "Ensembl version suffixes first)."
        )

    positions = [pos for pos, _ in mapped]
    target_labels = [label for _, label in mapped]
    relabelled = frame.iloc[positions].copy()
    relabelled.index = pd.Index(target_labels, name=frame.index.name)

    grouped = relabelled.groupby(level=0, sort=False)
    if aggregate is GeneAggregation.SUM:
        return grouped.sum()
    return grouped.mean()
