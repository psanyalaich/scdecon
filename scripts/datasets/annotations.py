"""Parse a GENCODE GTF into an Ensembl-ID -> gene-symbol mapping.

Dataset-adjacent ingestion (lives in ``scripts/``, not the package). The mapping
it produces is exactly the ``{gene_id -> symbol}`` table that
:func:`scdecon.genes.relabel_gene_index` consumes to harmonise recount3's
Ensembl-indexed bulk onto the symbol-indexed scRNA reference.

GENCODE GTF lines are tab-separated with nine columns; ``gene`` features carry a
``gene_id "..."; ...; gene_name "...";`` attribute string (column nine).
"""

from __future__ import annotations

import re
from pathlib import Path

from scdecon.genes import strip_ensembl_version
from scdecon.logging_utils import get_logger
from scripts.datasets._io import open_text

__all__ = ["parse_gtf_gene_map"]

logger = get_logger("scripts.datasets.annotations")

_GENE_ID_RE = re.compile(r'gene_id "([^"]+)"')
_GENE_NAME_RE = re.compile(r'gene_name "([^"]+)"')
_GENE_FEATURE_INDEX = 2  # column 3 (0-based) is the feature type
_ATTRIBUTES_INDEX = 8  # column 9 (0-based) holds the attribute string
_MIN_GTF_FIELDS = 9


def parse_gtf_gene_map(
    path: str | Path, *, strip_version: bool = True
) -> dict[str, str]:
    """Parse ``gene`` records from a GENCODE GTF into ``{gene_id -> symbol}``.

    Parameters
    ----------
    path:
        Path to a GENCODE GTF (optionally gzip-compressed).
    strip_version:
        If ``True`` (default), Ensembl version suffixes are stripped from the
        keys (``ENSG...\\.N`` -> ``ENSG...``) via
        :func:`scdecon.genes.strip_ensembl_version`, matching the unversioned
        identifiers callers use for harmonisation. The **first** occurrence of an
        identifier wins (deterministic).

    Returns
    -------
    dict[str, str]
        Mapping from gene identifier to gene symbol.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If no ``gene`` records could be parsed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GTF annotation not found: {path}")

    mapping: dict[str, str] = {}
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < _MIN_GTF_FIELDS or fields[_GENE_FEATURE_INDEX] != "gene":
                continue
            attributes = fields[_ATTRIBUTES_INDEX]
            id_match = _GENE_ID_RE.search(attributes)
            name_match = _GENE_NAME_RE.search(attributes)
            if id_match is None or name_match is None:
                continue
            gene_id = id_match.group(1)
            if strip_version:
                gene_id = strip_ensembl_version(gene_id)
            mapping.setdefault(gene_id, name_match.group(1))

    if not mapping:
        raise ValueError(f"No 'gene' records could be parsed from GTF: {path}")
    logger.info("Parsed %d gene records from %s", len(mapping), path.name)
    return mapping
