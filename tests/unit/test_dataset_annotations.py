"""Unit tests for scripts.datasets.annotations (GTF -> gene-symbol map)."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest
from scripts.datasets.annotations import parse_gtf_gene_map

_GTF = "\n".join(
    [
        "##description: tiny test GTF",
        "##provider: TEST",
        'chr1\tHAVANA\tgene\t1\t100\t.\t+\t.\tgene_id "ENSG001.3"; '
        'gene_type "protein_coding"; gene_name "TP53"; level 2;',
        # A non-gene feature for the same id must be ignored.
        'chr1\tHAVANA\ttranscript\t1\t100\t.\t+\t.\tgene_id "ENSG001.3"; '
        'transcript_id "ENST001.1"; gene_name "TP53";',
        'chr7\tENSEMBL\tgene\t5\t50\t.\t-\t.\tgene_id "ENSG002.1"; gene_name "EGFR";',
    ]
)


def _write(path: Path, *, gzip_it: bool) -> Path:
    text = _GTF + "\n"
    if gzip_it:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def test_parses_gene_map_stripping_versions(tmp_path: Path) -> None:
    mapping = parse_gtf_gene_map(_write(tmp_path / "a.gtf", gzip_it=False))
    assert mapping == {"ENSG001": "TP53", "ENSG002": "EGFR"}


def test_can_keep_versions(tmp_path: Path) -> None:
    mapping = parse_gtf_gene_map(
        _write(tmp_path / "a.gtf", gzip_it=False), strip_version=False
    )
    assert mapping == {"ENSG001.3": "TP53", "ENSG002.1": "EGFR"}


def test_reads_gzip(tmp_path: Path) -> None:
    mapping = parse_gtf_gene_map(_write(tmp_path / "a.gtf.gz", gzip_it=True))
    assert mapping == {"ENSG001": "TP53", "ENSG002": "EGFR"}


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_gtf_gene_map(tmp_path / "nope.gtf")


def test_no_gene_records_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.gtf"
    path.write_text("##only comments\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No 'gene' records"):
        parse_gtf_gene_map(path)
