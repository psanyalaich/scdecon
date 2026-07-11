"""Unit tests for scripts.datasets.recount3 (gene_sums loader)."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest
from scripts.datasets.recount3 import load_recount3_bulk

_GENE_SUMS = "\n".join(
    [
        "##annotation=G026",
        "##date.generated=2020-08-20",
        "gene_id\tS1\tS2\tS3",
        "ENSG00000278704.1\t10\t20\t0",
        "ENSG00000000003.14\t5\t0\t7",
    ]
)


def _write(path: Path, *, gzip_it: bool, body: str = _GENE_SUMS) -> Path:
    text = body + "\n"
    if gzip_it:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def test_loads_matrix_skipping_metadata(tmp_path: Path) -> None:
    frame = load_recount3_bulk(_write(tmp_path / "gs.tsv", gzip_it=False))
    assert list(frame.columns) == ["S1", "S2", "S3"]
    assert list(frame.index) == ["ENSG00000278704.1", "ENSG00000000003.14"]
    assert frame.index.name == "gene_id"
    assert frame.loc["ENSG00000278704.1", "S2"] == 20
    assert frame.loc["ENSG00000000003.14", "S3"] == 7


def test_reads_gzip(tmp_path: Path) -> None:
    frame = load_recount3_bulk(_write(tmp_path / "gs.tsv.gz", gzip_it=True))
    assert frame.shape == (2, 3)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_recount3_bulk(tmp_path / "nope.tsv")


def test_duplicate_gene_ids_raises(tmp_path: Path) -> None:
    body = "\n".join(
        [
            "##annotation=G026",
            "gene_id\tS1",
            "ENSG001.1\t3",
            "ENSG001.1\t4",
        ]
    )
    path = _write(tmp_path / "dup.tsv", gzip_it=False, body=body)
    with pytest.raises(ValueError, match="duplicate gene IDs"):
        load_recount3_bulk(path)
