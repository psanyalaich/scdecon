"""Unit tests for scripts.datasets.tirosh (GSE72056 loader)."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest
from scripts.datasets.tirosh import load_tirosh_reference

# 4 cells x 3 genes. Cells:
#   c1: malignant=1 (no), type=2 -> B
#   c2: malignant=2 (yes)        -> malignant
#   c3: malignant=1 (no), type=3 -> Macrophage
#   c4: malignant=0              -> unresolved
_MATRIX = "\n".join(
    [
        "Cell\tc1\tc2\tc3\tc4",
        "tumor\t1\t1\t2\t2",
        '"malignant(1=no,2=yes,0=unresolved)"\t1\t2\t1\t0',
        '"non-malignant cell type (1=T,2=B,3=Macro,4=Endo,5=CAF,6=NK)"\t2\t0\t3\t0',
        "GENE1\t1\t2\t0\t1",
        "GENE2\t0\t1\t1\t0",
        "GENE3\t3\t0\t2\t1",
    ]
)


def _write(path: Path, *, gzip_it: bool, body: str = _MATRIX) -> Path:
    text = body + "\n"
    if gzip_it:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def test_decodes_cell_types_and_excludes_unresolved(tmp_path: Path) -> None:
    adata = load_tirosh_reference(_write(tmp_path / "m.txt", gzip_it=False))
    assert adata.n_obs == 3  # c4 (unresolved) dropped
    assert adata.n_vars == 3
    assert list(adata.obs_names) == ["c1", "c2", "c3"]
    assert list(adata.obs["cell_type"]) == ["B", "malignant", "Macrophage"]
    assert list(adata.var_names) == ["GENE1", "GENE2", "GENE3"]
    assert list(adata.obs["tumor"]) == ["1", "1", "2"]


def test_can_keep_unresolved(tmp_path: Path) -> None:
    adata = load_tirosh_reference(
        _write(tmp_path / "m.txt", gzip_it=False), exclude_unresolved=False
    )
    assert adata.n_obs == 4
    assert adata.obs["cell_type"].tolist() == [
        "B",
        "malignant",
        "Macrophage",
        "unresolved",
    ]


def test_reconstructs_linear_and_log1p_scale(tmp_path: Path) -> None:
    adata = load_tirosh_reference(_write(tmp_path / "m.txt", gzip_it=False))
    # log2(TPM/10+1) value 1 for (c1, GENE1) -> linear (2**1 - 1) * 10 = 10.
    counts = adata.to_df(layer="counts")
    assert counts.loc["c1", "GENE1"] == pytest.approx(10.0)
    # value 0 -> linear 0.
    assert counts.loc["c1", "GENE2"] == pytest.approx(0.0)
    # .X is natural log1p of the linear values.
    x = adata.to_df()
    assert x.loc["c1", "GENE1"] == pytest.approx(np.log1p(10.0))
    assert x.loc["c1", "GENE2"] == pytest.approx(0.0)


def test_reads_gzip(tmp_path: Path) -> None:
    adata = load_tirosh_reference(_write(tmp_path / "m.txt.gz", gzip_it=True))
    assert adata.n_obs == 3


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_tirosh_reference(tmp_path / "nope.txt")


def test_bad_header_raises(tmp_path: Path) -> None:
    body = _MATRIX.replace("Cell\t", "NOTCELL\t", 1)
    path = _write(tmp_path / "bad.txt", gzip_it=False, body=body)
    with pytest.raises(ValueError, match="first field should be 'Cell'"):
        load_tirosh_reference(path)


def test_deduplicates_gene_symbols(tmp_path: Path) -> None:
    body = "\n".join([_MATRIX, "GENE1\t9\t9\t9\t9"])  # duplicate GENE1
    path = _write(tmp_path / "dup.txt", gzip_it=False, body=body)
    adata = load_tirosh_reference(path)
    assert list(adata.var_names) == ["GENE1", "GENE2", "GENE3"]
    # First occurrence kept: (c1, GENE1) linear stays 10, not the duplicate's value.
    assert adata.to_df(layer="counts").loc["c1", "GENE1"] == pytest.approx(10.0)
