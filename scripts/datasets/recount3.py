"""Load a recount3 TCGA ``gene_sums`` matrix into a gene x sample DataFrame.

Dataset-specific ingestion (lives in ``scripts/``, not the package). Verified
format of ``tcga.gene_sums.<PROJECT>.G026.gz``:

- ``##``-prefixed metadata lines (annotation, generation date);
- a header row ``gene_id\\t<sample_id>\\t<sample_id>...``;
- integer coverage-count rows keyed by version-suffixed Ensembl gene IDs
  (GENCODE v26), one column per sample.

The values are recount3 **coverage counts**, not read counts; downstream code is
responsible for any normalisation. Gene-ID harmonisation (Ensembl -> symbol) is a
separate step handled with :mod:`scdecon.genes`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scdecon.logging_utils import get_logger

__all__ = ["load_recount3_bulk"]

logger = get_logger("scripts.datasets.recount3")


def load_recount3_bulk(path: str | Path) -> pd.DataFrame:
    """Load a recount3 ``gene_sums`` matrix as genes (index) x samples (columns).

    Parameters
    ----------
    path:
        Path to a recount3 ``gene_sums`` file (optionally gzip-compressed). The
        ``##`` metadata lines are skipped automatically.

    Returns
    -------
    pandas.DataFrame
        Coverage counts, gene-indexed (Ensembl IDs, ``index.name == "gene_id"``),
        one column per sample. Row and column order are preserved as in the file.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the matrix is empty or has duplicate gene identifiers.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"recount3 gene_sums not found: {path}")

    frame = pd.read_csv(path, sep="\t", comment="#", index_col=0)
    if frame.empty:
        raise ValueError(f"recount3 gene_sums at {path} contains no data.")
    if frame.index.has_duplicates:
        raise ValueError(f"recount3 gene_sums at {path} has duplicate gene IDs.")

    frame.index = frame.index.astype(str)
    frame.index.name = "gene_id"
    logger.info(
        "Loaded recount3 bulk: %d genes x %d samples from %s",
        frame.shape[0],
        frame.shape[1],
        path.name,
    )
    return frame
