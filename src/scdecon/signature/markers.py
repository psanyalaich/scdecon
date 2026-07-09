"""Marker-gene selection for signature construction.

Selection is abstracted behind the :class:`MarkerSelector` interface so the
architecture is not tied to Scanpy. :class:`RankGenesGroupsSelector` is the sole
v1 implementation; it ranks genes per cell type with ``rank_genes_groups`` and
then applies a cross-type specificity filter (a gene kept only if it is a top
marker for exactly one cell type).

Selection reads log-normalised expression (as produced by
:mod:`scdecon.preprocessing`). It annotates ``adata.uns["rank_genes_groups"]``
in place (metadata only) and never modifies expression, obs/var, or dimensions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

import anndata
import pandas as pd
import scanpy as sc

from scdecon.logging_utils import get_logger
from scdecon.signature.params import SignatureConfig

__all__ = [
    "MarkerSelector",
    "MarkerSet",
    "RankGenesGroupsSelector",
    "select_markers",
]

logger = get_logger("signature.markers")


@dataclass(frozen=True)
class MarkerSet:
    """Type-specific marker genes, one ordered tuple per cell type."""

    per_type: Mapping[str, tuple[str, ...]]

    def genes(self) -> list[str]:
        """Return the deduplicated marker genes in deterministic order.

        Genes are grouped by sorted cell type and kept in rank order within a
        type; this is the row order of the signature matrix.
        """
        ordered: list[str] = []
        seen: set[str] = set()
        for cell_type in sorted(self.per_type):
            for gene in self.per_type[cell_type]:
                if gene not in seen:
                    seen.add(gene)
                    ordered.append(gene)
        return ordered

    def to_frame(self) -> pd.DataFrame:
        """Return a tidy, serialisable table with columns cell_type, gene, rank."""
        cell_types: list[str] = []
        genes: list[str] = []
        ranks: list[int] = []
        for cell_type in sorted(self.per_type):
            for rank, gene in enumerate(self.per_type[cell_type]):
                cell_types.append(cell_type)
                genes.append(gene)
                ranks.append(rank)
        return pd.DataFrame(
            {"cell_type": cell_types, "gene": genes, "rank": ranks},
            columns=["cell_type", "gene", "rank"],
        )


class MarkerSelector(ABC):
    """Abstract strategy for selecting cell-type marker genes."""

    @abstractmethod
    def select(self, adata: anndata.AnnData, config: SignatureConfig) -> MarkerSet:
        """Return specific marker genes per cell type."""
        raise NotImplementedError


class RankGenesGroupsSelector(MarkerSelector):
    """Marker selection via Scanpy ``rank_genes_groups`` + specificity filter."""

    def select(self, adata: anndata.AnnData, config: SignatureConfig) -> MarkerSet:
        _validate(adata, config)
        sc.tl.rank_genes_groups(
            adata,
            groupby=config.cell_type_key,
            method=str(config.method),
            n_genes=config.n_markers_per_type,
            use_raw=False,
        )
        ranked = _top_genes_per_type(adata, config)
        specific = _drop_shared(ranked)
        _log_summary(ranked, specific)
        return MarkerSet(per_type=specific)


def select_markers(
    adata: anndata.AnnData,
    config: SignatureConfig,
    selector: MarkerSelector | None = None,
) -> MarkerSet:
    """Select cell-type markers using ``selector``.

    Parameters
    ----------
    adata:
        Log-normalised single-cell data. ``adata.uns["rank_genes_groups"]`` is
        annotated in place; expression, obs/var, and dimensions are untouched.
    config:
        Marker-selection parameters.
    selector:
        Strategy to use. Defaults to :class:`RankGenesGroupsSelector`.

    Returns
    -------
    MarkerSet
        Deduplicated, type-specific markers.

    Raises
    ------
    ValueError
        If ``config.cell_type_key`` is absent from ``adata.obs``, there are
        fewer than two cell types, or any cell type has fewer than
        ``config.min_cells_per_type`` cells.
    """
    selector = selector or RankGenesGroupsSelector()
    return selector.select(adata, config)


def _validate(adata: anndata.AnnData, config: SignatureConfig) -> None:
    if config.cell_type_key not in adata.obs.columns:
        raise ValueError(
            f"cell_type_key '{config.cell_type_key}' not found in adata.obs."
        )
    counts = adata.obs[config.cell_type_key].value_counts()
    if len(counts) < 2:
        raise ValueError(
            f"Marker selection requires at least 2 cell types, got {len(counts)}."
        )
    too_small = counts[counts < config.min_cells_per_type]
    if len(too_small) > 0:
        raise ValueError(
            "These cell types have fewer than "
            f"{config.min_cells_per_type} cells: {too_small.to_dict()}."
        )


def _top_genes_per_type(
    adata: anndata.AnnData, config: SignatureConfig
) -> dict[str, list[str]]:
    ranked_df = sc.get.rank_genes_groups_df(adata, group=None)
    result: dict[str, list[str]] = {}
    for cell_type, group_df in ranked_df.groupby("group", sort=True, observed=True):
        genes = group_df["names"].tolist()[: config.n_markers_per_type]
        result[str(cell_type)] = [str(gene) for gene in genes]
    return result


def _drop_shared(ranked: dict[str, list[str]]) -> dict[str, tuple[str, ...]]:
    """Keep only genes that are a top marker for exactly one cell type."""
    occurrences = Counter(gene for genes in ranked.values() for gene in genes)
    return {
        cell_type: tuple(gene for gene in genes if occurrences[gene] == 1)
        for cell_type, genes in ranked.items()
    }


def _log_summary(
    ranked: dict[str, list[str]], specific: dict[str, tuple[str, ...]]
) -> None:
    for cell_type in sorted(ranked):
        n_ranked = len(ranked[cell_type])
        n_kept = len(specific[cell_type])
        logger.info(
            "Cell type %r: %d ranked -> %d specific markers (%d shared dropped).",
            cell_type,
            n_ranked,
            n_kept,
            n_ranked - n_kept,
        )
