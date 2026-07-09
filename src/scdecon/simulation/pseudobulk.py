r"""Pseudobulk simulation: synthetic bulk samples with known ground truth.

A pseudobulk sample reproduces the *physical* generative process of bulk
RNA-seq: it **sums the raw counts of real single cells** drawn at pre-specified
proportions. Because we choose the mixture, the composition is known exactly,
which makes it the gold-standard validation set for deconvolution (no download
required).

Mathematics
-----------
For sample ``j`` with target proportions ``p_j`` over ``C`` cell types and
``N = n_cells_per_sample`` cells:

    n_j = multinomial(N, p_j)                  # integer cells per type, Σ n_j = N
    b_j = Σ over sampled cells of their raw count vectors
    p_true_j = n_j / N                         # REALISED proportions (ground truth)

Cells are sampled **with replacement** within each cell type. The realised
proportions (not the requested ``p_j``) are recorded as ground truth, since those
are what actually went into the sum.

Leakage
-------
The cells used to build the signature must be **disjoint** from the cells used to
simulate pseudobulk, or validation is circular. :func:`split_reference` provides a
deterministic, cell-type-stratified split for exactly this purpose.

Simulator strategy
------------------
Simulation is abstracted behind :class:`BaseSimulator` (mirroring the ``Solver``
and ``MarkerSelector`` interfaces). :class:`CellSumSimulator` is the sole v1
implementation; :func:`simulate_pseudobulk` is a convenience defaulting to it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import anndata
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from scdecon.logging_utils import get_logger
from scdecon.simulation.params import ProportionPrior, SimulationConfig

__all__ = [
    "BaseSimulator",
    "CellSumSimulator",
    "PseudobulkDataset",
    "simulate_pseudobulk",
    "split_reference",
]

logger = get_logger("simulation.pseudobulk")


@dataclass(frozen=True, eq=False)
class PseudobulkDataset:
    """A simulated pseudobulk dataset with its ground-truth proportions.

    Attributes
    ----------
    bulk:
        Summed raw counts, genes (index) x samples (columns).
    proportions:
        Realised ground-truth proportions, cell types (index) x samples
        (columns). Each column is non-negative and sums to 1. The orientation
        matches :func:`scdecon.deconvolution.deconvolve` so metrics can compare
        them directly.
    """

    bulk: pd.DataFrame
    proportions: pd.DataFrame


class BaseSimulator(ABC):
    """Abstract strategy for generating pseudobulk datasets."""

    @abstractmethod
    def simulate(
        self, adata: anndata.AnnData, config: SimulationConfig
    ) -> PseudobulkDataset:
        """Generate a :class:`PseudobulkDataset` from single-cell data."""
        raise NotImplementedError


class CellSumSimulator(BaseSimulator):
    """Pseudobulk by summing raw counts of cells drawn at known proportions."""

    def simulate(
        self, adata: anndata.AnnData, config: SimulationConfig
    ) -> PseudobulkDataset:
        if config.cell_type_key not in adata.obs.columns:
            raise ValueError(
                f"cell_type_key '{config.cell_type_key}' not found in adata.obs."
            )
        counts = _select_counts(adata, config)  # (n_cells, n_genes)
        labels = adata.obs[config.cell_type_key].astype(str).to_numpy()
        genes = [str(gene) for gene in adata.var_names]
        cell_types = sorted(set(labels))
        positions_by_type = {ct: np.where(labels == ct)[0] for ct in cell_types}

        rng = np.random.default_rng(config.random_state)
        n_cells = config.n_cells_per_sample
        n_genes = counts.shape[1]
        n_types = len(cell_types)

        bulk = np.zeros((n_genes, config.n_samples), dtype=np.float64)
        proportions = np.zeros((n_types, config.n_samples), dtype=np.float64)

        for sample in range(config.n_samples):
            weights = _draw_proportions(rng, config, n_types)
            type_counts = rng.multinomial(n_cells, weights)
            proportions[:, sample] = type_counts / n_cells
            column = np.zeros(n_genes, dtype=np.float64)
            for type_index, cell_type in enumerate(cell_types):
                k = int(type_counts[type_index])
                if k == 0:
                    continue
                chosen = rng.choice(positions_by_type[cell_type], size=k, replace=True)
                column += np.asarray(counts[chosen].sum(axis=0), dtype=np.float64)
            bulk[:, sample] = column

        sample_names = [f"sample_{sample}" for sample in range(config.n_samples)]
        bulk_frame = pd.DataFrame(
            bulk, index=pd.Index(genes, name="gene"), columns=sample_names
        )
        proportions_frame = pd.DataFrame(
            proportions,
            index=pd.Index(cell_types, name="cell_type"),
            columns=sample_names,
        )
        logger.info(
            "Simulated %d pseudobulk samples (%d cells each) over %d cell types.",
            config.n_samples,
            n_cells,
            n_types,
        )
        return PseudobulkDataset(bulk=bulk_frame, proportions=proportions_frame)


def simulate_pseudobulk(
    adata: anndata.AnnData,
    config: SimulationConfig,
    simulator: BaseSimulator | None = None,
) -> PseudobulkDataset:
    """Generate a pseudobulk dataset using ``simulator``.

    Parameters
    ----------
    adata:
        Single-cell data (raw counts in ``.X`` or ``config.counts_layer``).
        Should be the **held-out** partition (see :func:`split_reference`) to
        avoid leakage with the signature.
    config:
        Simulation parameters.
    simulator:
        Strategy to use. Defaults to :class:`CellSumSimulator`.

    Returns
    -------
    PseudobulkDataset
        Summed-count bulk (genes × samples) and realised proportions
        (cell types × samples).

    Raises
    ------
    ValueError
        If ``config.cell_type_key`` is absent, or ``config.counts_layer`` is set
        but missing from ``adata.layers``.
    """
    simulator = simulator or CellSumSimulator()
    return simulator.simulate(adata, config)


def split_reference(
    adata: anndata.AnnData,
    cell_type_key: str = "cell_type",
    *,
    signature_fraction: float = 0.5,
    random_state: int = 0,
) -> tuple[anndata.AnnData, anndata.AnnData]:
    """Split cells into disjoint signature and held-out partitions.

    The split is **deterministic** (seeded) and **stratified by cell type**: each
    cell type is shuffled and split independently, so both partitions retain the
    cell-type structure. Use the signature partition to build the signature and
    the held-out partition to simulate pseudobulk, preventing leakage.

    Parameters
    ----------
    adata:
        Single-cell data to split. Not modified (copies are returned).
    cell_type_key:
        Column in ``adata.obs`` to stratify by.
    signature_fraction:
        Fraction of each cell type assigned to the signature partition, in
        ``(0, 1)``.
    random_state:
        Seed for the deterministic shuffle.

    Returns
    -------
    tuple[anndata.AnnData, anndata.AnnData]
        ``(signature_partition, heldout_partition)``, disjoint copies.

    Raises
    ------
    ValueError
        If ``signature_fraction`` is not in ``(0, 1)`` or ``cell_type_key`` is
        absent. (Note: a cell type with very few cells may land entirely on one
        side, depending on rounding.)
    """
    if not 0.0 < signature_fraction < 1.0:
        raise ValueError(
            f"signature_fraction must be in (0, 1), got {signature_fraction}"
        )
    if cell_type_key not in adata.obs.columns:
        raise ValueError(f"cell_type_key '{cell_type_key}' not found in adata.obs.")

    labels = adata.obs[cell_type_key].astype(str).to_numpy()
    rng = np.random.default_rng(random_state)
    signature_positions: list[int] = []
    heldout_positions: list[int] = []
    for cell_type in sorted(set(labels)):
        positions = np.where(labels == cell_type)[0]
        shuffled = rng.permutation(positions)
        n_signature = int(round(signature_fraction * len(shuffled)))
        signature_positions.extend(int(p) for p in shuffled[:n_signature])
        heldout_positions.extend(int(p) for p in shuffled[n_signature:])

    signature_positions.sort()
    heldout_positions.sort()
    return adata[signature_positions].copy(), adata[heldout_positions].copy()


def _draw_proportions(
    rng: np.random.Generator, config: SimulationConfig, n_types: int
) -> NDArray[np.float64]:
    if config.proportion_prior is ProportionPrior.DIRICHLET:
        alpha = np.full(n_types, config.dirichlet_alpha)
        return np.asarray(rng.dirichlet(alpha), dtype=np.float64)
    weights = rng.random(n_types)
    return np.asarray(weights / weights.sum(), dtype=np.float64)


def _select_counts(
    adata: anndata.AnnData, config: SimulationConfig
) -> NDArray[np.float64]:
    if config.counts_layer is None:
        matrix = adata.X
    else:
        if config.counts_layer not in adata.layers:
            raise ValueError(
                f"counts_layer '{config.counts_layer}' not found in adata.layers."
            )
        matrix = adata.layers[config.counts_layer]
    return _to_dense(matrix)


def _to_dense(matrix: Any) -> NDArray[np.float64]:
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray(), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)
