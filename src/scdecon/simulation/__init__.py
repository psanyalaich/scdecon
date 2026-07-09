"""Pseudobulk simulation: synthetic bulk with known ground-truth proportions.

Operates purely on in-memory AnnData objects (no file I/O). See
:func:`~scdecon.simulation.pseudobulk.split_reference` to create disjoint
signature / held-out partitions and avoid leakage.
"""

from scdecon.simulation.params import ProportionPrior, SimulationConfig
from scdecon.simulation.pseudobulk import (
    BaseSimulator,
    CellSumSimulator,
    PseudobulkDataset,
    simulate_pseudobulk,
    split_reference,
)

__all__ = [
    "BaseSimulator",
    "CellSumSimulator",
    "ProportionPrior",
    "PseudobulkDataset",
    "SimulationConfig",
    "simulate_pseudobulk",
    "split_reference",
]
