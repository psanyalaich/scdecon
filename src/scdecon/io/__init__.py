"""Input/output layer: readers and writers for scdecon data formats."""

from scdecon.io.readers import read_bulk, read_h5ad, read_metadata
from scdecon.io.writers import write_h5ad, write_table

__all__ = [
    "read_bulk",
    "read_h5ad",
    "read_metadata",
    "write_h5ad",
    "write_table",
]
