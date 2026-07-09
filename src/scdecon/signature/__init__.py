"""Signature construction: marker selection and signature-matrix assembly.

This layer operates purely on in-memory AnnData objects and pandas structures;
it performs no file I/O of its own (saving is the caller's responsibility via
:mod:`scdecon.io`).
"""

from scdecon.signature.build import build_signature
from scdecon.signature.markers import (
    MarkerSelector,
    MarkerSet,
    RankGenesGroupsSelector,
    select_markers,
)
from scdecon.signature.params import RankMethod, SignatureConfig

__all__ = [
    "MarkerSelector",
    "MarkerSet",
    "RankGenesGroupsSelector",
    "RankMethod",
    "SignatureConfig",
    "build_signature",
    "select_markers",
]
