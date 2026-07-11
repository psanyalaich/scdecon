"""Dataset-specific scripts and ingestion for scdecon (NOT part of the package).

Everything under ``scripts/`` is intentionally outside the installable
``scdecon`` package (ADR-0010): it hard-codes the layout, encoding, and download
locations of specific datasets (TCGA-SKCM via recount3, the Tirosh melanoma
scRNA reference). The package must never import from here; only generic,
dataset-agnostic capabilities (e.g. gene-ID harmonisation in ``scdecon.genes``)
belong in ``src/``.
"""
