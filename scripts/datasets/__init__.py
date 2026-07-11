"""Dataset-specific ingestion adapters (Tirosh scRNA, recount3 TCGA, GENCODE GTF).

These modules parse the exact on-disk formats of specific datasets and produce
in-memory objects the generic ``scdecon`` pipeline can consume. They are
deliberately not part of the installable package.
"""
