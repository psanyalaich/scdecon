# `data/` — local data (never committed)

This directory holds the raw and derived data for the M7 real-data workflow. **Its
contents are git-ignored** (see `.gitignore`); only this `README.md` and
`.gitkeep` are tracked. Never commit data files.

## Layout

```
data/
  raw/         # downloaded inputs, exactly as fetched (populated by the script)
  interim/     # intermediate artefacts (e.g. harmonised bulk)   [created as needed]
  processed/   # final results (proportions, figures)            [created as needed]
  download_manifest.json   # provenance: URLs, checksums, sizes, timestamps
```

## Fetching the inputs

Run the idempotent downloader (standard library only; re-running skips files that
already exist unless `--force` is given):

```bash
python -m scripts.download_data            # download all sources into data/raw/
python -m scripts.download_data --force    # re-download, overwriting
python -m scripts.download_data --only tcga_gene_sums gencode_gtf
```

It writes `data/raw/download_manifest.json` recording, for each source, the URL,
accession, SHA-256, byte size, and retrieval time — so a run is reproducible and
auditable even though the data itself is never committed.

## Sources (verified 2026-07-10)

| Name | Dataset | Provenance |
|------|---------|-----------|
| `tcga_gene_sums` | TCGA-SKCM bulk coverage counts | recount3 (GENCODE v26, Ensembl IDs) |
| `tcga_metadata` | TCGA-SKCM sample metadata | recount3 |
| `gencode_gtf` | GENCODE v26 gene annotation | recount3 (gene-ID → symbol bridge) |
| `tirosh_scrna` | Melanoma scRNA reference | GEO GSE72056 (Tirosh et al. 2016) |
