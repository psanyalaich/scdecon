# scdecon

*Single-cell-reference deconvolution of bulk tumour transcriptomes.*

Bulk RNA-seq measures the **average** expression of a whole tissue — a tumour
biopsy is a blend of cancer cells, T cells, B cells, macrophages, fibroblasts,
and more. **scdecon** estimates *what fraction of a bulk sample comes from each
cell type*, using an annotated single-cell RNA-seq atlas as the reference.

It builds a cell-type **signature matrix** from single-cell data, then solves a
constrained regression to infer cell-type **proportions** in bulk samples —
`Bulk ≈ Signature × Proportions`, subject to non-negativity (`p ≥ 0`) and
sum-to-one (`Σ p = 1`).

## What it does

- **Reference → signature.** Load an annotated `.h5ad`, run QC + normalisation,
  select marker genes, and build a linear-scale signature matrix.
- **Deconvolve bulk.** Estimate per-sample cell-type proportions with a choice of
  solvers (NNLS, ν-SVR, robust) behind one `Solver` interface.
- **Validate against ground truth it generates itself.** Simulate pseudobulk with
  known proportions, then measure recovery (RMSE / Pearson / Spearman).
- **Benchmark solvers fairly** on one shared pseudobulk set.
- **Apply to real data.** A reproducible melanoma (TCGA-SKCM + Tirosh) workflow
  lives in `scripts/` (see the [tutorial](tutorials/melanoma-tme.md)).

## Where to go next

- **[Getting started](getting-started.md)** — install, then run the pipeline
  from the command line or via Snakemake.
- **[CLI reference](cli.md)** — every command, its configuration, and the
  exit-code policy.
- **[API reference](api/index.md)** — the Python package, module by module.

!!! note "Scope"
    scdecon is research software. Quantitative accuracy is validated on
    self-generated synthetic ground truth; real-data results (e.g. the melanoma
    workflow) are **relative** cross-platform estimates, not absolute calibrated
    fractions. See the tutorial for the caveats.
