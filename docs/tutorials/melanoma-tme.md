# Tutorial — Melanoma TME deconvolution (TCGA-SKCM)

This walkthrough applies `scdecon` to **real** data: it estimates the cell-type
composition of TCGA-SKCM bulk melanoma tumours using the Tirosh et al. 2016
single-cell melanoma atlas as the reference, and sanity-checks the result against
a bulk cytotoxicity signature.

> **Scope.** Everything dataset-specific here lives in `scripts/` and never
> touches the `scdecon` package. Results are **relative composition estimates**,
> not absolute calibrated fractions (see the caveats at the end).

## Data

All inputs are downloaded (not committed) by the provenance-recording downloader:

```bash
python -m scripts.download_data          # -> data/raw/ + download_manifest.json
```

| Source | Dataset | Scale |
|--------|---------|-------|
| recount3 `tcga.gene_sums.SKCM.G026` | TCGA-SKCM bulk, 473 tumours | exonic coverage **counts**, Ensembl IDs |
| recount3 `human.gene_sums.G026.gtf` | GENCODE v26 annotation | Ensembl → symbol bridge |
| GEO **GSE72056** | Tirosh melanoma scRNA reference | `log2(TPM/10+1)`, gene symbols |

## Run

```bash
python -m scripts.melanoma_workflow      # writes data/processed/
```

Outputs (git-ignored, under `data/processed/`): `proportions.tsv`,
`signature.tsv`, `signature_heatmap.png`, `tnk_vs_cytotoxicity.png`, and
`qc_report.json` (all QC metrics below).

## Pipeline

1. **Load** the recount3 bulk, the GTF map, and the Tirosh reference
   (`scripts.datasets.*`).
2. **Gene-ID harmonisation** (`scdecon.genes`): strip Ensembl versions, measure
   `GeneMappingCoverage`, relabel the bulk to symbols. Coverage was **100%**
   (63,856/63,856) — the identifiers are fully compatible.
3. **Tumour signature** (`scdecon.signature`): Wilcoxon markers + specificity
   filter → a 159-gene × 7-cell-type signature (malignant, T, B, Macrophage,
   Endothelial, CAF, NK). The Tirosh `log2(TPM/10+1)` values are reconstructed to
   linear TPM inside the loader, so the existing signature pipeline applies
   unchanged.
4. **Deconvolution** (`scdecon.deconvolution`, NNLS).
5. **Sanity check**: combined **T+NK** predicted fraction vs a bulk cytotoxicity
   score (`GZMA, GZMB, PRF1, NKG7, GZMH`, mean log-CPM).

## What went wrong first (and why this section exists)

Identifier harmonisation alone was **not** sufficient. The workflow executed
correctly end-to-end, but the initial deconvolution — run directly on the
harmonised recount3 **counts** — produced a biologically implausible result:

- **T fraction ≈ 0.001, NK fraction ≈ 0** across all 473 tumours;
- solver relative residual **≈ 0.70** (the signature explained only ~30% of each
  bulk vector);
- T+NK vs cytotoxicity **Spearman ≈ 0.08** (no relationship).

Yet the signal was clearly present in the data: a **direct T-cell-marker score
correlated with the cytotoxicity score at Spearman ≈ 0.92**. So the identifiers
were right, the score was right, and the workflow ran — the *deconvolution step*
was failing to recover the immune signal the data clearly contained.

### Diagnosis (tested experimentally, not assumed)

The leading hypothesis was an **expression-space mismatch** between the bulk
(coverage counts) and the reference (TPM). We tested candidates rather than
assuming. The table reports the measured results; beneath it, the **Demonstrated**
points are direct experimental findings, and the **Interpretation** is a mechanism
the data are *consistent with* but do **not** prove.

| Transform | T fraction | Spearman(T+NK, cytotoxicity) |
|-----------|-----------|------------------------------|
| Raw counts (baseline) | 0.001 | 0.08 |
| **TPM (gene-length) normalisation** | 0.000 | **nan** (collapsed further) |
| Signature column normalisation | 0.001 | 0.08 |
| **Per-gene relative normalisation** | **0.021** | **0.74** |

**Demonstrated (directly, by experiment):**

- **Gene-length (TPM) normalisation, as implemented here, did not resolve the
  collapse** — the residual barely moved (0.70 → 0.62) and immune fractions stayed
  ≈0. This does **not** exclude other formulations of length normalisation; it
  shows only that length normalisation *as applied here* was not the fix.
- Library-size (CPM) normalisation changes nothing on its own (NNLS is per-sample
  scale-invariant).
- **Per-gene relative normalisation restored** a strong positive T+NK ↔
  cytotoxicity correlation.

**Interpretation (not proven):** the **evidence is most consistent with per-gene
magnitude dominance** — a handful of very high-expression genes dominate the NNLS
least-squares fit, so low-abundance immune-marker genes carry negligible weight.
This is an interpretation, **not** an established fact: because per-gene
normalisation rescales every gene at once, the experiments do **not** uniquely
identify the mechanism. Plausible alternatives that would produce the same
improvement include **signature conditioning / collinearity** (rescaling improves
the separability of the immune columns from the malignant/CAF columns),
**platform-specific per-gene effects** (Smart-seq2 vs bulk detection biases), and
the **generic benefit of feature standardisation** before a least-squares fit.
Per-gene scaling *resolves* the problem; it does not by itself tell us *why*.

### Fix — expression-space harmonisation

`harmonize_expression_space` (in `scripts/melanoma_workflow.py`, method
configurable — `mean` / `max` / `l2`, default `mean`) library-normalises the bulk
(CPM) and then divides **every gene row of both the signature and the bulk** by a
per-gene scale derived from the signature.

**Why this is mathematically reasonable.** Let `D` be the diagonal matrix whose
`i`-th entry is the scaling factor for gene `i`. The transform is a *left
multiplication by the invertible diagonal matrix* `D⁻¹`:

```
S' = D⁻¹ S
B' = D⁻¹ B
```

The linear mixture model `B ≈ S·p` therefore becomes

```
D⁻¹ B ≈ D⁻¹ S·p   ⟺   B' ≈ S'·p
```

with the **same** unknown proportions `p`. Equivalently, the harmonised NNLS
problem is a **weighted NNLS**: minimising

```
‖D⁻¹S·p − D⁻¹B‖²   =   (S·p − B)ᵀ D⁻² (S·p − B)
```

subject to `p ≥ 0`, i.e. ordinary NNLS with per-gene weights `1/Dᵢ²`.

Assumptions and consequences, stated explicitly:

- **`D` must be invertible** — every diagonal entry strictly positive. Genes whose
  signature-derived scale is zero are **dropped before scaling** (not divided by
  zero), so `D⁻¹` is well defined on the retained gene set.
- **The same `D` is applied to both matrices** (it is derived from the signature),
  which is what makes the row-scaling cancel in the structural model.
- **The biological model is unchanged; the estimator changes.** The *true* unknown
  `p` is preserved, but the NNLS *estimate* `p̂` changes — deliberately — because
  the objective is reweighted so every marker gene contributes comparably instead
  of a few high-magnitude genes dominating. (NNLS is additionally per-sample
  scale-invariant, so the CPM step does not change `p̂`; the effect comes entirely
  from the per-gene reweighting by `D⁻²`.)

## Result (before → after)

| Metric | Before (raw counts) | After (harmonised) |
|--------|--------------------:|-------------------:|
| Median relative residual¹ | 0.70 | 0.77 |
| Mean malignant fraction | 0.63 | 0.65 |
| Mean T fraction | 0.001 | 0.021 |
| Mean NK fraction | 0.000 | 0.000 |
| T+NK range | [0.00, 0.12] | [0.00, 0.40] |
| **Spearman(T+NK, cytotoxicity)** | **0.08** (p=0.07) | **0.74** (p≈2e-83) |
| Pearson(T+NK, cytotoxicity) | 0.05 | 0.61 |

After harmonisation the combined T+NK fraction correlates strongly with the bulk
cytotoxicity signature (Spearman 0.74) — the biologically expected direction.
Note, however, that the predicted T/NK fraction and the cytotoxicity score share
cytotoxic marker genes, so the *absolute* correlation is partly mechanical (see
**Caveats**). The informative result is the **contrast** with the baseline
(0.08 → 0.74 on the same genes), which shows the deconvolution now attributes
cytotoxic-gene expression to the T/NK cell types rather than mis-assigning it.

¹ The before and after residuals are measured in **different weighted spaces**
(the "after" residual is the weighted objective `(S·p−B)ᵀD⁻²(S·p−B)`), so their
magnitudes are **not directly comparable**. Before harmonisation the residual is
dominated by a few very high-magnitude genes, so matching those few yields a
*lower* normalised residual even though the decomposition is biologically wrong;
after harmonisation the residual reflects fitting all ~159 markers on an equal
footing — inherently harder. Improving biological *attribution* can therefore
legitimately *increase* the weighted residual. The restored T+NK dynamic range and
the correlation contrast, not the residual, are the decisive metrics.

## Caveats (no over-claiming)

- **Relative, not absolute.** Cross-platform (Smart-seq2 TPM reference vs
  recount3 coverage counts) and per-gene reweighting mean these are relative
  composition estimates. Absolute fractions should not be over-interpreted.
- **NK vs CD8-T is not identifiable.** NK cells share cytotoxic markers
  (`GZMB, NKG7, PRF1`) with CD8 T cells; NNLS assigns ≈0 to NK throughout. The
  reference also does not separate CD8 from CD4 T cells. This is why the primary
  read-out is the **combined T+NK** fraction (secondary T-only / NK-only and a
  KLRD1 sensitivity analysis are recorded in `qc_report.json`).
- **Partial circularity.** The cytotoxicity-score genes overlap the signature's
  T/NK marker genes, so the T+NK-vs-cytotoxicity correlation is **not** an
  independent accuracy measure — it is a **sanity check**. Its value comes from the
  before/after **contrast** (the same shared genes gave only Spearman ≈0.08 before
  harmonisation, so the correlation is not automatic), not from the absolute
  magnitude of 0.74. A fully independent validation would score on genes disjoint
  from the signature, or use an orthogonal readout (survival, published immune
  subtypes, or a second reference).
- **No ground truth.** TCGA has no measured composition, so this is a biological
  sanity check, not a benchmark. Quantitative validation lives in the synthetic
  recovery/benchmark tests.
- **Mechanism not uniquely identified.** Per-gene expression-space normalisation
  *resolves* the immune collapse, but the experiments do not prove *why* (see
  "Diagnosis" — signature conditioning/collinearity, platform-specific per-gene
  effects, and generic feature standardisation remain plausible contributors).
- **Provenance.** Exact URLs, checksums, and sizes are recorded in
  `data/raw/download_manifest.json`; the expression-space method and the rejected
  length-normalisation are recorded in `qc_report.json`
  (`expression_space_diagnostics`).
