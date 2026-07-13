# ADR-0010 — Real-data ingestion out-of-package; generic gene-ID harmonisation in-package

- **Status:** Accepted
- **Milestone:** M7
- **Date:** 2026-07-10

## Context

M7 applies the existing pipeline to real melanoma data: TCGA-SKCM bulk RNA-seq
(recount3, GENCODE v26, Ensembl gene IDs) and the Tirosh et al. 2016 melanoma
single-cell reference (GEO **GSE72056**, HGNC symbols, values on a
`log2(TPM/10 + 1)` scale). The datasets, accession IDs, and download URLs were
**verified live** during the M7 design review before any code was written.

Two mismatches between these datasets and the existing (M2/M3) pipeline drive the
architecture:

1. **Gene-ID namespaces differ** — bulk is Ensembl (version-suffixed), reference
   is HGNC symbols. `align_signature_and_bulk` currently intersects identifiers
   verbatim (`06_known_issues.md`), so the spaces must be harmonised first.
2. **Expression scales differ** — the reference is `log2(TPM/10 + 1)`, whereas
   `build_signature` assumes natural-`log1p` data and recovers linear space with
   `expm1` (`00_master_context.md §4`). Feeding the reference in verbatim would
   silently corrupt the signature.

## Problem

Decide **what M7 code becomes part of the long-term public package** and what
stays dataset-specific; how gene identifiers are harmonised; and how the reference
scale is reconciled without changing approved M2/M3 contracts.

## Alternatives considered

- **Dataset loaders:** ship reusable `datasets` adapters inside `src/scdecon/`
  vs keep all Tirosh/recount3 parsing in `scripts/`.
- **Harmonisation:** fold symbol↔Ensembl mapping into `align_signature_and_bulk`
  vs a separate, generic gene-ID utility applied before alignment.
- **Scale:** edit `build_signature`/`preprocess` to accept `log2(TPM/10+1)` vs an
  ingestion-side reconstruction to natural `log1p` so the existing contracts hold
  unchanged.

## Decision

1. **The package stays a general deconvolution framework, not a collection of
   dataset adapters.** Only genuinely dataset-agnostic, multi-study-reusable code
   enters `src/scdecon/`. Dataset-specific **download, parsing, and ingestion**
   (Tirosh matrix decoding, recount3 loading, GTF file parsing, scale
   reconstruction) lives in **`scripts/`** and is **not** imported by the package.
2. **Generic gene-ID harmonisation is in-package** as `src/scdecon/genes.py`:
   pure, in-memory utilities operating on strings and gene-indexed DataFrames and
   on an already-parsed identifier mapping. Introduced in P1:
   - `strip_ensembl_version` — drop a trailing `.<version>` from an Ensembl ID.
   - `relabel_gene_index` — relabel a gene-indexed matrix through a
     `{old_id -> new_id}` mapping, collapsing many-to-one collisions
     deterministically (`GeneAggregation.SUM` / `MEAN`) and dropping (or, if
     asked, rejecting) unmapped identifiers.
   `genes.py` reads **no files** and knows **nothing** about TCGA/Tirosh; the
   mapping table (e.g. parsed from the recount3 GTF) is produced by `scripts/` and
   passed in.
3. **Alignment is not changed.** Harmonisation is applied *before* `deconvolve`,
   so `align_signature_and_bulk` continues to receive a single, consistent ID
   space. Folding mapping into alignment remains a deferred, additive option.
4. **Scale is reconciled at ingestion**, in `scripts/`: reconstruct linear TPM
   via `(2^x − 1) × 10`, then hand the existing pipeline an AnnData whose `.X` is
   natural-`log1p(linear)` and whose `layers["counts"]` holds the linear values.
   `build_signature`/`preprocess` are then reused **verbatim**.
5. **Reference cell types** are `{malignant, T, B, Macrophage, Endothelial, CAF,
   NK}`; cells the source marks unresolved are **excluded** from signature
   construction (documented in the tutorial).
6. **Outputs are relative composition estimates, not absolute calibrated
   fractions** (cross-platform: Smart-seq2 TPM reference vs recount3 coverage
   counts). The biological sanity check is the **T-cell/CD8 fraction vs a
   cytotoxicity-signature score** correlation. No over-claiming.

## Rationale

Keeping dataset specifics out of `src/` prevents the framework from accreting
study-specific adapters and the maintenance/API-stability burden they carry, while
still making the *broadly reusable* capability (gene-ID harmonisation) a tested,
first-class part of the package. Reconstructing the scale at ingestion preserves
every approved M2/M3 contract — the central "integrate without architectural debt"
move. Framing results as relative honours the linear-mixing model's cross-platform
limits.

## Consequences

- New in-package public surface (Experimental): `scdecon.genes`
  (`strip_ensembl_version`, `relabel_gene_index`, `GeneAggregation`).
- **No new runtime dependency:** downloads and parsing in `scripts/` use the
  standard library (`urllib`, `hashlib`, `gzip`); `genes.py` uses only pandas.
- New architecture guardrails: `genes.py` must stay generic (no io/plotting/
  anndata/scanpy imports); `scripts/` must not be imported by `src/scdecon/**`.
- `data/` is git-ignored (`raw/`, `interim/`, `processed/`; only `README.md` +
  `.gitkeep` committed). **Data is never committed.** The real end-to-end run and
  its figure/tutorial are produced by the maintainer locally; CI never touches
  real data (tests use tiny synthetic fixtures and `file://`/monkeypatched fetch).

## Future review conditions

- If a *second* study reuses the same ingestion shape, promote only the
  genuinely-shared, dataset-agnostic parts into `src/` (never the study
  specifics).
- Revisit folding gene-ID mapping into `align_signature_and_bulk` if a call-site
  ergonomics need emerges; today it stays a separate, composable step.
