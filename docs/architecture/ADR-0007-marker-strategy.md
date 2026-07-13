# ADR-0007 — Marker strategy: Wilcoxon + specificity, fixed-N, linear mean

- **Status:** Accepted
- **Milestone:** M3
- **Date:** 2026 (M3)

## Context

The signature matrix's quality depends on (a) which genes are chosen as markers
and (b) what expression values represent each cell type.

## Problem

Choose a marker-selection method, a marker-count policy, and a profile statistic
that make the signature discriminative and consistent with the deconvolution
model `b ≈ S·p`.

## Alternatives considered

- **Selection method:** Wilcoxon one-vs-rest (`rank_genes_groups`); fold-change
  only; t-test; highly variable genes (HVG); curated marker databases.
- **Marker count:** fixed top-N per type vs adaptive thresholds (FDR / effect
  size).
- **Profile statistic:** arithmetic mean vs median vs trimmed mean.
- **Scale:** linear vs log for the signature values.

## Decision

- **Wilcoxon one-vs-rest** ranking, take a **fixed top-N** per cell type, then
  **drop genes that are top markers for more than one type** (cross-type
  specificity filter).
- Signature values are **arithmetic-mean, linear-scale** profiles:
  `mean(expm1(adata.X))` per cell type over the marker genes. Marker *selection*
  runs on the log-normalised data; profile *values* are linear.

## Rationale

- **Wilcoxon** is non-parametric — robust to scRNA's zero-inflated, non-normal
  counts (beats t-test); it uses between-type contrast, which conditions `S` well
  (beats HVG, which measures global variance, not between-type specificity).
- **Fixed top-N** gives every cell type equal footprint, keeping `S` balanced and
  well-conditioned; FDR/effect-size cutoffs yield wildly uneven counts and add
  interacting knobs without proven v1 benefit. Effect size is still used
  implicitly (top-N of rank-ordered scores).
- **Arithmetic mean** is the moment-matching estimator that makes `b ≈ S·p` exact
  in expectation (bulk is physically a sum of cells); median zeros out
  zero-inflated genes and trimmed mean discards genuinely-contributing cells —
  both break additivity.
- **Linear scale** is required for additive mixing; using log means would violate
  `b ≈ S·p`. This is the classic pitfall the design explicitly avoids.
- Marker databases add an external, versioned dependency and gene-ID-mapping
  burden, contradicting data-driven, self-validating v1.

## Consequences

- Discriminative, reproducible signatures with visible block structure.
- One-vs-rest can favour abundant types and gives weak signal to genes shared
  across types (a gene high in 2 of few types may barely rank) — documented; the
  `MarkerSelector` abstraction allows alternative strategies later.
- The specificity filter and linear/log split are covered by unit tests.

## Future review conditions

- Revisit with: cross-subject variance weighting (MuSiC-style), adaptive marker
  counts / FDR pre-filter, robust location statistics, or HVG pre-filtering for
  speed at scale — each addable behind the existing `MarkerSelector` /
  `build_signature` contracts.
