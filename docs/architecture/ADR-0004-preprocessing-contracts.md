# ADR-0004 — Preprocessing: config-driven, explicit mutation contracts

- **Status:** Accepted
- **Milestone:** M2
- **Date:** 2026 (M2)

## Context

Turn raw single-cell counts into analysis-ready data (QC filtering +
normalisation) without hidden constants and without losing raw counts.

## Problem

Decide how parameters are supplied, how functions mutate `AnnData`, how raw
counts are preserved, and what `preprocess` returns.

## Alternatives considered

- **Parameters:** free kwargs vs a frozen dataclass vs pulling M8's Pydantic
  config forward.
- **Mutation:** always copy (pure) vs in-place (Scanpy convention).
- **Raw counts:** `adata.raw` vs a named `layers[...]` vs recompute.
- **`preprocess` return:** `AnnData` vs `tuple[AnnData, QCSummary]` vs summary in
  `.uns`.

## Decision

- All parameters live in a **frozen `PreprocessConfig`** (no magic numbers).
  Pulling Pydantic (`config.py`) forward is rejected — that is M8 scope; M8 will
  construct/adapt `PreprocessConfig`.
- **Mutation contracts:**
  - `compute_qc_metrics` annotates `.obs`/`.var` **in place** (metadata only),
    returns the same object; never touches `.X`/layers/names/shape; recomputes
    deterministically if run again.
  - `filter_cells_and_genes` returns a **new** filtered AnnData + a typed
    `QCSummary`; the input is untouched (dimension change is a true
    transformation).
  - `normalize` transforms `.X` **in place** (its explicit purpose) after copying
    raw counts to `layers[config.counts_layer]`; **raises** if that layer already
    exists (no double-normalisation).
  - `preprocess` mutates the supplied `adata`'s metadata in place (no auto-copy;
    immutability is opt-in via `.copy()`) and returns a **new** filtered +
    normalised `AnnData`.
- Raw counts are preserved **once, consistently** in `layers[config.counts_layer]`
  (default `"counts"`).
- The QC summary travels as a **serialisable dict** in
  `result.uns["scdecon_qc_summary"]` (namespaced key), keeping `preprocess`'s
  return type `AnnData`.

## Rationale

In-place metadata annotation is cheap and matches Scanpy conventions; dimension
changes should be non-destructive; a single raw-count strategy avoids
inconsistency; a namespaced `.uns` key avoids collisions and keeps the summary
serialisable (a dataclass is not) so it survives `write_h5ad`.

## Consequences

- Scanpy becomes a dependency (installs cleanly on 3.11–3.14).
- Callers must pass `.copy()` if they need the original untouched.
- The QC summary is available typed (`QCSummary`) from `filter_cells_and_genes`
  and serialisable (dict) from `preprocess`.

## Future review conditions

- If provenance/QC needs richer persistence, extend the `.uns` payload or
  introduce a typed result object; revisit the copy policy if memory at scale
  (M7+) becomes a concern.
