# ADR-0005 — Marker selection behind an interface; typed method enum

- **Status:** Accepted
- **Milestone:** M3
- **Date:** 2026 (M3)

## Context

M3 builds the signature matrix. The first step is selecting cell-type marker
genes, currently via Scanpy `rank_genes_groups`.

## Problem

Avoid hard-wiring the architecture to Scanpy (so alternative selection strategies
can be added later), and avoid error-prone free-string method configuration.

## Alternatives considered

- A single concrete function calling Scanpy directly (no abstraction).
- A strategy interface with one implementation now (chosen).
- `method: str` free string vs a typed enum.

## Decision

- Introduce an abstract **`MarkerSelector`** strategy with
  `select(adata, config) -> MarkerSet`. The sole v1 implementation is
  **`RankGenesGroupsSelector`** (Scanpy `rank_genes_groups`, one-vs-rest, plus a
  cross-type specificity filter).
- `select_markers(adata, config, selector=None)` is a convenience that defaults
  to `RankGenesGroupsSelector`.
- The DE method is a **`RankMethod(StrEnum)`** (`WILCOXON`, `T_TEST`,
  `T_TEST_OVERESTIM_VAR`, `LOGREG`), validated by `SignatureConfig`.

## Rationale

The interface isolates Scanpy behind a stable contract, so marker-DB or
HVG-prefilter selectors can be added without changing the public API or
downstream `build_signature`. `StrEnum` makes invalid methods unrepresentable and
— being a `str` subclass — passes directly to Scanpy and serialises as its value.

## Consequences

- Slightly more structure now, but real extensibility and type-safety.
- `MarkerSet` (per-cell-type markers, with `genes()` and `to_frame()`) becomes a
  stable contract feeding `build_signature`.
- `RankGenesGroupsSelector` annotates `adata.uns["rank_genes_groups"]` in place
  (metadata only), consistent with the M2 mutation policy.

## Future review conditions

- Add new `MarkerSelector` implementations (marker databases, HVG pre-filter,
  adaptive-N) behind the same interface; revisit `select_markers`'s default if a
  better general strategy emerges.
