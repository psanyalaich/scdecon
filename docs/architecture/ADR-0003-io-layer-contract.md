# ADR-0003 — I/O layer: faithful readers/writers, fixed orientation

- **Status:** Accepted
- **Milestone:** M1
- **Date:** 2026 (M1)

## Context

The package needs to load an annotated scRNA `.h5ad`, bulk expression matrices,
and metadata, and to write tables/AnnData back — as the bottom layer everything
else builds on.

## Problem

Define an I/O contract that is reliable, testable, and cannot silently corrupt or
mis-orient scientific data.

## Alternatives considered

- Readers that also normalise/clean/reorder data vs readers that only validate
  structure and preserve data exactly.
- Flexible/auto-detected matrix orientation vs a single fixed convention.
- Duplicating separator logic in readers and writers vs sharing one helper.

## Decision

- Public API: `read_h5ad`, `read_bulk`, `read_metadata`, `write_table`,
  `write_h5ad`. All accept `str | Path` and normalise to `Path` immediately.
- Readers perform **structural validation only** (existence, empty table,
  duplicate identifiers, non-numeric where required) and **preserve data exactly**
  (no normalisation/filtering/reordering). `read_h5ad` returns sparse `.X`
  untouched.
- Bulk matrices use a fixed **genes × samples** orientation (first column = gene
  index). Separator inferred from suffix (`.tsv`→tab, `.csv`→comma); other
  suffixes require explicit `sep`.
- A single private `_resolve_separator` is shared by readers and writers so read
  and write formats cannot drift.
- Errors are user-oriented `ValueError`/`FileNotFoundError` with actionable
  messages; low-level pandas errors are wrapped.

## Rationale

Deconvolution later aligns bulk to the signature by **gene index**; a fixed
orientation plus byte-faithful preservation prevents silent transposition and
scale bugs. Keeping readers "dumb" makes them trivially testable and keeps
scientific transforms in the layers that own them.

## Consequences

- Downstream code can rely on a stable, gene-indexed contract.
- Dense/sparse handling is deferred to consumers (preprocessing/signature).
- `write_table` writes the index as column 0 so files round-trip through
  `read_bulk`.

## Future review conditions

- If a second consumer needs the separator logic publicly, promote
  `_resolve_separator`. If large-matrix performance demands it (M7+), add
  parquet/streaming/Polars paths behind the same API.
