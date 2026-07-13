# ADR-0006 — Signature matrix as a validated DataFrame

- **Status:** Accepted
- **Milestone:** M3
- **Date:** 2026 (M3)

## Context

`build_signature` produces the signature matrix `S` (genes × cell types), the
central artifact of the reference side and the input to the future deconvolution
solvers (M4+).

## Problem

Should the public return type be a bare `pandas.DataFrame` or a dedicated
`SignatureMatrix` dataclass (carrying provenance, invariants, serialisation)?

## Alternatives considered

- **`SignatureMatrix` wrapper:** typed identity, invariant enforcement, a home
  for provenance (config, markers, scale). Costs: custom `__eq__`/`__hash__`,
  conversion methods, a provenance-serialisation sidecar, and per-consumer
  unwrapping.
- **Validated `DataFrame`** (chosen): a documented contract + an internal
  validator.

## Decision

Return a **validated `pandas.DataFrame`**: index = **exactly `MarkerSet.genes()`**
(a public reproducibility guarantee), columns = sorted cell types, values =
linear-scale means. An internal `_validate_signature_frame` enforces the contract
(non-empty, unique gene index, numeric, finite, non-negative). No
`SignatureMatrix` for now.

## Rationale

- The primary consumer (M4 `Solver`) is deliberately data-format-agnostic and
  wants a matrix + gene index — a DataFrame provides both with zero ceremony
  (`.to_numpy()`, `.index`).
- Provenance (config, markers, scale) is already available at every call site, so
  it does not need to be smuggled through the return value.
- Serialisation is already solved (`io.write_table`/`read_bulk` round-trip); a
  wrapper would need a provenance sidecar we do not yet need.
- YAGNI: we do not yet know the exact provenance schema; committing to one now
  risks more churn than a future, controlled migration.

## Consequences

- Trivial interop with NumPy/pandas/Scanpy and the solvers; simple testing
  (`assert_frame_equal`).
- The frame contract (documented + validated) provides most of a wrapper's safety
  without its cost.
- Provenance for reports/plots is passed explicitly from call sites (M5/M6).

## Future review conditions

- Introduce `SignatureMatrix` **only** when provenance must be persisted *with*
  the matrix (likely M8/M9). It should wrap the same validated frame, expose
  `to_frame()`/`to_numpy()`, and reuse `_validate_signature_frame` via a
  `from_frame()` classmethod, so solvers and serialisation are unaffected. At
  that point, consider promoting `_validate_signature_frame` to a shared public
  validation utility (it is intentionally private while it has a single consumer).
