# ADR-0009 — Deconvolution: format-agnostic solver interface + separate alignment

- **Status:** Accepted
- **Milestone:** M4 (Phase 1)
- **Date:** 2026 (M4)

## Context

M4 introduces the bulk side: estimating cell-type proportions `p` from a bulk
sample `b` and the signature matrix `S` (`b ≈ S·p`). The blueprint requires the
deconvolution layer to be *data-format-agnostic* so solvers are pure and
trivially testable.

## Problem

Decide the solver interface, where gene-label handling lives, how proportions are
normalised, how degenerate cases are handled, and how low gene overlap is
treated.

## Alternatives considered

- **Solver input:** labelled `DataFrame` (carries gene labels) vs plain NumPy
  arrays.
- **Alignment:** inside the solver vs a separate adapter.
- **Normalisation:** enforce `Σp = 1` in the solver vs leave to the caller.
- **Degenerate (all-zero) solution:** return zeros vs raise.
- **Low gene overlap:** hard-coded threshold vs configurable; warn vs raise.

## Decision

1. **`Solver` ABC operates on NumPy arrays only.** `fit(signature, bulk) -> p`
   where `signature` is `(n_genes, n_cell_types)`, `bulk` is `(n_genes,)`, and
   `p` is `(n_cell_types,)` with `p ≥ 0`, `Σp = 1`. The deconvolution layer
   imports no pandas/anndata/scanpy/io/plotting in its **solver-core** modules
   (`base.py`, and later `nnls.py`, etc.).
2. **Alignment is a separate concern** (`align.py`,
   `align_signature_and_bulk`): the adapter between labelled (pandas) data and
   the array solvers. It intersects genes, preserves **signature row order**
   (deterministic), returns NumPy arrays plus labels (`AlignedInputs`), and
   raises an informative `ValueError` when no genes are shared.
3. **`Σp = 1` is part of the solver contract** (implementations renormalise).
4. **Degenerate solutions raise `ValueError`** (e.g. an all-zero NNLS result that
   cannot be normalised) with a message explaining likely causes (incompatible
   signature, zero-expression bulk sample, severe preprocessing mismatch).
   Returning zeros would silently hide failure and violate the "output is
   proportions" expectation.
5. **Low gene overlap warns (not raises)** below a **configurable**
   `min_overlap` (default `0.5`), exposed as a parameter — never hard-coded.

## Rationale

Keeping the numerical core free of labels/containers makes solvers pure,
swappable (behind `Solver`), and testable against analytic ground truth (exact
recovery). Alignment is inherently a labelled-data (pandas) concern and belongs
at the boundary, not inside the math. Enforcing `Σp = 1` in the solver gives every
consumer a consistent proportions contract. Failing loud on degenerate inputs
surfaces real problems (e.g. the M7 gene-ID-mismatch pitfall, previewed by the
empty-overlap error).

## Consequences

- SciPy becomes a direct dependency (used by `NNLSSolver` in M4 Phase 2).
- A new static guardrail enforces solver-core format-agnosticism and forbids the
  deconvolution layer from importing io/plotting/anndata/scanpy.
- A thin `deconvolve` orchestrator (M4 Phase 2) will align + loop `fit` over bulk
  samples and return a labelled `DataFrame`, keeping `fit` focused on the math.
- Numerical assumptions of `scipy.optimize.nnls` will be documented in `nnls.py`
  (Phase 2); any added tolerances/safeguards must be justified, not unexplained.

## Future review conditions

- Additional solvers (ν-SVR, robust) in M6 implement the same `Solver` contract.
- Revisit the alignment contract if gene-ID harmonisation (M7) needs mapping
  (symbol↔Ensembl) rather than a plain intersection.
