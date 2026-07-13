# ADR-0008 — Plotting isolated; core independent of matplotlib/seaborn

- **Status:** Accepted
- **Milestone:** M3
- **Date:** 2026 (M3)

## Context

M3 introduces the first figure (a signature heatmap). Figures will also be
produced by later milestones (benchmark scatter/bar plots, reports).

## Problem

Decide where plotting code lives and how to keep the computational core free of
the plotting stack.

## Alternatives considered

- Put plotting in `examples/`/`notebooks/` vs inside the package `src/`.
- Use matplotlib's global `pyplot` state vs the object-oriented `Figure` API.
- Let the core import plotting for convenience vs forbidding it.

## Decision

- Plotting lives in a package layer **`src/scdecon/plotting/`** (`figures.py`).
- Figures are drawn with matplotlib's **object-oriented `Figure` API** (no global
  `pyplot`), making rendering headless/CI-safe; seaborn is used for the heatmap.
- The **computational core** (`io`, `preprocessing`, `signature`) imports
  **nothing** from `plotting`, matplotlib, or seaborn. Plotting depends on core
  *outputs* (e.g. a signature `DataFrame`), never the reverse.
- This dependency direction is **enforced by a static test**
  (`tests/unit/test_architecture.py`).

## Rationale

- Plotting is reusable, tested, pipeline-invoked API (called by the CLI/report
  steps in M8 and benchmarks in M5/M6) — it belongs in the versioned, tested
  package, not in un-tested notebooks that drift.
- The OO `Figure` API avoids global state and backend/display requirements.
- Keeping the numeric core free of the plotting stack means solvers and pipelines
  never pull matplotlib/seaborn, and the dependency graph stays acyclic and
  reviewable.

## Consequences

- matplotlib and seaborn become direct dependencies (seaborn has no type stubs →
  a mypy override).
- The architecture guardrail now checks both "core ↛ io" and "core ↛ plotting
  stack".

## Future review conditions

- If figures need interactivity (e.g. Plotly/Streamlit stretch goals), add them
  in the plotting layer or a separate app package — never by importing UI/plot
  libraries into the core.
