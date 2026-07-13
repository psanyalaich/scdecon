# ADR-0013 — CLI as a thin composition root; declarative config; Snakemake calls the CLI

- **Status:** Accepted
- **Milestone:** M8
- **Date:** 2026-07-12

## Context

M8 adds the first user-facing interface layer: a Typer CLI (`scdecon.cli`), a
Pydantic run-configuration schema (`scdecon.config`, ADR follows the shape agreed
in Session 1), and a Snakemake workflow. The scientific package (M1–M7) is
complete and treated as the product; these three pieces are *interface and
orchestration* around it, not new science.

The blueprint's non-negotiables apply: science stays out of the CLI; the frozen
parameter dataclasses remain the single source of truth; no new abstractions
unless required; dataset-specific code stays in `scripts/` and the package never
imports it.

## Problem

Decide, for the interface layer:

1. Where solver **construction** happens (turning a configured name/params into a
   concrete `Solver`).
2. Whether the configuration layer stays purely declarative.
3. How Snakemake invokes the pipeline (shell out to the CLI vs. import Python).
4. Where dataset-specific real-data code lives now that a CLI exists.
5. How the process reports failures (exit-code policy).

## Alternatives considered

- **Solver construction:** in `config.py` (return a ready `Solver`) vs. in the CLI
  composition root vs. a new standalone factory module.
- **Config layer:** Pydantic re-declaring defaults/bounds and building behavioural
  objects vs. a declarative validation boundary that only constructs the existing
  frozen *parameter* dataclasses.
- **Snakemake:** `run:` blocks importing `scdecon` directly vs. `shell:` calling
  the installed `scdecon` CLI.
- **Real-data workflow:** promote `scripts/melanoma_workflow.py` into a CLI
  command vs. leave it in `scripts/`.
- **Exit codes:** rely on Python's default (0/1) vs. a small documented policy.

## Decision

1. **`cli.py` is a thin composition root.** It only: loads/validates config,
   constructs the objects the library needs (including the concrete `Solver`),
   calls existing public functions, and maps outcomes to exit codes. It contains
   no numerical/biological logic.
2. **Solver construction happens in the CLI**, in a private `_build_solver`
   (single solver) / `_build_named_solvers` (benchmark set). Rationale below. No
   new module is introduced — a factory module would be an abstraction without a
   second consumer (YAGNI); the composition root is exactly where wiring belongs.
3. **`config.py` stays declarative.** It parses YAML, validates structure/types,
   and constructs only the frozen *parameter* dataclasses (`PreprocessConfig`,
   `SignatureConfig`, `SimulationConfig`) via pass-through of only the
   user-set fields. It never builds a `Solver`; solver *selection and parameters*
   are carried as plain validated data (`SolverSettings`, `BenchmarkSettings`).
4. **Snakemake calls the CLI via `shell:`**, never importing package modules in
   `run:` blocks.
5. **The real-data melanoma workflow stays in `scripts/`.** The CLI does not
   expose expression-space harmonisation (a dataset-specific heuristic, ADR-0010).
6. **A small, documented exit-code policy** is defined (table below), even though
   only a subset of failure modes exists today.

## Rationale

- **Why solver construction is in the CLI, not config.** A `Solver` is a
  *behavioural* object (it runs an algorithm); configuration should describe
  *what* to run, not *assemble how*. Keeping construction in the composition root
  means `config.py` has no dependency on the solver-core modules (and therefore
  never drags scikit-learn into config import), stays trivially serialisable and
  testable as pure data, and leaves one obvious place — the interface layer —
  where declarative intent becomes live objects. This mirrors how `deconvolve`
  already accepts an injected `Solver` rather than a name (ADR-0009).
- **Why configuration stays declarative.** The frozen dataclasses are the single
  source of truth for defaults and range validation (DD-004/DD-012). If Pydantic
  re-declared defaults or bounds they would drift; forwarding only user-set fields
  keeps one source of truth and preserves meaningful `null`s (e.g. `target_sum`).
- **Why Snakemake calls the CLI.** The CLI is the supported, validated entry
  point; shelling out keeps rules as pure orchestration (no business logic in the
  workflow), gives each rule a real process boundary (clean idempotent file
  targets), and means the workflow exercises exactly what users run.
- **Why dataset code stays out of the package.** ADR-0010 quarantined
  dataset-specific ingestion/heuristics in `scripts/`; the CLI is package code and
  the guardrail forbids `scdecon` importing `scripts/`. Exposing the melanoma
  heuristic through the CLI would breach that boundary or prematurely promote an
  unproven heuristic. Deferred until (if ever) it proves broadly reusable.

## Exit-code policy

| Code | Meaning |
|------|---------|
| 0 | Success. |
| 1 | Unexpected/internal error (unhandled exception; traceback shown). |
| 2 | CLI usage error (bad/missing option) — emitted by Typer/Click. |
| 3 | Configuration error (invalid YAML, unknown key, out-of-range value, or a required path missing from the config). |
| 4 | Input error (a referenced input file does not exist / cannot be read). |
| 5 | Computation error (a library `ValueError` during a scientific step). |

Implemented via two context managers (`_config_errors`, `_compute_errors`) that
translate the exceptions the library already raises; unexpected exceptions
propagate to Typer as exit code 1.

## Consequences

- `typer` and `pydantic`/`pyyaml` become direct runtime dependencies; a
  `scdecon` console entry point (`scdecon.cli:app`) is declared in
  `pyproject.toml`.
- The CLI is the composition root that may import across layers (io, plotting,
  preprocessing, signature, deconvolution, simulation); this does not weaken the
  core guardrails, which constrain the *lower* layers. The existing
  `test_src_does_not_import_scripts` scan already covers `cli.py`/`config.py`.
- Snakemake (M8, Session 3) will depend on the CLI contract and the exit-code
  policy; both are now fixed points.

## Future review conditions

- If a second consumer needs solver construction from a name (e.g. a Python API
  helper), promote `_build_solver` into a small public factory then — not before.
- If expression-space harmonisation is generalised and validated across datasets,
  revisit exposing a real-data command (supersede the relevant part of ADR-0010).
- Revisit the exit-code table if new, distinct failure classes appear (e.g. a
  dedicated code for partial/resumable pipeline failures).
