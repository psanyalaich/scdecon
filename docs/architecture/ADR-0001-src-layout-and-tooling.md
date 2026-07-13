# ADR-0001 — src-layout package, setuptools, and tooling

- **Status:** Accepted
- **Milestone:** M0
- **Date:** 2026 (M0)

## Context

The repository needed a clean, installable, CI-green Python skeleton before any
scientific code, suitable for a long-term, portfolio-quality research package.

## Problem

Choose a project layout, build backend, and baseline tooling that are
conventional, low-friction, and enforce quality from the first commit.

## Alternatives considered

- **Layout:** flat layout vs `src/` layout.
- **Build backend:** setuptools vs hatchling vs poetry.
- **Entry point:** declare the `scdecon` console script now vs later.

## Decision

- Use a **`src/` layout** package.
- Use the **setuptools** build backend with a dynamic version read from
  `scdecon.__version__`.
- Co-locate ruff, mypy (strict), and pytest configuration in `pyproject.toml`.
- Target **Python 3.11+**; CI matrix 3.11/3.12/3.13.
- **Do not** declare the `scdecon` console entry point yet (deferred to M8 when
  `cli.py` exists).
- Zero runtime dependencies at M0.

## Rationale

- src-layout prevents importing the un-installed package by accident and forces
  testing the installed artifact.
- setuptools is ubiquitous and low-friction; no compelling reason to adopt
  hatchling/poetry (maintainer preference).
- Declaring an entry point pointing at a non-existent `cli.py` would break the
  install.

## Consequences

- `pip install -e ".[dev]"`, `pytest`, `ruff`, and `mypy` all pass from M0.
- The console command arrives in M8 with the CLI.

## Future review conditions

- Revisit the build backend if packaging needs (PyPI wheels, plugin entry
  points, bioconda) outgrow setuptools.
