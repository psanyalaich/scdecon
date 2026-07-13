# ADR-0002 — Do not pin mypy `python_version`

- **Status:** Accepted
- **Milestone:** M1
- **Date:** 2026 (M1)

## Context

CI ran mypy (strict) across a Python 3.11/3.12/3.13 matrix. After adding NumPy as
a dependency, CI failed **only on 3.12 and 3.13** with:

```
numpy/__init__.pyi: error: Type statement is only supported in Python 3.12 and
greater [syntax]
```

The 3.11 job and local dev (Python 3.14, but with an older, stale NumPy) passed.

## Problem

`[tool.mypy] python_version = "3.11"` forces mypy to parse **all** files —
including third-party stubs — under 3.11 grammar. NumPy ≥ 2.4 requires Python
≥ 3.12 and uses PEP 695 `type` statements in its stubs. On the 3.12/3.13 CI jobs,
pip resolves NumPy 2.5.x (with that syntax); under a 3.11 mypy target the stub is
a syntax error. The 3.11 job resolves an older NumPy without the syntax, so it
passed — explaining the version-split failure.

## Alternatives considered

1. **Upgrade mypy while keeping the pin** — does **not** help. Verified upstream
   that applying the version syntax gate to stubs is intentional: mypy issue
   #18701 was closed *not planned*; the relaxation request #21178 is open with no
   release; `per-file-target-version` (#19892) is an unresolved feature request.
2. **Bump the pin to `python_version = "3.12"`** — would stop enforcing the 3.11
   floor and let genuinely 3.11-incompatible syntax pass mypy, breaking real
   3.11 users.
3. **Ignore NumPy in mypy** (`follow_imports=skip`) — discards NumPy's valuable
   stubs, which the numeric core (M4+) relies on.
4. **Pin `numpy<2.4`** — fights the ecosystem and causes dependency drift.
5. **Remove the pin** (chosen).

## Decision

Remove `python_version` from `[tool.mypy]`. mypy then infers the target from the
interpreter it runs on; each CI matrix job type-checks under its own version, and
the **3.11 CI job continues to enforce the 3.11 floor** on our own code.

## Rationale

Smallest change; robust to future NumPy/stub evolution; better coverage (each
version checked under itself) while still guarding the declared minimum.

## Consequences

- CI is green across 3.11–3.13.
- A developer editing locally on a newer interpreter won't get an immediate 3.11
  warning for a 3.12-only syntax slip, but the 3.11 CI job catches it before
  merge.
- Third-party libs whose source can't be checked under our target are made opaque
  per-module via `[[tool.mypy.overrides]]` (anndata, scanpy, seaborn).

## Future review conditions

- Revisit if mypy ships a supported way to parse newer stub syntax under an older
  target (e.g. `per-file-target-version`), or if we want to enforce the floor on
  every matrix job (via `mypy --python-version ${{ matrix.python-version }}` in
  CI — an enhancement, not required).
