# Architecture Decision Records (ADRs)

This directory holds the project's Architecture Decision Records. Each ADR
captures a significant decision with its **Context, Problem, Alternatives
considered, Decision, Rationale, Consequences, and Future review conditions**.

ADRs are append-only history: when a decision changes, add a **new** ADR that
supersedes the old one (note the supersession) rather than editing the old one.

Each ADR is self-contained (Context, Problem, Alternatives, Decision, Rationale,
Consequences). For the user-visible release history see `CHANGELOG.md`.

## Index

| ADR | Title | Milestone | Status |
|-----|-------|-----------|--------|
| [ADR-0001](ADR-0001-src-layout-and-tooling.md) | src-layout package, setuptools, and tooling | M0 | Accepted |
| [ADR-0002](ADR-0002-mypy-python-version.md) | Do not pin mypy `python_version` | M1 | Accepted |
| [ADR-0003](ADR-0003-io-layer-contract.md) | I/O layer: faithful readers/writers, fixed orientation | M1 | Accepted |
| [ADR-0004](ADR-0004-preprocessing-contracts.md) | Preprocessing: config-driven, explicit mutation contracts | M2 | Accepted |
| [ADR-0005](ADR-0005-markerselector-and-rankmethod.md) | Marker selection behind an interface; typed method enum | M3 | Accepted |
| [ADR-0006](ADR-0006-signature-as-dataframe.md) | Signature matrix as a validated DataFrame | M3 | Accepted |
| [ADR-0007](ADR-0007-marker-strategy.md) | Marker strategy: Wilcoxon + specificity, fixed-N, linear mean | M3 | Accepted |
| [ADR-0008](ADR-0008-plotting-separation.md) | Plotting isolated; core independent of matplotlib/seaborn | M3 | Accepted |
| [ADR-0009](ADR-0009-deconvolution-solver-and-alignment.md) | Format-agnostic solver interface + separate alignment | M4 | Accepted |
| [ADR-0010](ADR-0010-real-data-ingestion-and-gene-harmonisation.md) | Real-data ingestion out-of-package; generic gene-ID harmonisation in-package | M7 | Accepted |
| [ADR-0013](ADR-0013-cli-composition-root.md) | CLI as a thin composition root; declarative config; Snakemake calls the CLI | M8 | Accepted |
