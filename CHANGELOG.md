# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is pre-1.0 (`0.x`), the public API may change between minor
versions; see `docs/project_memory/11_api_stability.md` for the stability policy.

Releases correspond to development milestones (M0 → v0.1.0, M1 → v0.2.0, …).

## [Unreleased]

## [1.0.0] - 2026-07-13 — First stable release (M9)

First stable release. From this version, the public Python API, the solver
interfaces, and the CLI contract follow the deprecation policy in
`docs/project_memory/11_api_stability.md`; the YAML run-configuration schema is
intentionally **not** frozen and may still evolve.

### Added
- **Dockerfile** (single-stage) + `.dockerignore` for reproducibly running the
  CLI and the Snakemake pipeline in a container; a Docker image smoke test in CI.
- **Documentation site** (MkDocs Material + mkdocstrings API reference),
  published to GitHub Pages via `.github/workflows/docs.yml`.
- **`CITATION.cff`** so the repository is citable.
- Documentation figures under `docs/assets/`, regenerated deterministically by
  `docs/generate_figures.py`.

### Changed
- **README overhaul**: hero, badges, quickstart (CLI / Snakemake / Docker),
  benchmark figures, and a citation section.
- **API freeze at 1.0.0**: the library API, the `Solver`/`MarkerSelector`
  extension interfaces, and the CLI command/exit-code contract are now Stable and
  covered by the deprecation policy. Documentation build tooling added as an
  optional `docs` extra.

## [0.9.0] - 2026-07-13 — CLI, configuration & Snakemake pipeline (M8)

### Added — package (`scdecon`)
- `scdecon.config` — a Pydantic run-configuration schema (`RunConfig` + typed
  sub-models) that parses/validates a YAML run file and constructs the existing
  frozen parameter dataclasses. It is a declarative validation boundary: the
  frozen dataclasses remain the single source of truth for defaults and range
  validation (only user-set fields are forwarded), and it builds no solver
  instances.
- `scdecon.cli` — a Typer command-line interface and composition root exposing
  `version`, `build-signature`, `simulate`, `deconvolve`, and `benchmark`. It
  loads config, constructs objects (including the solver), calls existing public
  library functions, and maps outcomes to a documented exit-code policy
  (0 success, 1 unexpected, 2 usage, 3 config, 4 input, 5 computation). No
  scientific logic lives in the CLI (ADR-0013).
- A `scdecon` console entry point (`scdecon.cli:app`) declared in
  `pyproject.toml`.

### Added — orchestration & tooling (not package API)
- `workflow/Snakefile` + `config/example_run.yaml` — a Snakemake pipeline that
  reproduces `build-signature → simulate → deconvolve → benchmark` by shelling
  out to the `scdecon` CLI (no rule imports Python modules). The run-config file
  is shared by the workflow and the CLI, so it stays a valid `RunConfig`.
- End-to-end integration tests (`tests/integration/`): the CLI pipeline (runs
  everywhere) and the Snakemake execution (skipped unless Snakemake is
  installed). A dedicated CI job installs the `pipeline` extra and runs both.

### Changed
- New runtime dependencies: `typer`, `pydantic`, `pyyaml`. New optional extra
  `pipeline` (`snakemake`) for running the workflow; dev extra gains
  `types-PyYAML`.

## [0.8.1] - 2026-07-11 — Release readiness (M7.5)

### Fixed
- `scdecon.__version__` now reports the release version instead of the
  placeholder `0.0.0`; the distribution metadata matches on (re)install.

### Added
- `py.typed` marker (PEP 561) + packaging config, so `scdecon`'s type hints are
  exported to downstream consumers.
- Coverage measurement via `pytest-cov` (built into `pytest`); coverage is 96%
  (package + scripts). Two targeted tests for previously untested public error
  contracts (`align_proportions` duplicate sample labels; `read_metadata` empty).

## [0.8.0] - 2026-07-10 — Real tumour data & expression-space harmonisation (M7)

### Added — package (`scdecon`)
- `scdecon.genes` — generic, dataset-agnostic gene-identifier harmonisation:
  `strip_ensembl_version`, `relabel_gene_index` (relabel a gene-indexed matrix
  through an id→id mapping, collapsing collisions), and `GeneAggregation`
  (`SUM`/`MEAN`).
- `scdecon.genes.GeneMappingCoverage` + `compute_mapping_coverage` — a
  first-class mapping-coverage QC metric; `relabel_gene_index` warns below a
  configurable `min_coverage` and rejects non-numeric columns.

### Added — scripts (`scripts/`, dataset-specific; **not** package API)
- `scripts/datasets/` — dataset ingestion: recount3 `gene_sums` loader, GENCODE
  GTF → gene-symbol map, and the Tirosh GSE72056 scRNA loader (cell-type decoding
  + `log2(TPM/10+1)` → linear-TPM reconstruction so the existing signature
  pipeline applies unchanged).
- `scripts/download_data.py` — idempotent downloader: SHA-256 checksums, atomic
  writes, `Content-Length` verification, and a JSON provenance manifest (standard
  library only; data is never committed).
- `scripts/melanoma_workflow.py` — the end-to-end TCGA-SKCM workflow, including
  `harmonize_expression_space` (configurable per-gene scaling `mean`/`max`/`l2`)
  and a before/after QC comparison.

### Changed
- Architecture guardrails extended: `scdecon.genes` must stay dataset-agnostic
  (no io/plotting/anndata/scanpy), and the `scdecon` package must not import the
  `scripts/` package (dependency flows scripts → scdecon only).

### Documentation
- `docs/tutorials/melanoma-tme.md` — real-data walkthrough documenting the
  baseline failure, the finding that **length normalisation (as implemented) did
  not resolve it**, the expression-space harmonisation (with its `D⁻¹` /
  weighted-NNLS justification), and the T+NK-vs-cytotoxicity **sanity check**
  (Spearman 0.08 → 0.74; partially circular — see the tutorial caveats). Separates
  demonstrated results from interpretation throughout.
- ADR-0010 — real-data ingestion: generic harmonisation in `src/`, dataset
  specifics in `scripts/`.
- Expanded known limitations: relative (not absolute) fractions, no TCGA ground
  truth, Smart-seq2 vs recount3 platform differences, NK/CD8-T
  non-identifiability, and harmonisation being evidence-supported (not proven
  uniquely optimal).

## [0.7.0] - 2026-07-09 — Additional solvers & benchmarking (M6)

### Added
- `NuSVRSolver` (+ `NuSVRConfig`) — CIBERSORT-style linear ν-support-vector
  regression (scikit-learn), coefficients clipped to ≥0 and renormalised.
- `RobustSolver` (+ `RobustConfig`, `RobustLoss`) — robust non-negative
  regression via `scipy.optimize.least_squares` (`soft_l1` default, `huber`
  optional). Both implement the existing `Solver` contract.
- `scikit-learn` added as a direct dependency (isolated to the ν-SVR solver).
- `run_benchmark` + `BenchmarkResult` — a solver-agnostic, fair-by-construction
  benchmark harness: the signature/bulk are aligned once and every supplied
  solver is scored over the identical inputs with the same metrics. Depends only
  on the `Solver` interface (enforced by a guardrail); solver names are preserved
  exactly; runtimes are informational only.
- `scdecon.plotting.plot_benchmark` — per-solver metric bar chart.

### Changed
- Architecture guardrails extended: scikit-learn is imported only by
  `deconvolution/nusvr.py`, and `deconvolution/benchmark.py` imports no concrete
  solver. `plotting` now also depends on `deconvolution` (for `BenchmarkResult`).

## [0.6.0] - 2026-07-09 — Pseudobulk simulation & validation framework (M5)

### Added
- `scdecon.simulation` package (M5):
  - `SimulationConfig` (frozen, validated) and `ProportionPrior` (StrEnum).
  - `split_reference` — a deterministic, cell-type-stratified split into disjoint
    signature / held-out partitions, preventing leakage between signature
    construction and pseudobulk simulation.
  - `BaseSimulator` / `CellSumSimulator` / `simulate_pseudobulk` — generate
    pseudobulk by summing the raw counts of single cells drawn (with replacement)
    at multinomial-sampled proportions; the **realised** proportions are recorded
    as ground truth.
  - `PseudobulkDataset` — `bulk` (genes × samples) and `proportions`
    (cell types × samples, matching `deconvolve`'s orientation).
- `scdecon.validation` package (M5):
  - `align_proportions` — strict alignment of truth/prediction proportion frames
    (identical cell-type and sample labels).
  - `evaluate` → `ValidationReport` — overall + per-cell-type RMSE, Pearson, and
    Spearman across samples; a lightweight, stable result object.
- `scdecon.plotting.plot_truth_vs_prediction` — truth-vs-prediction scatter grid.
- End-to-end recovery test proving the full pipeline (split → signature →
  simulate → deconvolve → evaluate) recovers known composition on synthetic data.

### Changed
- Architecture guardrail extended: `scdecon.simulation` and `scdecon.validation`
  must not import `scdecon.io` or the plotting stack. `plotting` now depends on
  `validation` (for shared truth/prediction alignment).

### Documentation
- Added this `CHANGELOG.md` (Keep a Changelog format).
- Added `docs/project_memory/11_api_stability.md` recording API stability tiers,
  deprecation policy, and the semantic-versioning strategy.
- Updating the changelog and the API-stability document is now part of the
  Definition of Done for every release.

## [0.5.0] - 2026-07-09 — First end-to-end deconvolution workflow (M4)

### Added
- `scdecon.deconvolution` package:
  - `Solver` — abstract, format-agnostic solver interface (NumPy arrays only);
    `fit(signature, bulk) -> p` with `p >= 0` and `sum(p) == 1`.
  - `NNLSSolver` — non-negative least squares (`scipy.optimize.nnls`) minimising
    `‖S·x − b‖₂` subject to `x >= 0`, renormalised to `sum(p) == 1`; raises
    `ValueError` on a degenerate all-zero solution.
  - `align_signature_and_bulk` / `AlignedInputs` — a pandas-aware adapter that
    restricts a signature and bulk matrix to their shared genes (in signature row
    order) with a configurable low-overlap warning (`min_overlap`, default 0.5).
  - `deconvolve` — orchestrator returning per-sample proportions as a
    `cell types × samples` DataFrame.
- `scipy` added as a direct runtime dependency.
- Permanent solver regression tests: exact recovery (canonical), scale
  invariance, non-negativity, sum-to-one, determinism, small-noise recovery,
  degenerate all-zero → error, and input non-mutation.

### Changed
- Architecture guardrail (`tests/unit/test_architecture.py`) extended: the
  deconvolution layer must not import io/plotting/anndata/scanpy, and the
  solver-core modules must be NumPy/SciPy-only.

### Documentation
- Established the self-describing project-memory system
  (`docs/project_memory/00`–`10`), Architecture Decision Records
  (`docs/architecture/ADR-0001`–`ADR-0009`), and `CLAUDE.md`. Documentation
  became part of the Definition of Done.

## [0.4.0] - 2026-07-09 — Signature matrix generation (M3)

### Added
- `scdecon.signature` package:
  - `RankMethod` (StrEnum) and `SignatureConfig` (frozen, validated).
  - `MarkerSelector` strategy interface with `RankGenesGroupsSelector` (Scanpy
    `rank_genes_groups`, one-vs-rest, plus a cross-type specificity filter),
    `MarkerSet`, and `select_markers`.
  - `build_signature` — assembles the signature matrix as a validated
    `DataFrame` (genes × sorted cell types), using arithmetic-mean **linear-scale**
    profiles; row order is exactly `MarkerSet.genes()`.
- `scdecon.plotting.plot_signature_heatmap` — headless-safe signature heatmap.
- `matplotlib` and `seaborn` added as direct runtime dependencies.

### Changed
- Architecture guardrail extended so the computational core (io, preprocessing,
  signature) imports neither `scdecon.plotting` nor matplotlib/seaborn.

## [0.3.0] - 2026-07-09 — Single-cell preprocessing pipeline (M2)

### Added
- `scdecon.logging_utils` — `configure_logging`, `get_logger` (structured,
  idempotent, namespaced logging).
- `scdecon.preprocessing` package:
  - `PreprocessConfig` (frozen, validated) — the single source of truth for
    QC/normalisation parameters (no magic numbers).
  - `compute_qc_metrics` (in-place metadata), `filter_cells_and_genes`
    (non-destructive) + `QCSummary`, `normalize` (raw counts preserved in
    `layers["counts"]`; refuses to double-normalise), and the `preprocess`
    orchestrator (QC summary stored at `.uns["scdecon_qc_summary"]`).
- `scanpy` added as a direct runtime dependency.
- First static architecture guardrail test.

## [0.2.0] - 2026-07-09 — Data I/O layer (M1)

### Added
- `scdecon.io` package: readers (`read_h5ad`, `read_bulk`, `read_metadata`) and
  writers (`write_table`, `write_h5ad`). Paths accept `str | Path` and normalise
  to `pathlib.Path`; bulk matrices use a fixed genes × samples orientation;
  structural-only validation with actionable errors; data preserved exactly.
- `anndata`, `numpy`, `pandas` runtime dependencies; `pandas-stubs` dev
  dependency; small committed text fixtures.

### Fixed
- Removed the pinned mypy `python_version = "3.11"`, which caused CI failures on
  Python 3.12/3.13 when mypy parsed NumPy's PEP 695 (`type` statement) stubs under
  a 3.11 target. mypy now infers the target from the interpreter; the 3.11 CI job
  still enforces the language floor. (See ADR-0002.)

## [0.1.0] - 2026-07-08 — Project scaffold & development infrastructure (M0)

### Added
- Installable `src/`-layout Python package (`scdecon`) built with setuptools,
  with a dynamic version.
- Tooling: ruff (lint + format), mypy (strict), pytest — all configured in
  `pyproject.toml`; pre-commit hooks.
- GitHub Actions CI (lint → format → type-check → test) across Python
  3.11/3.12/3.13.
- MIT license, README, smoke test, conda `environment.yml`
  (conda-forge + bioconda), `.gitattributes`, `.editorconfig`.

[Unreleased]: https://github.com/psanyalaich/scdecon/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/psanyalaich/scdecon/releases/tag/v1.0.0
[0.9.0]: https://github.com/psanyalaich/scdecon/releases/tag/v0.9.0
[0.8.1]: https://github.com/psanyalaich/scdecon/releases/tag/v0.8.1
[0.8.0]: https://github.com/psanyalaich/scdecon/releases/tag/v0.8.0
[0.7.0]: https://github.com/psanyalaich/scdecon/releases/tag/v0.7.0
[0.6.0]: https://github.com/psanyalaich/scdecon/releases/tag/v0.6.0
[0.5.0]: https://github.com/psanyalaich/scdecon/releases/tag/v0.5.0
[0.4.0]: https://github.com/psanyalaich/scdecon/releases/tag/v0.4.0
[0.3.0]: https://github.com/psanyalaich/scdecon/releases/tag/v0.3.0
[0.2.0]: https://github.com/psanyalaich/scdecon/releases/tag/v0.2.0
[0.1.0]: https://github.com/psanyalaich/scdecon/releases/tag/v0.1.0
