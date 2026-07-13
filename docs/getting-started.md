# Getting started

## Installation

scdecon requires **Python 3.11+**.

```bash
git clone https://github.com/psanyalaich/scdecon
cd scdecon
pip install -e .
```

Optional extras:

```bash
pip install -e ".[dev]"       # linting, typing, tests
pip install -e ".[pipeline]"  # Snakemake, to run the workflow
pip install -e ".[docs]"      # build this documentation site
```

Installing the package provides the `scdecon` command-line tool.

!!! tip "If `scdecon` isn't on your PATH"
    The console script is installed into your environment's scripts directory,
    which may not be on `PATH`. The equivalent, PATH-independent invocation is
    `python -m scdecon.cli` — e.g. `python -m scdecon.cli version`.

## Configure a run

Every command reads a single YAML **run configuration**. Paths that a command
needs but that are missing are reported as configuration errors. A minimal
example (`run.yaml`):

```yaml
paths:
  reference: data/reference.h5ad          # your annotated single-cell .h5ad
  signature: results/signature.tsv
  bulk: results/pseudobulk.tsv
  truth: results/truth_proportions.tsv
  proportions: results/estimated_proportions.tsv
  metrics: results/benchmark_metrics.tsv

markers:
  n_markers_per_type: 25
  method: wilcoxon

simulation:
  n_samples: 50
  n_cells_per_sample: 500
  cell_type_key: cell_type

solver:
  name: nnls

benchmark:
  solvers: [nnls, nusvr, robust]
```

Fields left unset inherit the library defaults. See
[`config/example_run.yaml`](https://github.com/psanyalaich/scdecon/blob/main/config/example_run.yaml)
for the full annotated example.

## Run the pipeline from the CLI

```bash
scdecon build-signature --config run.yaml   # reference -> signature matrix
scdecon simulate        --config run.yaml   # reference -> pseudobulk + truth
scdecon deconvolve      --config run.yaml   # signature + bulk -> proportions
scdecon benchmark       --config run.yaml   # compare solvers vs. truth
```

Each command exits `0` on success and uses a documented non-zero
[exit-code policy](cli.md#exit-codes) on failure.

## Run the whole pipeline with Snakemake

With the `pipeline` extra installed, the Snakemake workflow reproduces the same
steps by calling the CLI (run from the repository root):

```bash
snakemake --cores 1                                  # uses config/example_run.yaml
snakemake --cores 1 --config run_config=run.yaml     # your own configuration
snakemake -n                                         # dry run: show the plan
```

The workflow is an automated user of the CLI — it shells out to `scdecon` and
never imports the package directly, so the library, CLI, and pipeline stay
cleanly separated.

## Next steps

- The **[CLI reference](cli.md)** documents every command and option.
- The **[API reference](api/index.md)** documents the Python package for use as a
  library.
- The **[melanoma tutorial](tutorials/melanoma-tme.md)** walks through a real
  TCGA-SKCM analysis end to end.
