# CLI reference

The `scdecon` command-line tool is a thin **composition root**: it loads and
validates a YAML run configuration, constructs the objects the library needs
(including the solver), calls the package's public functions, and maps outcomes
to predictable exit codes. It contains no scientific logic — that all lives in
the package.

All commands except `version` take `--config/-c` pointing at a YAML run file, and
accept `--verbose/-v` for debug-level logging.

```bash
scdecon --help
```

## Commands

### `version`

```bash
scdecon version
```

Print the installed scdecon version and exit.

### `build-signature`

```bash
scdecon build-signature --config run.yaml
```

Reads `paths.reference` (`.h5ad`), preprocesses (QC + normalisation), selects
marker genes, builds the signature matrix, and writes it to `paths.signature`
(optionally a heatmap to `paths.heatmap`).

### `simulate`

```bash
scdecon simulate --config run.yaml
```

Reads `paths.reference`, simulates pseudobulk samples with known proportions, and
writes the bulk matrix to `paths.bulk` and the ground-truth proportions to
`paths.truth`.

### `deconvolve`

```bash
scdecon deconvolve --config run.yaml
```

Reads `paths.signature` and `paths.bulk` (TSV), runs the solver selected under
`solver.name` (`nnls`, `nusvr`, or `robust`), and writes the estimated
proportions (cell types × samples, each column summing to 1) to
`paths.proportions`.

### `benchmark`

```bash
scdecon benchmark --config run.yaml
```

Reads `paths.signature`, `paths.bulk` and `paths.truth`, runs the solver set
under `benchmark.solvers`, and writes per-solver metrics to `paths.metrics`
(optionally a bar chart to `paths.benchmark_plot`).

## Exit codes

The CLI uses a predictable exit-code policy so it composes cleanly in scripts and
the Snakemake workflow:

| Code | Meaning |
|------|---------|
| `0` | Success. |
| `1` | Unexpected/internal error (an unhandled exception; traceback shown). |
| `2` | CLI usage error (bad/missing option) — emitted by Typer/Click. |
| `3` | Configuration error (invalid YAML, unknown key, out-of-range value, or a required path missing from the config). |
| `4` | Input error (a referenced input file does not exist / cannot be read). |
| `5` | Computation error (a library error during a scientific step, e.g. no shared genes or a degenerate solver result). |

The command names, options, and this exit-code policy are part of the stable
public interface as of v1.0.0.
