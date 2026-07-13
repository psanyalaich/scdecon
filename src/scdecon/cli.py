"""Typer command-line interface for scdecon (the composition root).

This module is an **interface**, not the application. Its only responsibilities
are to load and validate configuration, construct the objects the library needs
(including the concrete :class:`~scdecon.deconvolution.base.Solver`), call the
existing public library functions, and translate outcomes into predictable exit
codes. No scientific or computational logic lives here -- every numerical step is
a call into the already-tested package (see ADR-0013).

Commands (all except ``version`` take ``--config`` pointing at a YAML run file
validated by :class:`scdecon.config.RunConfig`):

- ``version``          -- print the installed version.
- ``build-signature``  -- reference ``.h5ad`` -> signature matrix TSV.
- ``simulate``         -- reference ``.h5ad`` -> pseudobulk + ground-truth TSVs.
- ``deconvolve``       -- signature + bulk TSV -> estimated proportions TSV.
- ``benchmark``        -- signature + bulk + truth TSV -> per-solver metrics TSV.

Exit-code policy (see ADR-0013 for the rationale):

===== ==================================================================
 Code  Meaning
===== ==================================================================
 0     Success.
 1     Unexpected/internal error (an unhandled exception; traceback shown).
 2     CLI usage error (bad/missing option) -- emitted by Typer/Click.
 3     Configuration error (invalid YAML, unknown key, bad/out-of-range
       value, or a required path missing from the config).
 4     Input error (a referenced input file does not exist / cannot be read).
 5     Computation error (a library ``ValueError`` during a scientific step,
       e.g. no shared genes or a degenerate solver result).
===== ==================================================================
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from enum import IntEnum
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from pydantic import ValidationError

from scdecon import __version__
from scdecon.config import RunConfig, SolverName, SolverSettings
from scdecon.deconvolution import (
    NNLSSolver,
    NuSVRConfig,
    NuSVRSolver,
    RobustConfig,
    RobustSolver,
    Solver,
    deconvolve,
    run_benchmark,
)
from scdecon.io import read_bulk, read_h5ad, write_table
from scdecon.logging_utils import configure_logging, get_logger
from scdecon.plotting import plot_benchmark, plot_signature_heatmap
from scdecon.preprocessing import preprocess
from scdecon.signature import build_signature, select_markers
from scdecon.simulation import simulate_pseudobulk

logger = get_logger("cli")


class ExitCode(IntEnum):
    """Predictable process exit codes (see the module-level policy table)."""

    SUCCESS = 0
    UNEXPECTED = 1
    USAGE = 2
    CONFIG_ERROR = 3
    INPUT_ERROR = 4
    RUNTIME_ERROR = 5


app = typer.Typer(
    name="scdecon",
    help="Single-cell-reference deconvolution of bulk tumour transcriptomes.",
    add_completion=False,
    no_args_is_help=True,
)

ConfigOption = Annotated[
    Path,
    typer.Option("--config", "-c", help="Path to the YAML run configuration."),
]
VerboseOption = Annotated[
    bool,
    typer.Option("--verbose", "-v", help="Enable debug-level logging."),
]


# --------------------------------------------------------------------------- #
# Composition-root helpers (plumbing only -- no science)
# --------------------------------------------------------------------------- #
def _configure_logging(verbose: bool) -> None:
    """Set up logging at the requested verbosity."""
    configure_logging(logging.DEBUG if verbose else logging.INFO)


def _fail(code: ExitCode, message: str) -> NoReturn:
    """Log an error and exit with the given code."""
    logger.error(message)
    raise typer.Exit(int(code))


@contextmanager
def _config_errors() -> Iterator[None]:
    """Map configuration-phase failures to :attr:`ExitCode.CONFIG_ERROR`.

    Covers Pydantic validation, a missing config file, out-of-range values that
    the frozen dataclasses reject, and required paths absent from the config.
    """
    try:
        yield
    except ValidationError as exc:
        _fail(ExitCode.CONFIG_ERROR, f"invalid configuration:\n{exc}")
    except (ValueError, FileNotFoundError) as exc:
        _fail(ExitCode.CONFIG_ERROR, f"configuration error: {exc}")


@contextmanager
def _compute_errors() -> Iterator[None]:
    """Map computation-phase failures to input/runtime exit codes.

    A missing data file is an input error; a library ``ValueError`` during a
    scientific step is a computation error. Any other exception propagates and
    is reported by Typer as an unexpected error (exit code 1).
    """
    try:
        yield
    except FileNotFoundError as exc:
        _fail(ExitCode.INPUT_ERROR, f"input file not found: {exc}")
    except ValueError as exc:
        _fail(ExitCode.RUNTIME_ERROR, f"computation failed: {exc}")


def _require_path(value: Path | None, name: str) -> Path:
    """Return ``value`` or raise a config error naming the missing path.

    Raised inside a :func:`_config_errors` block, so a missing required path is
    reported as a configuration error rather than an unexpected failure.
    """
    if value is None:
        raise ValueError(
            f"required path 'paths.{name}' is not set in the configuration"
        )
    return value


def _build_solver(settings: SolverSettings) -> Solver:
    """Construct the configured solver.

    Solver construction lives in the CLI composition root (not in the declarative
    ``config`` layer, see ADR-0013): configuration selects a solver by name and
    supplies parameters; here we turn that selection into a concrete
    :class:`~scdecon.deconvolution.base.Solver`. Parameters irrelevant to the
    chosen solver are ignored.
    """
    if settings.name == "nnls":
        return NNLSSolver()
    if settings.name == "nusvr":
        nusvr_overrides = {} if settings.nu is None else {"nu": settings.nu}
        return NuSVRSolver(NuSVRConfig(**nusvr_overrides))
    if settings.name == "robust":
        robust_overrides: dict[str, Any] = {}
        if settings.loss is not None:
            robust_overrides["loss"] = settings.loss
        if settings.f_scale is not None:
            robust_overrides["f_scale"] = settings.f_scale
        return RobustSolver(RobustConfig(**robust_overrides))
    raise ValueError(f"unknown solver: {settings.name!r}")  # unreachable (Literal)


def _build_named_solvers(names: list[SolverName]) -> dict[str, Solver]:
    """Build a name -> solver mapping (default parameters) for benchmarking."""
    return {name: _build_solver(SolverSettings(name=name)) for name in names}


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
@app.command()
def version() -> None:
    """Print the installed scdecon version and exit."""
    typer.echo(__version__)


@app.command("build-signature")
def build_signature_command(
    config: ConfigOption, verbose: VerboseOption = False
) -> None:
    """Build a signature matrix from an annotated single-cell reference.

    Reads ``paths.reference`` (.h5ad), preprocesses, selects markers, builds the
    signature, and writes it to ``paths.signature`` (optionally a heatmap to
    ``paths.heatmap``).
    """
    _configure_logging(verbose)
    with _config_errors():
        cfg = RunConfig.load(config)
        preprocess_config = cfg.to_preprocess_config()
        signature_config = cfg.to_signature_config()
        reference = _require_path(cfg.paths.reference, "reference")
        signature_out = _require_path(cfg.paths.signature, "signature")
    with _compute_errors():
        adata = read_h5ad(reference)
        adata = preprocess(adata, preprocess_config)
        markers = select_markers(adata, signature_config)
        signature = build_signature(adata, markers, signature_config)
        write_table(signature, signature_out)
        if cfg.paths.heatmap is not None:
            plot_signature_heatmap(signature, cfg.paths.heatmap)
    logger.info("build-signature: wrote %s", signature_out)


@app.command()
def simulate(config: ConfigOption, verbose: VerboseOption = False) -> None:
    """Simulate pseudobulk samples with known proportions from a reference.

    Reads ``paths.reference`` (.h5ad), simulates pseudobulk, and writes the bulk
    matrix to ``paths.bulk`` and the ground-truth proportions to ``paths.truth``.
    """
    _configure_logging(verbose)
    with _config_errors():
        cfg = RunConfig.load(config)
        simulation_config = cfg.to_simulation_config()
        reference = _require_path(cfg.paths.reference, "reference")
        bulk_out = _require_path(cfg.paths.bulk, "bulk")
        truth_out = _require_path(cfg.paths.truth, "truth")
    with _compute_errors():
        adata = read_h5ad(reference)
        dataset = simulate_pseudobulk(adata, simulation_config)
        write_table(dataset.bulk, bulk_out)
        write_table(dataset.proportions, truth_out)
    logger.info("simulate: wrote %s and %s", bulk_out, truth_out)


@app.command("deconvolve")
def deconvolve_command(config: ConfigOption, verbose: VerboseOption = False) -> None:
    """Estimate cell-type proportions for a bulk matrix.

    Reads ``paths.signature`` and ``paths.bulk`` (TSV), runs the configured
    solver, and writes the proportions to ``paths.proportions``.
    """
    _configure_logging(verbose)
    with _config_errors():
        cfg = RunConfig.load(config)
        solver = _build_solver(cfg.solver)
        signature_in = _require_path(cfg.paths.signature, "signature")
        bulk_in = _require_path(cfg.paths.bulk, "bulk")
        proportions_out = _require_path(cfg.paths.proportions, "proportions")
    with _compute_errors():
        signature = read_bulk(signature_in)
        bulk = read_bulk(bulk_in)
        proportions = deconvolve(signature, bulk, solver=solver)
        write_table(proportions, proportions_out)
    logger.info("deconvolve: wrote %s", proportions_out)


@app.command()
def benchmark(config: ConfigOption, verbose: VerboseOption = False) -> None:
    """Compare solvers on simulated bulk with known ground truth.

    Reads ``paths.signature``, ``paths.bulk`` and ``paths.truth`` (TSV), runs the
    configured solver set, and writes per-solver metrics to ``paths.metrics``
    (optionally a bar chart to ``paths.benchmark_plot``).
    """
    _configure_logging(verbose)
    with _config_errors():
        cfg = RunConfig.load(config)
        solvers = _build_named_solvers(cfg.benchmark.solvers)
        signature_in = _require_path(cfg.paths.signature, "signature")
        bulk_in = _require_path(cfg.paths.bulk, "bulk")
        truth_in = _require_path(cfg.paths.truth, "truth")
        metrics_out = _require_path(cfg.paths.metrics, "metrics")
    with _compute_errors():
        signature = read_bulk(signature_in)
        bulk = read_bulk(bulk_in)
        truth = read_bulk(truth_in)
        result = run_benchmark(signature, bulk, truth, solvers)
        write_table(result.to_frame(), metrics_out)
        if cfg.paths.benchmark_plot is not None:
            if cfg.benchmark.metric is not None:
                plot_benchmark(
                    result, cfg.paths.benchmark_plot, metric=cfg.benchmark.metric
                )
            else:
                plot_benchmark(result, cfg.paths.benchmark_plot)
    logger.info("benchmark: wrote %s", metrics_out)


if __name__ == "__main__":  # pragma: no cover
    app()
