"""Declarative run-configuration schema (Pydantic) for the scdecon CLI.

This module is the **validation boundary** between a YAML run-configuration file
and the frozen parameter dataclasses that already govern the science
(:class:`~scdecon.preprocessing.params.PreprocessConfig`,
:class:`~scdecon.signature.params.SignatureConfig`,
:class:`~scdecon.simulation.params.SimulationConfig`). Its only jobs are to
**parse** YAML and **validate** its structure/types, then hand back the existing
frozen configs unchanged.

Design constraints (mirroring the project blueprint):

- **The frozen dataclasses stay the single source of truth for defaults and
  semantic (range) validation.** The Pydantic models never re-declare a default
  or a numeric bound. Every field is optional; only the fields a user actually
  set are forwarded (via :data:`BaseModel.model_fields_set`), so omitted fields
  fall through to the dataclass default and an explicit ``null`` (e.g.
  ``target_sum: null``) is preserved as ``None`` rather than silently reset.
- **Configuration is declarative.** This module performs no I/O beyond reading
  the YAML file and constructs no behavioural objects. In particular it does not
  build :class:`~scdecon.deconvolution.base.Solver` instances -- solver *selection
  and parameters* are validated here as plain data and assembled by the CLI
  composition root.
- **Light imports.** Only the stdlib-backed ``.params`` submodules are imported,
  so importing :mod:`scdecon.config` does not pull in scanpy/scikit-learn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from scdecon.deconvolution.params import RobustLoss
from scdecon.preprocessing.params import PreprocessConfig
from scdecon.signature.params import RankMethod, SignatureConfig
from scdecon.simulation.params import ProportionPrior, SimulationConfig

#: The solver names the CLI understands. Kept as a single ``Literal`` alias so
#: both single-solver (``deconvolve``) and multi-solver (``benchmark``) selection
#: validate against the same set without introducing a new enum type.
SolverName = Literal["nnls", "nusvr", "robust"]


def _default_benchmark_solvers() -> list[SolverName]:
    """Default solver set for ``benchmark`` (typed so it satisfies the field)."""
    return ["nnls"]


def _set_fields(model: BaseModel) -> dict[str, Any]:
    """Return ``{field: value}`` for only the fields the user explicitly set.

    Using :attr:`~pydantic.BaseModel.model_fields_set` (rather than a ``None``
    sentinel) preserves an explicitly provided ``null`` -- e.g. ``target_sum:
    null`` meaning "median normalisation" -- as ``None``, while letting omitted
    fields fall through to the frozen dataclass default. This is what keeps the
    dataclasses the single source of truth for defaults.

    Parameters
    ----------
    model:
        A settings sub-model whose set fields should be forwarded.

    Returns
    -------
    dict[str, Any]
        The explicitly set field values, suitable for ``**``-expansion into the
        corresponding frozen dataclass constructor.
    """
    return {name: getattr(model, name) for name in model.model_fields_set}


class PathsConfig(BaseModel):
    """Input/output file locations referenced by the CLI commands.

    Every path is optional here: this schema does not know which command will
    run, so it validates only that provided values are well-formed paths.
    Whether a given path is *required* (and whether an input file must exist) is
    the CLI command's concern -- missing inputs fail loudly in the I/O layer.
    """

    model_config = ConfigDict(extra="forbid")

    reference: Path | None = None
    """Annotated single-cell reference ``.h5ad`` (``build-signature``, ``simulate``)."""
    signature: Path | None = None
    """Signature matrix TSV (``build-signature`` output; ``deconvolve`` input)."""
    bulk: Path | None = None
    """Bulk expression TSV (``simulate`` output; ``deconvolve``/``benchmark`` input)."""
    truth: Path | None = None
    """Ground-truth proportions TSV (``simulate`` output; ``benchmark`` input)."""
    proportions: Path | None = None
    """Estimated proportions TSV (``deconvolve`` output)."""
    metrics: Path | None = None
    """Benchmark metrics TSV (``benchmark`` output)."""
    heatmap: Path | None = None
    """Optional signature-heatmap PNG (``build-signature``)."""
    benchmark_plot: Path | None = None
    """Optional benchmark bar-chart PNG (``benchmark``)."""


class PreprocessingSettings(BaseModel):
    """Overrides for :class:`~scdecon.preprocessing.params.PreprocessConfig`.

    Fields default to ``None`` and are forwarded only when set, so unspecified
    values inherit the dataclass defaults. Semantic validation (e.g.
    ``max_pct_mito`` in ``[0, 100]``) is performed by the dataclass, not here.
    """

    model_config = ConfigDict(extra="forbid")

    min_genes: int | None = None
    min_cells: int | None = None
    max_pct_mito: float | None = None
    mito_prefix: str | None = None
    target_sum: float | None = None
    counts_layer: str | None = None


class MarkerSettings(BaseModel):
    """Overrides for :class:`~scdecon.signature.params.SignatureConfig`."""

    model_config = ConfigDict(extra="forbid")

    cell_type_key: str | None = None
    n_markers_per_type: int | None = None
    method: RankMethod | None = None
    min_cells_per_type: int | None = None


class SimulationSettings(BaseModel):
    """Overrides for :class:`~scdecon.simulation.params.SimulationConfig`."""

    model_config = ConfigDict(extra="forbid")

    n_samples: int | None = None
    n_cells_per_sample: int | None = None
    cell_type_key: str | None = None
    proportion_prior: ProportionPrior | None = None
    dirichlet_alpha: float | None = None
    random_state: int | None = None
    counts_layer: str | None = None


class SolverSettings(BaseModel):
    """Solver selection and parameters for ``deconvolve``.

    This is *declarative selection only*: the CLI composition root turns
    ``name`` (plus any relevant parameters) into a concrete
    :class:`~scdecon.deconvolution.base.Solver`. Parameters not relevant to the
    chosen solver are simply ignored by that factory.
    """

    model_config = ConfigDict(extra="forbid")

    name: SolverName = "nnls"
    nu: float | None = None
    """``nu`` for the nu-SVR solver (forwarded to ``NuSVRConfig`` by the CLI)."""
    loss: RobustLoss | None = None
    """Robust loss for the robust solver (forwarded to ``RobustConfig``)."""
    f_scale: float | None = None
    """Soft-margin scale for the robust solver (forwarded to ``RobustConfig``)."""


class BenchmarkSettings(BaseModel):
    """Solver set and reporting metric for ``benchmark``."""

    model_config = ConfigDict(extra="forbid")

    solvers: list[SolverName] = Field(
        default_factory=_default_benchmark_solvers, min_length=1
    )
    """Names of the solvers to compare (must be non-empty)."""
    metric: str | None = None
    """Metric for ranking/plotting; ``None`` lets the library default apply."""


class RunConfig(BaseModel):
    """Top-level validated run configuration for the scdecon CLI.

    Groups the per-concern settings and exposes ``to_*_config`` adapters that
    build the existing frozen parameter dataclasses. Adapter calls may raise
    :class:`ValueError` from a dataclass ``__post_init__`` when a value is out of
    range -- that is the intended fail-loud path and the single place semantic
    validation lives.
    """

    model_config = ConfigDict(extra="forbid")

    paths: PathsConfig = Field(default_factory=PathsConfig)
    preprocessing: PreprocessingSettings = Field(default_factory=PreprocessingSettings)
    markers: MarkerSettings = Field(default_factory=MarkerSettings)
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    solver: SolverSettings = Field(default_factory=SolverSettings)
    benchmark: BenchmarkSettings = Field(default_factory=BenchmarkSettings)

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        """Parse and validate a YAML run-configuration file.

        Parameters
        ----------
        path:
            Path to a YAML file whose top level is a mapping (an empty file is
            treated as "all defaults").

        Returns
        -------
        RunConfig
            The validated configuration.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If the YAML root is not a mapping.
        pydantic.ValidationError
            If any field has the wrong type, an unknown key is present, or an
            enum/selection value is not permitted (message lists the offending
            fields).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError(
                f"config root must be a mapping (key: value), got {type(data).__name__}"
            )
        return cls.model_validate(data)

    def to_preprocess_config(self) -> PreprocessConfig:
        """Build the frozen :class:`PreprocessConfig` from the set overrides."""
        return PreprocessConfig(**_set_fields(self.preprocessing))

    def to_signature_config(self) -> SignatureConfig:
        """Build the frozen :class:`SignatureConfig` from the set overrides."""
        return SignatureConfig(**_set_fields(self.markers))

    def to_simulation_config(self) -> SimulationConfig:
        """Build the frozen :class:`SimulationConfig` from the set overrides."""
        return SimulationConfig(**_set_fields(self.simulation))
