"""Tests for the Pydantic run-configuration layer (``scdecon.config``).

The configuration layer is a validation boundary: it parses YAML, validates its
structure/types, and constructs the existing frozen parameter dataclasses. These
tests pin the contract that matters most -- that the frozen dataclasses remain
the single source of truth for defaults and semantic validation, and that
malformed configs fail loudly with informative errors.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from scdecon.config import RunConfig
from scdecon.preprocessing.params import PreprocessConfig
from scdecon.signature.params import RankMethod, SignatureConfig
from scdecon.simulation.params import ProportionPrior, SimulationConfig


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "run.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_empty_config_matches_dataclass_defaults(tmp_path: Path) -> None:
    """An empty config yields exactly the frozen-dataclass defaults (no drift)."""
    cfg = RunConfig.load(_write(tmp_path, ""))
    assert cfg.to_preprocess_config() == PreprocessConfig()
    assert cfg.to_signature_config() == SignatureConfig()
    assert cfg.to_simulation_config() == SimulationConfig()


def test_overrides_are_applied_and_others_defaulted(tmp_path: Path) -> None:
    """Only set fields are overridden; the rest inherit dataclass defaults."""
    cfg = RunConfig.load(
        _write(
            tmp_path,
            """
            preprocessing:
              min_genes: 100
              max_pct_mito: 10.0
            markers:
              n_markers_per_type: 40
              method: t-test
            simulation:
              n_samples: 5
              proportion_prior: uniform
            """,
        )
    )
    pre = cfg.to_preprocess_config()
    assert pre.min_genes == 100
    assert pre.max_pct_mito == 10.0
    assert pre.min_cells == PreprocessConfig().min_cells  # untouched -> default

    sig = cfg.to_signature_config()
    assert sig.n_markers_per_type == 40
    assert sig.method is RankMethod.T_TEST

    sim = cfg.to_simulation_config()
    assert sim.n_samples == 5
    assert sim.proportion_prior is ProportionPrior.UNIFORM


def test_explicit_null_target_sum_is_preserved(tmp_path: Path) -> None:
    """An explicit ``null`` is kept (median mode), not reset to the default."""
    cfg = RunConfig.load(_write(tmp_path, "preprocessing:\n  target_sum: null\n"))
    assert cfg.to_preprocess_config().target_sum is None
    # Sanity: omitting it entirely falls back to the dataclass default instead.
    default_sum = RunConfig.load(_write(tmp_path, "")).to_preprocess_config().target_sum
    assert default_sum == PreprocessConfig().target_sum


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    """Unknown keys are a hard error (``extra='forbid'``)."""
    with pytest.raises(ValidationError):
        RunConfig.load(_write(tmp_path, "preprocessing:\n  bogus_key: 1\n"))


def test_unknown_top_level_section_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        RunConfig.load(_write(tmp_path, "not_a_section: {}\n"))


def test_out_of_range_value_surfaces_dataclass_error(tmp_path: Path) -> None:
    """Range validation is delegated to the frozen dataclass (fail loud)."""
    cfg = RunConfig.load(_write(tmp_path, "preprocessing:\n  min_genes: -1\n"))
    with pytest.raises(ValueError, match="min_genes must be >= 0"):
        cfg.to_preprocess_config()


def test_invalid_enum_is_rejected_at_load(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        RunConfig.load(_write(tmp_path, "markers:\n  method: not-a-method\n"))


def test_wrong_type_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        RunConfig.load(_write(tmp_path, "simulation:\n  n_samples: not-an-int\n"))


def test_paths_parsed_as_path_objects(tmp_path: Path) -> None:
    cfg = RunConfig.load(
        _write(
            tmp_path,
            """
            paths:
              reference: data/ref.h5ad
              signature: out/signature.tsv
            """,
        )
    )
    assert cfg.paths.reference == Path("data/ref.h5ad")
    assert cfg.paths.signature == Path("out/signature.tsv")
    assert cfg.paths.bulk is None


def test_solver_selection_validated(tmp_path: Path) -> None:
    cfg = RunConfig.load(_write(tmp_path, "solver:\n  name: nusvr\n  nu: 0.25\n"))
    assert cfg.solver.name == "nusvr"
    assert cfg.solver.nu == 0.25
    with pytest.raises(ValidationError):
        RunConfig.load(_write(tmp_path, "solver:\n  name: bogus\n"))


def test_benchmark_solver_list_validated(tmp_path: Path) -> None:
    cfg = RunConfig.load(_write(tmp_path, "benchmark:\n  solvers: [nnls, robust]\n"))
    assert cfg.benchmark.solvers == ["nnls", "robust"]
    # Empty solver list is rejected (min_length=1).
    with pytest.raises(ValidationError):
        RunConfig.load(_write(tmp_path, "benchmark:\n  solvers: []\n"))
    # An unknown solver name in the list is rejected.
    with pytest.raises(ValidationError):
        RunConfig.load(_write(tmp_path, "benchmark:\n  solvers: [nnls, bogus]\n"))


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        RunConfig.load(tmp_path / "does_not_exist.yaml")


def test_non_mapping_root_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="config root must be a mapping"):
        RunConfig.load(_write(tmp_path, "- just\n- a\n- list\n"))
