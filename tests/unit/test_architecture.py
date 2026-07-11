"""Architectural guardrails enforced as tests.

Dependency direction is checked statically (by parsing imports) so the
guarantees hold regardless of test import order:

- The computation layers (preprocessing, signature, deconvolution) must not
  import the I/O layer ``scdecon.io``.
- The computational core (io, preprocessing, signature, deconvolution) must not
  import the plotting stack (``scdecon.plotting``, matplotlib, seaborn): plotting
  depends on the core, never the reverse.
- The deconvolution layer must not import biological data containers (anndata,
  scanpy); it is a numerical layer.
- The solver-core modules (deconvolution/base.py, nnls.py, ...) must be
  format-agnostic: NumPy/SciPy only, no pandas/anndata/scanpy/io/plotting.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

import scdecon.deconvolution
import scdecon.io
import scdecon.preprocessing
import scdecon.signature
import scdecon.simulation
import scdecon.validation

#: Modules inside ``scdecon.deconvolution`` that hold the numerical solver core
#: and must operate purely on NumPy/SciPy arrays.
_SOLVER_CORE_FILES = ("base.py", "nnls.py", "nusvr.py", "robust.py")


def _imports(tree: ast.AST) -> Iterator[tuple[str, int]]:
    """Yield (module, relative_level) for every import statement in a module."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, 0
        elif isinstance(node, ast.ImportFrom):
            yield (node.module or ""), node.level


def _matches(module: str, level: int, absolute: set[str], relative: set[str]) -> bool:
    """Whether an import refers to any forbidden package (absolute or relative)."""
    names = absolute if level == 0 else relative
    return any(module == name or module.startswith(f"{name}.") for name in names)


def _offenders(
    package: ModuleType, absolute: set[str], relative: set[str]
) -> list[str]:
    assert package.__file__ is not None
    package_dir = Path(package.__file__).parent
    found: list[str] = []
    for path in sorted(package_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for module, level in _imports(tree):
            if _matches(module, level, absolute, relative):
                found.append(f"{package.__name__}/{path.name}: {module!r}")
    return found


@pytest.mark.parametrize(
    "package",
    [
        scdecon.preprocessing,
        scdecon.signature,
        scdecon.deconvolution,
        scdecon.simulation,
        scdecon.validation,
    ],
    ids=lambda pkg: pkg.__name__,
)
def test_computation_layer_does_not_import_io(package: ModuleType) -> None:
    offenders = _offenders(package, {"scdecon.io"}, {"io"})
    assert not offenders, (
        f"{package.__name__} must not import scdecon.io; found: {offenders}"
    )


@pytest.mark.parametrize(
    "package",
    [
        scdecon.io,
        scdecon.preprocessing,
        scdecon.signature,
        scdecon.deconvolution,
        scdecon.simulation,
        scdecon.validation,
    ],
    ids=lambda pkg: pkg.__name__,
)
def test_core_does_not_import_plotting_stack(package: ModuleType) -> None:
    offenders = _offenders(
        package,
        {"scdecon.plotting", "matplotlib", "seaborn"},
        {"plotting"},
    )
    assert not offenders, (
        f"{package.__name__} must not import the plotting stack (plotting depends "
        f"on the core, never the reverse); found: {offenders}"
    )


def test_deconvolution_does_not_import_biological_containers() -> None:
    offenders = _offenders(scdecon.deconvolution, {"anndata", "scanpy"}, set())
    assert not offenders, (
        "scdecon.deconvolution is a numerical layer and must not import anndata "
        f"or scanpy; found: {offenders}"
    )


def test_sklearn_is_isolated_to_nusvr() -> None:
    assert scdecon.deconvolution.__file__ is not None
    package_dir = Path(scdecon.deconvolution.__file__).parent
    offenders: list[str] = []
    for path in sorted(package_dir.rglob("*.py")):
        if path.name == "nusvr.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for module, level in _imports(tree):
            if level == 0 and (module == "sklearn" or module.startswith("sklearn.")):
                offenders.append(f"{path.name}: {module!r}")
    assert not offenders, (
        f"scikit-learn must be isolated to deconvolution/nusvr.py; found: {offenders}"
    )


_CONCRETE_SOLVERS = {"NNLSSolver", "NuSVRSolver", "RobustSolver"}
_CONCRETE_SOLVER_MODULES = {
    "scdecon.deconvolution.nnls",
    "scdecon.deconvolution.nusvr",
    "scdecon.deconvolution.robust",
}


def test_benchmark_is_solver_agnostic() -> None:
    """benchmark.py must depend only on the Solver ABC, never a concrete solver."""
    assert scdecon.deconvolution.__file__ is not None
    benchmark_path = Path(scdecon.deconvolution.__file__).parent / "benchmark.py"
    tree = ast.parse(benchmark_path.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _CONCRETE_SOLVER_MODULES:
                    offenders.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module in _CONCRETE_SOLVER_MODULES:
                offenders.append(node.module)
            for alias in node.names:
                if alias.name in _CONCRETE_SOLVERS:
                    offenders.append(alias.name)
    assert not offenders, (
        "benchmark.py must be solver-agnostic (import only the Solver ABC); "
        f"found references to concrete solvers: {offenders}"
    )


def test_src_does_not_import_scripts() -> None:
    """The installable package must never depend on dataset-specific scripts/.

    ``scripts/`` holds dataset-specific ingestion (ADR-0010) and lives outside the
    package; the dependency may only point scripts -> scdecon, never the reverse.
    """
    import scdecon

    assert scdecon.__file__ is not None
    package_dir = Path(scdecon.__file__).parent
    offenders: list[str] = []
    for path in sorted(package_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for module, level in _imports(tree):
            if level == 0 and (module == "scripts" or module.startswith("scripts.")):
                offenders.append(f"{path.name}: {module!r}")
    assert not offenders, (
        f"scdecon must not import the scripts/ package; found: {offenders}"
    )


def test_genes_module_is_dataset_agnostic() -> None:
    """scdecon.genes holds generic gene-ID utilities and must stay reusable.

    It must not reach into the I/O or plotting layers, pull in biological data
    containers (anndata/scanpy), or otherwise couple to any dataset or format --
    it operates only on in-memory strings/DataFrames plus a caller-supplied
    mapping.
    """
    import scdecon

    assert scdecon.__file__ is not None
    genes_path = Path(scdecon.__file__).parent / "genes.py"
    forbidden = {
        "scdecon.io",
        "scdecon.plotting",
        "anndata",
        "scanpy",
        "matplotlib",
        "seaborn",
    }
    tree = ast.parse(genes_path.read_text(encoding="utf-8"))
    offenders = [
        f"genes.py: {module!r}"
        for module, level in _imports(tree)
        if level == 0
        and any(module == name or module.startswith(f"{name}.") for name in forbidden)
    ]
    assert not offenders, (
        "scdecon.genes must stay dataset-agnostic (no io/plotting/anndata/scanpy); "
        f"found: {offenders}"
    )


def test_solver_core_modules_are_format_agnostic() -> None:
    assert scdecon.deconvolution.__file__ is not None
    package_dir = Path(scdecon.deconvolution.__file__).parent
    forbidden = {"pandas", "anndata", "scanpy", "scdecon.io", "scdecon.plotting"}
    offenders: list[str] = []
    for filename in _SOLVER_CORE_FILES:
        path = package_dir / filename
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for module, level in _imports(tree):
            if level == 0 and any(
                module == name or module.startswith(f"{name}.") for name in forbidden
            ):
                offenders.append(f"{filename}: {module!r}")
    assert not offenders, (
        "solver-core modules must be NumPy/SciPy-only (no pandas/anndata/scanpy/"
        f"io/plotting); found: {offenders}"
    )
