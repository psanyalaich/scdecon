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
    [scdecon.preprocessing, scdecon.signature, scdecon.deconvolution],
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
