"""Architectural guardrails enforced as tests.

The preprocessing layer must operate purely on in-memory AnnData objects and
must not reach into the I/O layer: ``scdecon.io`` stays solely responsible for
reading and writing data. This is checked statically (by parsing imports) so the
guarantee holds regardless of test import order.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import scdecon.preprocessing


def _imports(tree: ast.AST) -> Iterator[tuple[str, int]]:
    """Yield (module, relative_level) for every import statement in a module."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, 0
        elif isinstance(node, ast.ImportFrom):
            yield (node.module or ""), node.level


def _references_io(module: str, level: int) -> bool:
    """Whether an import refers to the scdecon.io package (absolute or relative)."""
    if level == 0:
        return module == "scdecon.io" or module.startswith("scdecon.io.")
    # relative import from within the scdecon package, e.g. ``from ..io import x``
    return module == "io" or module.startswith("io.")


def test_preprocessing_does_not_import_io() -> None:
    package_dir = Path(scdecon.preprocessing.__file__).parent
    offenders: list[str] = []
    for path in sorted(package_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for module, level in _imports(tree):
            if _references_io(module, level):
                offenders.append(f"{path.name}: level={level} module={module!r}")
    assert not offenders, (
        "scdecon.preprocessing must not import scdecon.io (I/O stays in the io "
        f"layer); found: {offenders}"
    )
