"""Smoke test: the package imports and exposes a version string."""

import scdecon


def test_package_imports() -> None:
    assert scdecon is not None


def test_version_is_nonempty_string() -> None:
    assert isinstance(scdecon.__version__, str)
    assert scdecon.__version__
