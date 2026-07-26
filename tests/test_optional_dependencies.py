"""Tests for the optional-extra import helper (eeo._optional)."""

import importlib
import importlib.util

import pytest

import eeo
from eeo._optional import import_optional

# Modules provided by the optional extras, paired with the extra that
# installs them. Kept in sync with [project.optional-dependencies].
EXTRA_MODULES = [
    ("pystac_client", "stac"),
    ("planetary_computer", "stac"),
    ("xarray", "xarray"),
    ("rioxarray", "xarray"),
]


def test_import_optional_returns_installed_module():
    # numpy is a core dependency, so the success path is exercised without
    # requiring any extra to be installed.
    module = import_optional("numpy", extra="stac", purpose="STAC search")

    assert module is importlib.import_module("numpy")


@pytest.mark.parametrize(("module", "extra"), EXTRA_MODULES)
def test_extra_module_is_importable_or_raises_helpful_error(module, extra):
    """Whether or not the extra is installed, the outcome is documented."""
    purpose = f"the {extra} extra"
    if importlib.util.find_spec(module) is not None:
        assert import_optional(module, extra=extra, purpose=purpose) is not None
        return

    with pytest.raises(eeo.MissingDependencyError) as excinfo:
        import_optional(module, extra=extra, purpose=purpose)

    assert f"pip install 'easy-eo[{extra}]'" in str(excinfo.value)


def test_missing_module_raises_missing_dependency_error():
    with pytest.raises(eeo.MissingDependencyError) as excinfo:
        import_optional(
            "eeo_not_a_real_package",
            extra="stac",
            purpose="STAC search",
        )

    message = str(excinfo.value)
    assert "STAC search" in message
    assert "eeo_not_a_real_package" in message
    assert "pip install 'easy-eo[stac]'" in message


def test_missing_dependency_error_is_catchable_as_import_error():
    with pytest.raises(ImportError):
        import_optional("eeo_not_a_real_package", extra="stac", purpose="STAC search")

    with pytest.raises(eeo.EEOError):
        import_optional("eeo_not_a_real_package", extra="stac", purpose="STAC search")


@pytest.mark.parametrize(("module", "extra"), EXTRA_MODULES)
def test_simulated_absence_of_an_installed_package(monkeypatch, module, extra):
    """The absent case is reported even when the package is installed."""

    def fake_import_module(name):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(eeo.MissingDependencyError) as excinfo:
        import_optional(module, extra=extra, purpose=f"the {extra} extra")

    message = str(excinfo.value)
    assert module in message
    assert f"pip install 'easy-eo[{extra}]'" in message


def test_error_inside_the_optional_package_propagates_unchanged(monkeypatch):
    """A missing *transitive* dependency is not blamed on the extra."""

    def fake_import_module(name):
        raise ModuleNotFoundError(
            "No module named 'some_transitive_dep'", name="some_transitive_dep"
        )

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        import_optional("pystac_client", extra="stac", purpose="STAC search")

    assert not isinstance(excinfo.value, eeo.MissingDependencyError)
    assert excinfo.value.name == "some_transitive_dep"
