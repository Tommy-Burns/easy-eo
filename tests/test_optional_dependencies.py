"""Tests for the optional-extra import helper (eeo._optional)."""

import importlib
import importlib.metadata
import importlib.util
import json

import pytest

import eeo
from eeo import _optional
from eeo._optional import import_optional

# Modules provided by the optional extras, paired with the extra that
# installs them. Kept in sync with [project.optional-dependencies].
EXTRA_MODULES = [
    ("pystac_client", "stac"),
    ("planetary_computer", "stac"),
    ("xarray", "xarray"),
    ("rioxarray", "xarray"),
]

# `dev` is tooling, not a runtime feature: it is never passed to
# import_optional and needs no conda mapping.
NON_FEATURE_EXTRAS = {"dev"}

# The autouse fixture below replaces the module attribute, so the tests that
# exercise detection itself hold on to the real function.
detect_conda_install = _optional._installed_by_conda


@pytest.fixture(autouse=True)
def _pip_installed(monkeypatch):
    """Pin install detection to pip, so the hint does not depend on the runner.

    The install-manager branch is covered explicitly by the tests below; every
    other test in this module asserts the pip hint, which would otherwise flip
    on a machine where Easy-EO happens to be conda-installed.
    """
    monkeypatch.setattr(_optional, "_installed_by_conda", lambda: False)


def _write_conda_record(prefix, name="easy-eo-0.2.0-pyhcf101f3_0.json"):
    """Create a conda-meta record of the shape conda writes."""
    conda_meta = prefix / "conda-meta"
    conda_meta.mkdir(exist_ok=True)
    (conda_meta / name).write_text(json.dumps({"name": name.rsplit("-", 2)[0]}))


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


# ---------------------------------------------------------------------------
# Install-manager detection and the hint it produces
# ---------------------------------------------------------------------------


def test_conda_install_is_detected_from_the_conda_meta_record(monkeypatch, tmp_path):
    _write_conda_record(tmp_path)
    monkeypatch.setattr(_optional.sys, "prefix", str(tmp_path))

    assert detect_conda_install() is True


def test_environment_without_conda_meta_is_not_a_conda_install(monkeypatch, tmp_path):
    monkeypatch.setattr(_optional.sys, "prefix", str(tmp_path))

    assert detect_conda_install() is False


def test_pip_install_into_a_conda_environment_is_not_a_conda_install(monkeypatch, tmp_path):
    """A conda env whose easy-eo came from pip must still get the pip hint."""
    (tmp_path / "conda-meta").mkdir()
    _write_conda_record(tmp_path, name="numpy-2.5.1-py314h2b28147_0.json")
    monkeypatch.setattr(_optional.sys, "prefix", str(tmp_path))

    assert detect_conda_install() is False


def test_a_similarly_named_conda_package_is_not_mistaken_for_easy_eo(monkeypatch, tmp_path):
    """The record glob also matches a metapackage such as easy-eo-stac."""
    _write_conda_record(tmp_path, name="easy-eo-stac-0.2.0-pyhd8ed1ab_0.json")
    monkeypatch.setattr(_optional.sys, "prefix", str(tmp_path))

    assert detect_conda_install() is False


def test_unreadable_environment_reports_unknown(monkeypatch):
    def boom(self):
        raise OSError("permission denied")

    monkeypatch.setattr(_optional.Path, "is_dir", boom)

    assert detect_conda_install() is None


def test_conda_install_gets_the_conda_command_not_the_pip_extra(monkeypatch):
    monkeypatch.setattr(_optional, "_installed_by_conda", lambda: True)

    with pytest.raises(eeo.MissingDependencyError) as excinfo:
        import_optional("eeo_not_a_real_package", extra="stac", purpose="STAC search")

    message = str(excinfo.value)
    assert "conda install -c conda-forge pystac-client planetary-computer" in message
    # The bracket syntax is not merely unhelpful under conda, it cannot parse.
    assert "easy-eo[stac]" not in message
    # And the mixed install that would otherwise break on the next conda update
    # is warned against rather than left to be discovered.
    assert "Do not pip install" in message


def test_undetectable_install_offers_both_commands(monkeypatch):
    monkeypatch.setattr(_optional, "_installed_by_conda", lambda: None)

    with pytest.raises(eeo.MissingDependencyError) as excinfo:
        import_optional("eeo_not_a_real_package", extra="xarray", purpose="xarray interop")

    message = str(excinfo.value)
    assert "pip install 'easy-eo[xarray]'" in message
    assert "conda install -c conda-forge xarray rioxarray" in message


@pytest.mark.parametrize("by_conda", [True, False, None])
def test_an_unmapped_extra_falls_back_to_the_pip_hint(monkeypatch, by_conda):
    """An extra with no conda mapping must not crash the error path."""
    monkeypatch.setattr(_optional, "_installed_by_conda", lambda: by_conda)

    with pytest.raises(eeo.MissingDependencyError) as excinfo:
        import_optional("eeo_not_a_real_package", extra="lazy", purpose="lazy backend")

    assert "pip install 'easy-eo[lazy]'" in str(excinfo.value)


def test_every_declared_extra_has_conda_package_names():
    """A new extra cannot ship without its conda equivalent."""
    declared = set(importlib.metadata.metadata("easy-eo").get_all("Provides-Extra") or [])
    feature_extras = declared - NON_FEATURE_EXTRAS

    assert feature_extras, "no feature extras found in the installed metadata"
    assert feature_extras <= set(_optional._CONDA_PACKAGES)
