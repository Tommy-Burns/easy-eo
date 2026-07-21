import platform

import eeo
from eeo._show_versions import _get_versions


def test_show_versions_is_public():
    assert "show_versions" in eeo.__all__
    assert eeo.show_versions is not None


def test_get_versions_reports_core_stack():
    info = _get_versions()

    # Required components per the bug-report template.
    for key in ("easy-eo", "python", "OS", "rasterio", "GDAL", "numpy", "geopandas", "matplotlib"):
        assert key in info
        assert info[key], f"{key} version should be non-empty"

    assert info["python"] == platform.python_version()


def test_get_versions_reports_optional_extras():
    info = _get_versions()

    # Optional-extra deps are always listed, present or not.
    for key in ("xarray", "rioxarray", "dask", "pystac-client", "planetary-computer"):
        assert key in info


def test_show_versions_prints_report(capsys):
    eeo.show_versions()
    out = capsys.readouterr().out

    assert "Easy-EO version information" in out
    for label in ("easy-eo", "python", "rasterio", "GDAL", "numpy", "geopandas", "matplotlib"):
        assert label in out


def test_show_versions_returns_none():
    assert eeo.show_versions() is None
