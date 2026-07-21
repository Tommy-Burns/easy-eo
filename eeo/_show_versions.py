"""Environment reporting for bug reports and support requests."""

from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version

# Optional dependencies that back the planned optional extras. Reported so a
# bug report shows which extras are active. Maps the display label to the
# installed distribution name.
_OPTIONAL_DISTRIBUTIONS = {
    "xarray": "xarray",
    "rioxarray": "rioxarray",
    "dask": "dask",
    "pystac-client": "pystac-client",
    "planetary-computer": "planetary-computer",
}


def _distribution_version(name: str) -> str:
    """Return an installed distribution's version, or ``"not installed"``.

    Parameters
    ----------
    name : str
        Distribution (PyPI) name to look up.

    Returns
    -------
    str
        The installed version string, or ``"not installed"`` if the
        distribution is not present.
    """
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def _get_versions() -> dict[str, str]:
    """Collect version information for Easy-EO and its dependencies.

    Returns
    -------
    dict of str to str
        Ordered mapping of component name to version string. Covers easy-eo,
        the Python runtime and platform, the core geospatial stack (rasterio,
        GDAL, numpy, geopandas, matplotlib), and each optional-extra
        dependency (reported as ``"not installed"`` when absent).
    """
    # Imported lazily to keep `import eeo` light; all are hard dependencies.
    import geopandas
    import matplotlib
    import numpy
    import rasterio

    gdal_version = getattr(rasterio, "__gdal_version__", None) or rasterio.gdal_version()

    info = {
        "easy-eo": _distribution_version("easy-eo"),
        "python": platform.python_version(),
        "OS": f"{platform.system()} {platform.release()}",
        "rasterio": rasterio.__version__,
        "GDAL": str(gdal_version),
        "numpy": numpy.__version__,
        "geopandas": geopandas.__version__,
        "matplotlib": matplotlib.__version__,
    }
    for label, dist in _OPTIONAL_DISTRIBUTIONS.items():
        info[label] = _distribution_version(dist)
    return info


def show_versions() -> None:
    """Print Easy-EO, Python, and dependency version information.

    Writes a human-readable report to standard output covering easy-eo, the
    Python runtime and operating system, the core geospatial stack (rasterio,
    GDAL, numpy, geopandas, matplotlib), and the optional-extra dependencies
    (each shown as its version or ``not installed``). Intended for pasting
    into bug reports.

    Returns
    -------
    None
        The report is printed; nothing is returned.

    Examples
    --------
    >>> import eeo
    >>> eeo.show_versions()  # doctest: +SKIP
    Easy-EO version information
    ...
    """
    info = _get_versions()
    width = max(len(name) for name in info)

    lines = ["Easy-EO version information", "=========================="]
    for name, value in info.items():
        lines.append(f"{name:<{width}} : {value}")
    print("\n".join(lines))
