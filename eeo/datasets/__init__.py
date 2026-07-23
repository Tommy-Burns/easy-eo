"""Curated sample datasets with checksum-verified caching.

The functions here download a small, hosted Sentinel-2 / Copernicus-DEM sample
so tutorials and quickstarts run in minutes without hunting for data. Files are
cached under ``~/.cache/easy-eo`` (override with ``EEO_DATA_DIR``) and verified
against a checksum shipped inside the package, so a fetch is fast after the
first call and never returns corrupt data.

Downloading uses only the Python standard library - no extra dependency.

Examples
--------
>>> import eeo
>>> ds = eeo.datasets.load("sentinel2_small")
>>> ndvi = ds.ndvi(red="red", nir="nir")
>>> boundary = eeo.datasets.fetch("sentinel2_small_boundary")
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio

from eeo.core.core import EEORasterDataset
from eeo.core.loader import load_array, load_raster

from ._cache import DatasetError, cache_dir, ensure_asset
from ._registry import available, get_dataset

__all__ = ["fetch", "load", "info", "available", "cache_dir", "DatasetError"]


def fetch(name: str) -> Path | list[Path]:
    """Download a sample dataset and return its cached path(s).

    The file is downloaded on first use and verified against a pinned checksum;
    subsequent calls return the cached copy without touching the network.

    Parameters
    ----------
    name : str
        A registered dataset name. See :func:`available` for the full list.

    Returns
    -------
    pathlib.Path or list of pathlib.Path
        The cached file path for a single-file dataset, or a list of paths (in
        band order) for a multi-file dataset such as ``"sentinel2_small"``.

    Raises
    ------
    KeyError
        If ``name`` is not a registered dataset.
    DatasetError
        If a download fails or a file fails checksum verification.

    Examples
    --------
    >>> boundary = fetch("sentinel2_small_boundary")
    >>> band_paths = fetch("sentinel2_small")
    """
    dataset = get_dataset(name)
    paths = [ensure_asset(asset) for asset in dataset.assets]
    return paths if dataset.multi else paths[0]


def load(name: str) -> EEORasterDataset | Path:
    """Fetch a sample dataset and open it ready to use.

    Raster datasets are returned as an :class:`~eeo.core.core.EEORasterDataset`
    with band names and acquisition timestamp already set; a single-file raster
    is opened lazily via :func:`eeo.load_raster`, while a multi-band dataset
    backed by several single-band files is read into an in-memory stack (the
    sample is small by design). Vector datasets have no raster representation,
    so their cached path is returned unchanged — read them with GeoPandas.

    Parameters
    ----------
    name : str
        A registered dataset name. See :func:`available` for the full list.

    Returns
    -------
    EEORasterDataset or pathlib.Path
        An opened dataset for raster entries; the cached file path for vector
        entries.

    Raises
    ------
    KeyError
        If ``name`` is not a registered dataset.
    DatasetError
        If a download fails or a file fails checksum verification.

    Notes
    -----
    A multi-file raster (e.g. ``"sentinel2_small"``) is read eagerly into a
    NumPy-backed dataset when stacked; single-file rasters remain lazy until an
    operation needs pixels.

    Examples
    --------
    >>> ds = load("sentinel2_small")
    >>> ds.band_names
    ['blue', 'green', 'red', 'nir']
    >>> import geopandas as gpd
    >>> roi = gpd.read_file(load("sentinel2_small_boundary"))
    """
    dataset = get_dataset(name)
    paths = fetch(name)

    if dataset.kind == "vector":
        return paths  # type: ignore[return-value]  # single-asset vector -> Path

    if not dataset.multi:
        return load_raster(
            str(paths),  # single-asset raster -> Path
            timestamp=dataset.timestamp,
            band_names=dataset.band_names or None,
        )

    return _load_stack(paths, dataset.timestamp, dataset.nodata, dataset.band_names)  # type: ignore[arg-type]


def _load_stack(
    paths: list[Path],
    timestamp: datetime | None,
    nodata: float | int | None,
    band_names: list[str | None],
) -> EEORasterDataset:
    """Read single-band files into one georeferenced multi-band dataset."""
    bands = []
    transform = crs = None
    for path in paths:
        with rasterio.open(path) as src:
            if transform is None:
                transform, crs = src.transform, src.crs
            bands.append(src.read(1))
    stack = np.stack(bands)
    return load_array(
        stack,
        transform=transform,
        crs=crs,
        nodata=nodata,
        timestamp=timestamp,
        band_names=band_names or None,
    )


def info(name: str) -> str:
    """Return a human-readable description and attribution for a dataset.

    Parameters
    ----------
    name : str
        A registered dataset name.

    Returns
    -------
    str
        A multi-line summary: the description followed by the data
        licence/attribution that must accompany reuse.

    Raises
    ------
    KeyError
        If ``name`` is not a registered dataset.

    Examples
    --------
    >>> print(info("sentinel2_small"))
    """
    dataset = get_dataset(name)
    files = ", ".join(asset.remote for asset in dataset.assets)
    return (
        f"{dataset.name} ({dataset.kind})\n"
        f"  {dataset.description}\n"
        f"  files: {files}\n"
        f"  attribution: {dataset.attribution}"
    )
