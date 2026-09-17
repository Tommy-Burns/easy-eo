"""Input/output functionality for Easy-EO."""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine

from eeo.core.core import EEORasterDataset
from eeo.core.exceptions import BackendError, MissingDependencyError, ValidationError
from eeo.core.types import ChunkSpec, StrPath


def load_raster(
    path: StrPath,
    *,
    chunks: ChunkSpec | None = None,
    timestamp: datetime | None = None,
    attrs: dict | None = None,
    band_names: list[str | None] | None = None,
) -> EEORasterDataset:
    """Open a raster file as an EEORasterDataset.

    The file is opened but pixel data is not read until an operation needs
    it, so opening a large scene is cheap.

    Parameters
    ----------
    path : str or path-like
        Path to a GDAL-readable raster file.
    chunks : str or int or dict or None, default None
        ``None`` opens the file with rasterio. Anything else opens it on the
        lazy backend: an :class:`xarray.DataArray` split into dask chunks of
        this size, from which only the bands and windows an operation asks for
        are computed. ``"auto"`` lets dask choose sizes aligned with the file's
        internal blocks; an int sets the size of every dimension; a dict sets
        some of ``"band"``, ``"y"`` and ``"x"`` (e.g. ``{"y": 2048,
        "x": 2048}``). Needs the ``lazy`` extra
        (``pip install "easy-eo[lazy]"``).
    timestamp : datetime.datetime or None, default None
        Optional acquisition time carried with the dataset and preserved
        through operations.
    attrs : dict or None, default None
        Optional free-form tags dict carried with the dataset and preserved
        through operations.
    band_names : list of (str or None) or None, default None
        Optional per-band names, one entry per band. When omitted, names are
        read from the file's GDAL band descriptions; when given, they override
        whatever the file declares and must match the band count.

    Returns
    -------
    EEORasterDataset
        A rasterio-backed dataset, or an xarray-backed one when ``chunks`` is
        given.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    BackendError
        If the file exists but cannot be opened as a raster.
    MissingDependencyError
        If ``chunks`` is given and the ``lazy`` extra is not installed.
    ValidationError
        If ``chunks`` is not a valid chunk specification, or ``band_names`` is
        given and its length does not match the band count.

    Examples
    --------
    >>> ds = load_raster("scene.tif")
    >>> ds = load_raster("stack.tif", band_names=["blue", "green", "red", "nir"])
    >>> ds = load_raster("scene.tif", chunks="auto")  # lazy, dask-chunked
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f'the file "{path}" does not exist')
    try:
        ds = EEORasterDataset.from_path(path, chunks=chunks)
    except (MissingDependencyError, ValidationError):
        raise
    except Exception as e:
        raise BackendError(f'file "{path}" could not be opened as a raster') from e

    ds.timestamp = timestamp
    if attrs is not None:
        ds.attrs = dict(attrs)
    if band_names is not None:
        # Explicit names win over the file's own band descriptions.
        ds.band_names = band_names
    return ds


def load_array(
    array: np.ndarray,
    *,
    transform: Affine | None = None,
    crs: CRS | int | str | None = None,
    nodata: float | int | None = None,
    timestamp: datetime | None = None,
    attrs: dict | None = None,
    band_names: list[str | None] | None = None,
) -> EEORasterDataset:
    """Wrap an in-memory NumPy array as an EEORasterDataset.

    Parameters
    ----------
    array : numpy.ndarray
        Raster values, shaped ``(height, width)`` for a single band or
        ``(bands, height, width)`` for multiple bands.
    transform : affine.Affine or None, default None
        Affine geotransform mapping pixel to world coordinates. If None, the
        dataset has no meaningful georeferencing.
    crs : rasterio.crs.CRS or int or str or None, default None
        Coordinate reference system (e.g. an EPSG code such as ``4326``). If
        None, the dataset is unreferenced.
    nodata : float or int or None, default None
        Value marking nodata pixels, stored in the metadata.
    timestamp : datetime.datetime or None, default None
        Optional acquisition time carried with the dataset and preserved
        through operations.
    attrs : dict or None, default None
        Optional free-form tags dict carried with the dataset and preserved
        through operations.
    band_names : list of (str or None) or None, default None
        Optional per-band names, one entry per band (``None`` for an unnamed
        band). Must match the band count.

    Returns
    -------
    EEORasterDataset
        A NumPy-backed dataset. The array is wrapped without copying;
        operations that need rasterio (clipping, resampling, ...) promote it
        on demand.

    Raises
    ------
    ValidationError
        If ``array`` is not a NumPy array, is neither 2D nor 3D, or
        ``band_names`` is given and its length does not match the band count.

    Examples
    --------
    >>> import numpy as np
    >>> ds = load_array(np.zeros((64, 64), dtype="float32"), crs=4326)
    >>> ds = load_array(rgb, crs=4326, band_names=["red", "green", "blue"])
    """
    if not isinstance(array, np.ndarray):
        raise ValidationError(f"array must be a NumPy ndarray; got {type(array).__name__}")

    if array.ndim not in (2, 3):
        raise ValidationError(
            "array must be 2D (height, width) or 3D (bands, height, width); "
            f"got {array.ndim}D with shape {array.shape}"
        )

    return EEORasterDataset.from_array(
        array=array,
        transform=transform,
        crs=crs,
        nodata=nodata,
        timestamp=timestamp,
        attrs=attrs,
        band_names=band_names,
    )
