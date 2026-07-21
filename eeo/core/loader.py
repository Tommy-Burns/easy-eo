"""Input/output functionality for Easy-EO."""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine

from eeo.core.core import EEORasterDataset
from eeo.core.exceptions import BackendError, ValidationError


def load_raster(
    path: str,
    *,
    timestamp: datetime | None = None,
    attrs: dict | None = None,
) -> EEORasterDataset:
    """Open a raster file as an EEORasterDataset.

    The file is opened but pixel data is not read until an operation needs
    it, so opening a large scene is cheap.

    Parameters
    ----------
    path : str
        Path to a GDAL-readable raster file.
    timestamp : datetime.datetime or None, default None
        Optional acquisition time carried with the dataset and preserved
        through operations.
    attrs : dict or None, default None
        Optional free-form tags dict carried with the dataset and preserved
        through operations.

    Returns
    -------
    EEORasterDataset
        A rasterio-backed dataset.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    BackendError
        If the file exists but cannot be opened as a raster.

    Examples
    --------
    >>> ds = load_raster("scene.tif")
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f'the file "{path}" does not exist')
    try:
        ds = EEORasterDataset.from_path(path)
    except Exception as e:
        raise BackendError(f'file "{path}" could not be opened as a rasterio dataset') from e

    ds.timestamp = timestamp
    if attrs is not None:
        ds.attrs = dict(attrs)
    return ds


def load_array(
    array: np.ndarray,
    *,
    transform: Affine | None = None,
    crs: CRS | int | str | None = None,
    nodata: float | int | None = None,
    timestamp: datetime | None = None,
    attrs: dict | None = None,
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

    Returns
    -------
    EEORasterDataset
        A NumPy-backed dataset. The array is wrapped without copying;
        operations that need rasterio (clipping, resampling, ...) promote it
        on demand.

    Raises
    ------
    ValidationError
        If ``array`` is not a NumPy array, or is neither 2D nor 3D.

    Examples
    --------
    >>> import numpy as np
    >>> ds = load_array(np.zeros((64, 64), dtype="float32"), crs=4326)
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
    )
