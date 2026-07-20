"""
Input/Output functionalities for easy-eo
"""

from __future__ import annotations

import os

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine

from eeo.core.core import EEORasterDataset


def load_raster(path: str) -> EEORasterDataset:
    if not os.path.isfile(path):
        raise FileNotFoundError(f'The file "{path}" does not exist')
    try:
        return EEORasterDataset.from_path(path)
    except Exception as e:
        raise RuntimeError(f'File "{path}" could not be opened as a rasterio dataset') from e


def load_array(
    array: np.ndarray,
    *,
    transform: Affine | None = None,
    crs: CRS | int | str | None = None,
    nodata: float | int | None = None,
) -> EEORasterDataset:
    if not isinstance(array, np.ndarray):
        raise TypeError("The array must be a numpy array")

    if array.ndim not in (2, 3):
        raise ValueError("The array must be 2D or 3D (bands, height, width)")

    return EEORasterDataset.from_array(array=array, transform=transform, crs=crs, nodata=nodata)
