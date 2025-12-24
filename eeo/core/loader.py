"""
Input/Output functionalities for easy-eo
"""
from __future__ import annotations

import os

import numpy as np
from typing import Optional, Tuple
from rasterio.transform import Affine
from rasterio.crs import CRS
from eeo.core.core import EEORasterDataset


def load_raster(path: str) -> EEORasterDataset:
    if not os.path.isfile(path):
        raise FileNotFoundError(f'The file "{path}" does not exist')
    try:
        return EEORasterDataset.from_path(path)
    except Exception as e:
        raise RuntimeError( f'File "{path}" could not be opened as a rasterio dataset') from e


def load_array(
        array: np.ndarray,
        *,
        transform: Optional[Affine] = None,
        crs: Optional[CRS | int | str] = None,
        nodata: Optional[float | int] = None,
) -> EEORasterDataset:
    if not isinstance(array, np.ndarray):
        raise TypeError('The array must be a numpy array')

    if array.ndim not in (2, 3):
        raise ValueError('The array must be 2D or 3D (bands, height, width)')

    return EEORasterDataset.from_array(
        array=array,
        transform=transform,
        crs=crs,
        nodata=nodata
    )
