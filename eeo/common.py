from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from rasterio.enums import Resampling

if TYPE_CHECKING:
    # Type-hints only: eeo.core's package init (via load_ops()) imports
    # eeo.ops/eeo.analysis/etc, which import from this module - a real
    # runtime import here would be circular. Neither function below needs
    # EEORasterDataset at runtime; both just call duck-typed methods on it.
    from eeo.core.core import EEORasterDataset


def normalize_resampling_method(value):
    """Function to normalize resampling method either user passes a rasterio.enums.Resampling or a string"""
    if isinstance(value, Resampling):
        return value
    if isinstance(value, str):
        name = value.lower().strip()
        try:
            return Resampling[name]
        except KeyError as e:
            valid = ", ".join([r.name for r in Resampling])
            raise ValueError(f"Invalid resampling method {value}. Valid values are: {valid}") from e
    raise TypeError("resampling method must be one from rasterio.enums.Resampling or a string")


# Helper function for raster auto-alignment
def align_raster_to_target(
    ds: EEORasterDataset, target: EEORasterDataset, method: str = "bilinear"
) -> EEORasterDataset:
    """Resample dataset to match target raster's shape and transform"""
    if ds.get_shape() == target.get_shape() or ds.get_transform() == target.get_transform():
        return ds
    return ds.resample(size=target.get_shape(), resampling_method=method)


# helper to mask nodata values from an EEORasterDataset
def mask_nodata(ds: EEORasterDataset, array: np.ndarray) -> np.ndarray:
    nodata = ds.get_metadata().get("nodata", None)
    if nodata is not None:
        array = np.where(array == nodata, np.nan, array)
    return array
