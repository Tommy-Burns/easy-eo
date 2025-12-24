from typing import Union, Optional

from pyproj import CRS
from rasterio.warp import Resampling

from eeo.core.core import EEORasterDataset
from eeo.core.types import ResamplingMethod


def reproject_raster(
    ds: EEORasterDataset,
    *,
    target_crs: Union[int, str, CRS],
    resampling_method: Union[Resampling, ResamplingMethod] = "nearest",
) -> EEORasterDataset:
    """
    Reproject a raster to a new CRS.


    :param ds: EEORasterDataset
        Input raster dataset.

    :param target_crs: int | str | pyproj.CRS
        Target CRS (EPSG code, proj string, or a CRS object).

    :param resampling_method: rasterio.enums.Resampling | Literal
        The resampling method used during reprojection. May be either a
        `Resampling` enum value or one of the following string literals:
        ``"nearest", "bilinear", "cubic", "cubic_spline", "lanczos", "average",
        "mode", "max", "min", "med", "q1", "q3"``.
        Default is ``nearest``.


    :return: ``EEORasterDataset``.
    """
    ...
