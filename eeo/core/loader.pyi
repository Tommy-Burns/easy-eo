from typing import Optional

import numpy as np
from rasterio.transform import Affine
from rasterio.crs import CRS
from eeo.core.core import EEORasterDataset


def load_raster(path: str) -> EEORasterDataset:
    """
        Load a raster file from ``path`` as an ``EEORasterDataset``. The file must be a GDAL-readable raster and is opened using ``rasterio.open``.

        :param path: Path to the raster file
        :return: An EEORasterDataset
        :raises FileNotFoundError: If the file does not exist
        :raises RuntimeError: If rasterio cannot open the file
        """
    ...


def load_array(
        array: np.ndarray,
        *,
        transform: Optional[Affine] = None,
        crs: Optional[CRS | int | str] = None,
        nodata: Optional[float | int] = None,
) -> EEORasterDataset:
    """
    Load a NumPy array into an ``EEORasterDataset``.

    :param array: Raster data. Can be either:
                  - 2D array with shape ``(H, W)`` for single-band rasters
                  - 3D array with shape ``(bands, H, W)`` for multi-band rasters
    :param transform: Affine transform describing the pixel-to-world mapping.
                      Defaults to the identity transform.
    :param crs: Coordinate reference system.
    :param nodata: NoData value for the raster.

    :return: In-memory raster dataset.
    """
    ...