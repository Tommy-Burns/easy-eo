from typing import Optional

import numpy as np
import rasterio as rio
from rasterio.crs import CRS
from rasterio.transform import Affine

from eeo.core.adapters import BaseRasterAdapter, RasterioAdapter


class NumpyRasterioAdapter(BaseRasterAdapter):
    """ Numpy backend adapter for EEORasterDataset """
    def __init__(
        self,
        array: np.ndarray,
        transform: Affine,
        crs: CRS,
        driver: str = "GTiff",
        nodata: Optional[float] = None,
    ):
        if array.ndim == 2:
            array = array[np.newaxis, ...]

        self._array = array
        self._transform = transform
        self._crs = crs
        self._nodata = nodata
        self._driver = driver

    # ========================
    # Metadata
    # ========================
    def get_crs(self) -> CRS:
        return self._crs

    def get_transform(self) -> Affine:
        return self._transform

    def get_bounds(self):
        h, w = self.get_shape()
        return rio.transform.array_bounds(h, w, self._transform)

    def get_shape(self) -> tuple[int, int]:
        return self._array.shape[-2:]

    def get_width(self) -> int:
        return self.get_shape()[1]

    def get_height(self) -> int:
        return self.get_shape()[0]

    def get_count(self) -> int:
        return self._array.shape[0]

    def get_nodata(self) -> Optional[float]:
        return self._nodata

    def get_metadata(self) -> dict:
        return {
            "dtype": self._array.dtype,
            "nodata": self._nodata,
            "transform": self._transform,
            "crs": self._crs,
            "driver": self._driver,
            "count": self.get_count(),
            "width": self.get_width(),
            "height": self.get_height(),
        }

    # ========================
    # Data Access
    # ========================
    def read(self, *args, **kwargs) -> np.ndarray:
        return self._array

    def read_band(self, idx: int) -> np.ndarray:
        if idx < 1 or idx > self.get_count():
            raise IndexError(f"Band index {idx} out of range")
        return self._array[idx - 1]

    # ========================
    # Persistence
    # ========================
    def write(self, path: str, driver: str = "GTiff") -> None:
        adapter = RasterioAdapter.from_array(
            array=self._array,
            transform=self._transform,
            crs=self._crs,
            nodata=self._nodata,
        )
        adapter.write(path, driver=driver)
        adapter.close()

    def close(self) -> None:
        pass

    # ========================
    # Backend Access
    # ========================
    @property
    def backend(self) -> np.ndarray:
        return self._array
