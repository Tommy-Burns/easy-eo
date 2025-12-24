"""
Core functionalities for easy-eo
"""
from __future__ import annotations

from typing import Union

import numpy as np
import rasterio as rio
from rasterio import CRS
from rasterio.transform import Affine

from eeo.core.adapters import NumpyRasterioAdapter
from eeo.core.adapters import RasterioAdapter, BaseRasterAdapter


# IO helper
def _save_raster(dataset: rio.DatasetReader, path: str, driver: str = "GTiff") -> None:
    profile = dataset.profile.copy()
    if driver != "GTiff":
        profile.update(driver=driver)
    with rio.open(path, mode="w", **profile) as dst:
        dst.write(dataset.read())


# Core class
class EEORasterDataset:
    def __init__(self, adapter: BaseRasterAdapter, path: str | None = None):
        self._adapter = adapter
        self.path = path

    # ========================
    # Constructors
    # ========================
    @classmethod
    def from_path(cls, path: str) -> "EEORasterDataset":
        adapter = RasterioAdapter.from_path(path)
        return cls(adapter=adapter, path=path)

    @classmethod
    def from_rasterio(cls, dataset: rio.DatasetReader) -> "EEORasterDataset":
        return cls(adapter=RasterioAdapter(dataset))

    @classmethod
    def from_array(
            cls,
            array: np.ndarray,
            transform: Affine,
            crs: Union[CRS, str, int],
            driver: str = "GTiff",
            nodata=None,
    ) -> "EEORasterDataset":
        adapter = NumpyRasterioAdapter(
            array=array,
            transform=transform,
            crs=crs,
            nodata=nodata,
            driver=driver,
        )
        return cls(adapter=adapter)

    # ========================
    # Conversion between adapters
    # ========================

    def to_rasterio(self) -> "EEORasterDataset":
        backend = self._adapter.backend

        # already a rasterio backend
        if isinstance(backend, rio.DatasetReader):
            return self

        array = self.read()
        transform = self.get_transform()
        crs = self.get_crs()
        nodata = self._adapter.get_nodata()

        adapter = RasterioAdapter.from_array(
            array=array,
            transform=transform,
            crs=crs,
            nodata=nodata,
        )
        return EEORasterDataset(adapter=adapter)

    def to_array(self) -> np.ndarray:
        return self.read()

    # ========================
    # Metadata
    # ========================

    def read(self, *args, **kwargs):
        """Forward rasterio.read"""
        return self._adapter.read(*args, **kwargs)

    def get_crs(self):
        return self._adapter.get_crs()

    def get_transform(self):
        return self._adapter.get_transform()

    def get_shape(self):
        return self._adapter.get_shape()

    def get_bounds(self):
        return self._adapter.get_bounds()

    def get_metadata(self):
        return self._adapter.get_metadata()

    def get_width(self):
        return self._adapter.get_width()

    def get_height(self):
        return self._adapter.get_height()

    def get_count(self):
        return self._adapter.get_count()

    def get_index(self):
        return self.ds.index

    def get_band(self, idx: int) -> np.ndarray:
        return self._adapter.read_band(idx)

    # ========================
    # Saving
    # ========================
    def save_raster(self, path: str, driver: str="GTiff") -> None:
        self._adapter.write(path=path, driver=driver)

    # ========================
    # Lifecycle
    # ========================
    def close(self) -> None:
        self._adapter.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ========================
    # Constructors
    # ========================
    def _bind(self, func):
        """Helper that wraps external functions as bound methods"""
        def method(*args, **kwargs):
            result = func(*args, **kwargs)
            return self if result is None else result
        return method

    # ========================
    # Adapter access
    # ========================
    @property
    def ds(self):
        return self._adapter.backend


    # ========================
    # Arithmetic Operators
    # ========================
    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.subtract(other)

    def __rsub__(self, other):
        # implement for only raster - scalar
        if isinstance(other, (int, float)):
            return self.multiply(-1).add(other)
        return NotImplemented

    def __mul__(self, other):
        return self.multiply(other)

    def __rmul__(self, other):
        return self.multiply(other)

    def __truediv__(self, other):
        return self.divide(other)

    def __rtruediv__(self, other):
        # implement scalar / raster
        if isinstance(other, (int, float)):
            return self.power(-1).multiply(other)
        return NotImplemented

    def __pow__(self, exponent):
        return self.power(exponent)

