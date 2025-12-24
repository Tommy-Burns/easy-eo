from __future__ import annotations

from typing import Optional

import numpy as np
import rasterio as rio
from rasterio.io import DatasetReader, MemoryFile

from .base import BaseRasterAdapter


class RasterioAdapter(BaseRasterAdapter):
    """
    Rasterio-backend adapter for EEORasterDataset
    """
    def __init__(
            self,
            dataset: DatasetReader,
            *,
            memory_file: MemoryFile | None = None,
    ) -> None:
        self._ds = dataset
        self._memory_file = memory_file

    # ========================
    # Factories
    # ========================
    @classmethod
    def from_path(cls, path: str) -> "RasterioAdapter":
        try:
            dataset = rio.open(path)
        except Exception as e:
            raise RuntimeError(f"Failed to open raster: {path}") from e
        return cls(dataset)

    @classmethod
    def from_array(
            cls,
            array: np.ndarray,
            *,
            transform,
            crs,
            nodata: Optional[float] = None,
            dtype: Optional[str] = None,
    ) -> "RasterioAdapter":
        """
        Create an in-memory adapter from a NumPy array
        """
        if array.ndim == 2:
            array = array[np.newaxis, ...]

        count, height, width = array.shape
        memfile = MemoryFile()
        dataset = memfile.open(
            driver="GTiff",
            height=height,
            width=width,
            count=count,
            transform=transform,
            crs=crs,
            nodata=nodata,
            dtype=dtype or array.dtype,
        )
        dataset.write(array)
        return cls(dataset, memory_file=memfile)

    # ========================
    # Metadata
    # ========================
    def get_crs(self):
        return self._ds.crs

    def get_transform(self):
        return self._ds.transform

    def get_bounds(self):
        return self._ds.bounds

    def get_shape(self):
        return self._ds.shape

    def get_width(self):
        return self._ds.width

    def get_height(self):
        return self._ds.height

    def get_count(self):
        return self._ds.count

    def get_nodata(self):
        return self._ds.nodata

    def get_metadata(self):
        return self._ds.meta.copy()

    # ========================
    # Data Access
    # ========================
    def read(self, *args, **kwargs) -> np.ndarray:
        return self._ds.read(*args, **kwargs)

    def read_band(self, idx: int) -> np.ndarray:
        if idx < 1 or idx > self._ds.count:
            raise IndexError(f"Band index {idx} out of range")
        return self._ds.read(idx)

    # ========================
    # Persistence
    # ========================
    def write(self, path: str, driver: str = "GTiff") -> None:
        meta = self._ds.meta.copy()
        meta.update(driver=driver)

        with rio.open(path, "w", **meta) as dst:
            for i in range(1, self._ds.count + 1):
                dst.write(self._ds.read(i), i)

    def close(self) -> None:
        try:
            self._ds.close()
        finally:
            if self._memory_file is not None:
                self._memory_file.close()

    # ========================
    # Backend Access
    # ========================
    @property
    def backend(self) -> DatasetReader:
        return self._ds
