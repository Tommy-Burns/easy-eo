"""Rasterio-backed raster adapter."""

from __future__ import annotations

import numpy as np
import rasterio as rio
from rasterio.io import DatasetReader, MemoryFile

from eeo.core.exceptions import BackendError

from .base import BaseRasterAdapter


class RasterioAdapter(BaseRasterAdapter):
    """Rasterio-backed raster adapter for EEORasterDataset."""

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
    def from_path(cls, path: str) -> RasterioAdapter:
        try:
            dataset = rio.open(path)
        except Exception as e:
            raise BackendError(f"failed to open raster: {path}") from e
        return cls(dataset)

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        *,
        transform,
        crs,
        nodata: float | None = None,
        dtype: str | None = None,
    ) -> RasterioAdapter:
        """Create an in-memory adapter from a NumPy array."""
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

    def get_band_descriptions(self) -> list[str | None]:
        # rasterio exposes GDAL band descriptions as a length-count tuple with
        # None for unnamed bands; normalise blank strings to None too.
        descriptions = self._ds.descriptions or (None,) * self._ds.count
        return [(d or None) for d in descriptions]

    # ========================
    # Data Access
    # ========================
    def read(self, *args, **kwargs) -> np.ndarray:
        return self._ds.read(*args, **kwargs)

    def read_band(self, idx: int) -> np.ndarray:
        if idx < 1 or idx > self._ds.count:
            raise IndexError(
                f"band index {idx} out of range; dataset has {self._ds.count} "
                f"band(s) (valid 1..{self._ds.count})"
            )
        return self._ds.read(idx)

    # ========================
    # Persistence
    # ========================
    def write(
        self, path: str, driver: str = "GTiff", band_names: list[str | None] | None = None
    ) -> None:
        meta = self._ds.meta.copy()
        meta.update(driver=driver)

        with rio.open(path, "w", **meta) as dst:
            for i in range(1, self._ds.count + 1):
                dst.write(self._ds.read(i), i)
            # Flush the in-memory names to GDAL band descriptions; unnamed
            # bands are left alone so the file records no description at all.
            for i, name in enumerate(band_names or [], start=1):
                if name:
                    dst.set_band_description(i, name)

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
