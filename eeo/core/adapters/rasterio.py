"""Rasterio-backed raster adapter."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import rasterio as rio
from rasterio.io import DatasetReader, DatasetWriter, MemoryFile

from eeo.core.exceptions import BackendError
from eeo.core.remote import open_env
from eeo.core.types import StrPath

from .base import BaseRasterAdapter


class RasterioAdapter(BaseRasterAdapter):
    """Rasterio-backed raster adapter for EEORasterDataset.

    Prefer the factories — :meth:`from_path`, :meth:`from_array`,
    :meth:`write_in_memory` and :meth:`from_memory_file` — over calling the
    constructor directly.

    Parameters
    ----------
    dataset : rasterio.io.DatasetReader
        Open rasterio dataset to wrap. Usually a reader; an in-memory result is
        always one, since the library reopens finished rasters read-only.
    memory_file : rasterio.io.MemoryFile or None, default None
        The in-memory file ``dataset`` was opened from, if any. The adapter
        takes ownership: it keeps the file alive for as long as the dataset
        reads from it, and closes it in :meth:`close`.
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
    def from_path(cls, path: StrPath) -> RasterioAdapter:
        try:
            with open_env(path):
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
        """Create an in-memory adapter from a NumPy array.

        This is how every scene loader and ``to_rasterio()`` produce their
        result, so it goes through :meth:`write_in_memory` and hands back a
        read-only raster rather than an open writer.

        Parameters
        ----------
        array : numpy.ndarray
            Pixels shaped ``(bands, height, width)``, or ``(height, width)``
            for a single band.
        transform : affine.Affine
            Affine geotransform mapping pixel to world coordinates.
        crs : rasterio.crs.CRS or str or int
            Coordinate reference system.
        nodata : float or int or None, default None
            Value marking nodata pixels, or None if the array declares none.
        dtype : str or None, default None
            Dtype to store the raster in; defaults to the array's own.

        Returns
        -------
        RasterioAdapter
            Adapter over a read-only in-memory GeoTIFF holding ``array``.
        """
        if array.ndim == 2:
            array = array[np.newaxis, ...]

        count, height, width = array.shape
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": count,
            "transform": transform,
            "crs": crs,
            "nodata": nodata,
            "dtype": dtype or array.dtype,
        }
        return cls.write_in_memory(profile, lambda dst: dst.write(array))

    @classmethod
    def write_in_memory(
        cls, profile: dict, fill: Callable[[DatasetWriter], object]
    ) -> RasterioAdapter:
        """Write a new in-memory raster and return it opened read-only.

        ``fill`` receives the open writer and writes the pixels. The writer is
        then closed and the same in-memory file reopened for reading, which is
        the whole point of this helper: every in-memory raster the library
        produces goes through here or :meth:`from_memory_file`.

        Why not simply return the writer: closing it flushes its freshly
        written blocks out of GDAL's block cache. Left open, they stay in the
        cache *dirty*, and each operation in a chain adds another raster's
        worth. Once that exceeds the cache — 5% of RAM by default, so about
        400 MB on an 8 GB laptop — GDAL must flush dirty blocks one band at a
        time to make room, and for a pixel-interleaved GeoTIFF that path is
        catastrophically slow. Measured on a real 6-band, 5490x5490 Sentinel-2
        stack with default settings: a three-operation chain took 0.7 s,
        124 s and 354 s per step with writers left open, against about a
        second each once results were flushed.

        Parameters
        ----------
        profile : dict
            Creation options for the raster (driver, dtype, shape, crs, ...).
        fill : callable
            Called once with the open writer; writes the raster's pixels.

        Returns
        -------
        RasterioAdapter
            Adapter over the finished raster, opened read-only, owning its
            ``MemoryFile``.
        """
        memfile = MemoryFile()
        try:
            with memfile.open(**profile) as dst:
                fill(dst)
        except Exception:
            memfile.close()
            raise
        return cls.from_memory_file(memfile)

    @classmethod
    def from_memory_file(cls, memfile: MemoryFile) -> RasterioAdapter:
        """Open a fully written ``MemoryFile`` read-only and adopt it.

        For writers that cannot hand their work to :meth:`write_in_memory`
        in one callback — the block-wise engine writes one window at a time —
        but must still end the same way: writer closed, raster reopened
        read-only. See :meth:`write_in_memory` for why.

        Parameters
        ----------
        memfile : rasterio.io.MemoryFile
            In-memory file whose writer has already been closed.

        Returns
        -------
        RasterioAdapter
            Adapter over the raster, opened read-only, owning ``memfile``.
        """
        return cls(memfile.open(), memory_file=memfile)

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
        self, path: StrPath, driver: str = "GTiff", band_names: list[str | None] | None = None
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
