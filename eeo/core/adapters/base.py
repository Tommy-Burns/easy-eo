"""Backend adapter interface for EEORasterDataset."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.transform import Affine


class BaseRasterAdapter(ABC):
    """Backend-agnostic interface that every raster backend must implement.

    Concrete adapters (NumPy-backed, rasterio-backed, and future lazy
    backends) implement these methods so that operations in ``eeo`` stay
    backend-agnostic. Metadata accessors return rasterio/affine types
    regardless of the underlying backend.
    """

    ###########################
    # METADATA
    ##########################
    @abstractmethod
    def get_crs(self) -> CRS:
        """Return the coordinate reference system."""
        ...

    @abstractmethod
    def get_transform(self) -> Affine:
        """Return the affine transform mapping pixel to world coordinates."""
        ...

    @abstractmethod
    def get_bounds(self) -> BoundingBox:
        """Return the spatial bounds as ``(left, bottom, right, top)``."""
        ...

    @abstractmethod
    def get_shape(self) -> tuple[int, int]:
        """Return the raster shape as ``(height, width)`` in pixels."""
        ...

    @abstractmethod
    def get_width(self) -> int:
        """Return the raster width in pixels."""
        ...

    @abstractmethod
    def get_height(self) -> int:
        """Return the raster height in pixels."""
        ...

    @abstractmethod
    def get_count(self) -> int:
        """Return the number of bands."""
        ...

    @abstractmethod
    def get_nodata(self) -> float | None:
        """Return the nodata value, or ``None`` if unset."""
        ...

    @abstractmethod
    def get_metadata(self) -> dict[Any, Any]:
        """Return the raster profile (dtype, nodata, transform, crs, ...)."""
        ...

    ###########################
    # DATA ACCESS
    ##########################

    @abstractmethod
    def read(self, *args, **kwargs) -> np.ndarray:
        """Read the raster as an array of shape ``(bands, height, width)``."""
        ...

    @abstractmethod
    def read_band(self, idx: int) -> np.ndarray:
        """Read a single band by its 1-based index."""
        ...

    ###########################
    # Persistence
    ##########################
    @abstractmethod
    def write(self, path: str, driver: str = "GTiff") -> None:
        """Write the raster to ``path`` using the given GDAL driver."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the backend."""
        ...

    ###########################
    # BACKEND ACCESS - RETURNING THE UNDERLYING DATASET
    ##########################
    @property
    @abstractmethod
    def backend(self) -> Any:
        """Return the underlying backend object.

        Returns the raw ``rasterio.DatasetReader`` or ``numpy.ndarray``.
        This bypasses Easy-EO's abstractions; use it only for interop.
        """
        ...
