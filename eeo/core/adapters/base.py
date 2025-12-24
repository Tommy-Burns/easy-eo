from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.transform import Affine


class BaseRasterAdapter(ABC):
    ###########################
    # METADATA
    ##########################
    @abstractmethod
    def get_crs(self) -> CRS:
        ...

    @abstractmethod
    def get_transform(self) -> Affine:
        ...

    @abstractmethod
    def get_bounds(self) -> BoundingBox:
        ...

    @abstractmethod
    def get_shape(self) -> Tuple[int, int]:
        ...

    @abstractmethod
    def get_width(self) -> int:
        ...

    @abstractmethod
    def get_height(self) -> int:
        ...

    @abstractmethod
    def get_count(self) -> int:
        ...

    @abstractmethod
    def get_nodata(self) -> Optional[float]:
        ...

    @abstractmethod
    def get_metadata(self) -> dict[Any, Any]:
        ...

    ###########################
    # DATA ACCESS
    ##########################

    @abstractmethod
    def read(self, *args, **kwargs) -> np.ndarray:
        """
        Read the entire raster as a NumPy array
        For multiband rasters, returns an array of shape (bands, height, width)
        """
        ...

    @abstractmethod
    def read_band(self, idx: int) -> np.ndarray:
        """
        Read a single band (1-based index).
        """
        ...

    ###########################
    # Persistence
    ##########################
    @abstractmethod
    def write(self, path: str, driver: str = "GTiff") -> None:
        ...


    @abstractmethod
    def close(self) -> None:
        ...

    ###########################
    # BACKEND ACCESS - RETURNING THE UNDERLYING DATASET
    ##########################
    @property
    @abstractmethod
    def backend(self) -> Any:
        """
        Return the underlying backend object (rasterio.DatasetReader or numpy.ndarray)

        WARNING: This bypasses Easy-EO abstractions
        """
        ...
