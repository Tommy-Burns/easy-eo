"""Backend adapters abstracting NumPy- and rasterio-backed rasters."""

from .base import BaseRasterAdapter
from .numpy import NumpyRasterioAdapter
from .rasterio import RasterioAdapter

__all__ = [
    "BaseRasterAdapter",
    "RasterioAdapter",
    "NumpyRasterioAdapter",
]
