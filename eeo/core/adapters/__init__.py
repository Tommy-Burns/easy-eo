"""Backend adapters abstracting NumPy-, rasterio- and xarray-backed rasters."""

from .base import BaseRasterAdapter
from .numpy import NumpyRasterioAdapter
from .rasterio import RasterioAdapter

# Importing the module does not import xarray; the adapter does so on use.
from .xarray import XarrayAdapter

__all__ = [
    "BaseRasterAdapter",
    "RasterioAdapter",
    "NumpyRasterioAdapter",
    "XarrayAdapter",
]
