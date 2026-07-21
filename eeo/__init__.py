"""Easy-EO: chainable raster processing for Earth Observation.

A lightweight library of chainable raster operations, algebra, and
visualization built on rasterio, NumPy, GeoPandas, and matplotlib.
"""

from ._show_versions import show_versions
from .analysis import *
from .core import (
    AlignmentError,
    BackendError,
    CRSMismatchError,
    EEOError,
    ValidationError,
    load_array,
    load_raster,
)
from .core.adapters import *
from .ops import *
from .preprocessing import *
from .viz import *

__all__ = [
    "load_raster",
    "load_array",
    "show_versions",
    "EEOError",
    "ValidationError",
    "CRSMismatchError",
    "AlignmentError",
    "BackendError",
]

__version__ = "0.1.0b1"
