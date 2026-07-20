"""
Easy-EO
A lightweight Earth Observation utilities library for chainable raster operations
"""

from .analysis import *
from .core import load_array, load_raster
from .core.adapters import *
from .ops import *
from .preprocessing import *
from .viz import *

__all__ = [
    "load_raster",
    "load_array",
]

__version__ = "0.1.0b1"
