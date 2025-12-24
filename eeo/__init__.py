"""
Easy-EO
A lightweight Earth Observation utilities library for chainable raster operations
"""


from .core import load_raster, load_array
from .analysis import *
from .ops import *
from .viz import *
from .preprocessing import *
from .core.adapters import *


__all__ = [
    "load_raster",
    "load_array",
]

__version__ = "0.1.0b1"
