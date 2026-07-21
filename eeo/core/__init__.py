"""Core dataset class, loaders, decorators, and backend adapters."""

from .core import EEORasterDataset
from .loader import load_array, load_raster
from .plugins import load_ops

load_ops()

__all__ = [
    "EEORasterDataset",
    "load_raster",
    "load_array",
]
