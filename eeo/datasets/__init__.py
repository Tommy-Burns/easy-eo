"""Curated sample datasets with checksum-verified caching.

The bundled Sentinel-2 / Copernicus-DEM sample lets tutorials and quickstarts
run in minutes without hunting for data. Reach it through
:func:`load_sample_dataset`, whose attributes are the individual files:

>>> from eeo.datasets import load_sample_dataset
>>> from eeo import load_raster
>>> sd = load_sample_dataset()
>>> scene = load_raster(sd.sentinel2_cog_stacked)  # doctest: +SKIP
>>> dem = load_raster(sd.copernicus_dem)           # doctest: +SKIP

Files are cached under ``~/.cache/easy-eo`` (override with ``EEO_DATA_DIR``) and
verified against a checksum shipped inside the package, so a fetch is fast after
the first call and never returns corrupt data. Downloading uses only the Python
standard library - no extra dependency.
"""

from __future__ import annotations

from ._cache import DatasetError, cache_dir
from ._samples import SampleDataset, SamplePath, load_sample_dataset

__all__ = [
    "load_sample_dataset",
    "SampleDataset",
    "SamplePath",
    "cache_dir",
    "DatasetError",
]
