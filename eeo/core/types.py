"""Shared type aliases for the public API."""

import os
from typing import Literal

#: Anything accepted where a filesystem path is expected: a ``str`` or any
#: object implementing ``os.PathLike`` (``pathlib.Path``, the sample-data
#: handles returned by :func:`eeo.datasets.load_sample_dataset`, ...).
StrPath = str | os.PathLike

# Auto-complete literal strings for rasterio resampling
ResamplingMethod = Literal[
    "nearest",
    "bilinear",
    "cubic",
    "cubic_spline",
    "lanczos",
    "average",
    "mode",
    "max",
    "min",
    "med",
    "q1",
    "q3",
]
