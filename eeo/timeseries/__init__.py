"""Temporal analysis: ordered, timestamped collections of rasters.

Holds :class:`~eeo.timeseries.core.EEOTimeSeries` and the
:func:`~eeo.timeseries.core.time_series` entry point. Importing this package
needs no optional dependency; the ``stac`` extra is required only when a series
is actually built from a catalog search, and the ``lazy`` extra only when
``chunks=`` asks for the dask-chunked backend.
"""

from .binning import TemporalBins
from .core import EEOTimeSeries, time_series

__all__ = ["EEOTimeSeries", "TemporalBins", "time_series"]
