"""Streaming reductions over a raster's blocks.

:mod:`eeo.core.blockwise` streams a *transformation* — one block in, one block
out. This module streams a *reduction*: the whole raster is read a window at a
time and collapsed to a few numbers, so a statistic can be taken over a scene
that does not fit in memory.

Operations needing a global statistic are two-pass by nature — the statistic
cannot be known until every pixel has been seen, and cannot be applied until
the statistic is known. Nothing here hides that; the passes are explicit, and
each op's docstring says how many it makes.

One thing is deliberately *not* streamed. An exact percentile of floating-point
data cannot be had in bounded memory: unlike a minimum or a mean it is not a
running accumulation, and unlike integer data there is no finite set of values
to count. Integer rasters — which is every raw Sentinel-2 and Landsat band, and
so every raster large enough for this to matter — get an exact histogram and
stream. Floating-point rasters fall back to reading the band, which is what
they did before, and :func:`valid_percentiles` says so in its docstring rather
than quietly returning an approximation.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from eeo.common import get_nodata
from eeo.core.blockwise import block_windows, resolve_block_shape
from eeo.core.core import EEORasterDataset

#: Largest integer histogram this module will build, in bins. One bin per
#: distinct integer value, so 2^21 covers every raster whose values span up to
#: about two million — far more than the 65,536 a uint16 band can hold. A
#: wider integer range falls back to reading the band, as floats do.
MAX_HISTOGRAM_BINS = 1 << 21


def valid_mask(block: np.ndarray, nodata: float | None) -> np.ndarray:
    """Return a boolean mask of the pixels a statistic should count.

    A pixel is invalid if it equals the declared nodata value or, in floating
    data, if it is NaN. The NaN half matters for agreement with the whole-array
    form: ``numpy.nanmin`` and friends skip every NaN, not only the ones that
    came from a declared sentinel, so a statistic that counted a stray NaN
    would disagree with the answer this library used to give.

    Parameters
    ----------
    block : numpy.ndarray
        Pixels to classify.
    nodata : float or int or None
        The raster's declared nodata value, or None if it declares none.

    Returns
    -------
    numpy.ndarray
        Boolean array, True where the pixel counts.
    """
    invalid = None
    if nodata is not None:
        if isinstance(nodata, float) and math.isnan(nodata):
            invalid = np.isnan(block)
        else:
            invalid = block == nodata
    if np.issubdtype(block.dtype, np.floating):
        not_a_number = np.isnan(block)
        invalid = not_a_number if invalid is None else (invalid | not_a_number)
    if invalid is None:
        return np.ones(block.shape, dtype=bool)
    return ~invalid


def stream_windows(
    ds: EEORasterDataset,
    band: int | None = None,
    *,
    block_shape: tuple[int, int] | None = None,
) -> Iterator[tuple[Any, np.ndarray, np.ndarray]]:
    """Iterate a raster block by block, with each block's validity mask.

    This is the single place a reduction decides how to cut a raster into
    windows, so patching :func:`resolve_block_shape` here is enough to change
    every reduction's blocking at once.

    Parameters
    ----------
    ds : EEORasterDataset
        Raster to read. Promoted to the rasterio backend, since the NumPy
        adapter ignores a window and would return the whole array per block.
    band : int or None, default None
        1-based band index to read, or None to read every band.
    block_shape : tuple of int or None, default None
        Block shape as ``(height, width)``; defaults to the engine's own.

    Yields
    ------
    tuple
        ``(window, block, valid)`` — the window, its pixels in the source
        dtype, and the boolean mask of pixels a statistic should count.
    """
    ds = ds.to_rasterio()
    nodata = get_nodata(ds)
    shape = ds.get_shape()
    for window in block_windows(shape, block_shape or resolve_block_shape(shape)):
        block = ds.read(window=window) if band is None else ds.read(band, window=window)
        yield window, block, valid_mask(block, nodata)


def valid_min_max(ds: EEORasterDataset, band: int | None = None) -> tuple[float, float]:
    """Return the minimum and maximum over a raster's valid pixels.

    One streaming pass. Blocks holding no valid pixel are skipped rather than
    reduced — not an optimisation: reducing an empty selection raises, and on a
    partly-filled scene most edge blocks are exactly that.

    Parameters
    ----------
    ds : EEORasterDataset
        Raster to measure.
    band : int or None, default None
        1-based band index, or None for every band.

    Returns
    -------
    tuple of float
        ``(minimum, maximum)``, or ``(nan, nan)`` if no pixel is valid — which
        carries through a rescaling to an all-NaN result, the same answer the
        whole-array form gives.
    """
    low, high = np.inf, -np.inf
    seen = False
    for _window, block, valid in stream_windows(ds, band):
        if not valid.any():
            continue
        values = block[valid]
        seen = True
        low = min(low, float(values.min()))
        high = max(high, float(values.max()))
    if not seen:
        return float("nan"), float("nan")
    return low, high


def valid_mean_std(ds: EEORasterDataset, band: int | None = None) -> tuple[float, float]:
    """Return the mean and population standard deviation of the valid pixels.

    One streaming pass. Blocks are combined with Chan's parallel update rather
    than by accumulating a running sum of squares: over a hundred million
    pixels of four-digit reflectance the sum of squares reaches ~1e16, where
    float64 has already run out of significant digits and the variance starts
    losing precision. Chan's form carries a mean and a centred second moment
    per block instead, so nothing large is ever subtracted from anything large.

    Parameters
    ----------
    ds : EEORasterDataset
        Raster to measure.
    band : int or None, default None
        1-based band index, or None for every band.

    Returns
    -------
    tuple of float
        ``(mean, std)`` over valid pixels, with ``std`` the population
        deviation (``ddof=0``, matching ``numpy.nanstd``), or ``(nan, nan)``
        if no pixel is valid.
    """
    count = 0
    mean = 0.0
    moment2 = 0.0
    for _window, block, valid in stream_windows(ds, band):
        if not valid.any():
            continue
        values = block[valid].astype(np.float64)
        block_count = values.size
        block_mean = float(values.mean())
        block_moment2 = float(((values - block_mean) ** 2).sum())

        total = count + block_count
        delta = block_mean - mean
        mean += delta * block_count / total
        moment2 += block_moment2 + delta**2 * count * block_count / total
        count = total

    if count == 0:
        return float("nan"), float("nan")
    return mean, math.sqrt(moment2 / count)


def _integer_histogram(ds, band, low, high) -> np.ndarray | None:
    """Count every distinct integer value in ``[low, high]``, or None if too wide.

    Returns one count per integer, so the result describes the data exactly —
    no binning, and therefore no approximation in any order statistic read off
    it.
    """
    span = int(high) - int(low) + 1
    if span > MAX_HISTOGRAM_BINS:
        return None
    counts = np.zeros(span, dtype=np.int64)
    for _window, block, valid in stream_windows(ds, band):
        if not valid.any():
            continue
        offsets = block[valid].astype(np.int64) - int(low)
        counts += np.bincount(offsets, minlength=span)
    return counts


def _order_statistic(cumulative: np.ndarray, low: int, rank: int) -> float:
    """Return the ``rank``-th smallest value (0-based) described by a histogram."""
    index = int(np.searchsorted(cumulative, rank, side="right"))
    return float(low + index)


def valid_percentiles(
    ds: EEORasterDataset, quantiles: Sequence[float], band: int | None = None
) -> list[float]:
    """Return percentiles of a raster's valid pixels.

    Integer rasters are measured exactly from a streaming histogram — one
    counter per distinct value, so no binning and no approximation — in two
    passes: one for the value range, one to count. This is the case that
    matters, since every raw Sentinel-2 and Landsat band is integer, and so is
    every raster big enough for the memory to be a problem.

    Floating-point rasters read the band instead. An exact percentile of
    floating-point data is not a running accumulation like a minimum or a mean,
    and has no finite set of values to count, so it cannot be had in bounded
    memory; returning an approximation without saying so would be worse than
    reading the array. An integer raster whose values span more than
    ``MAX_HISTOGRAM_BINS`` falls back the same way.

    Parameters
    ----------
    ds : EEORasterDataset
        Raster to measure.
    quantiles : sequence of float
        Percentiles to compute, each in ``[0, 100]``.
    band : int or None, default None
        1-based band index, or None for every band.

    Returns
    -------
    list of float
        One value per requested percentile, interpolated linearly between
        neighbouring order statistics exactly as ``numpy.percentile`` does, so
        the streamed answer equals the whole-array answer rather than merely
        approximating it.

    Notes
    -----
    Makes two passes over an integer raster and holds one counter per distinct
    value (at most ``MAX_HISTOGRAM_BINS``). Reads the band into memory for
    floating-point data — see above.
    """
    dtype = np.dtype(ds.get_metadata()["dtype"])
    counts = None
    low = float("nan")
    if np.issubdtype(dtype, np.integer):
        low, high = valid_min_max(ds, band)
        if not math.isnan(low):
            counts = _integer_histogram(ds, band, low, high)

    if counts is None:
        return _percentiles_by_reading(ds, quantiles, band)

    # At least one pixel is counted here: the histogram is only built when the
    # range pass found a valid pixel, and it counts the same pixels.
    total = int(counts.sum())
    cumulative = np.cumsum(counts)

    results = []
    for quantile in quantiles:
        # numpy's default "linear" method: an index into the sorted values
        # that may land between two of them, then interpolate.
        virtual = quantile / 100.0 * (total - 1)
        lower_rank = math.floor(virtual)
        upper_rank = math.ceil(virtual)
        lower = _order_statistic(cumulative, int(low), lower_rank)
        if upper_rank == lower_rank:
            results.append(lower)
            continue
        upper = _order_statistic(cumulative, int(low), upper_rank)
        results.append(lower + (virtual - lower_rank) * (upper - lower))
    return results


def _percentiles_by_reading(ds, quantiles, band) -> list[float]:
    """Percentiles from the whole array, for data no histogram can describe."""
    ds = ds.to_rasterio()
    block = ds.read() if band is None else ds.read(band)
    valid = valid_mask(block, get_nodata(ds))
    if not valid.any():
        return [float("nan")] * len(quantiles)
    # float64, because that is what the whole-array form ended up using: its
    # nodata masking went through np.where(..., np.nan, ...), which promotes.
    # Interpolating in float32 instead shifts the thresholds in the last few
    # digits and the stretch visibly follows.
    values = block[valid].astype(np.float64)
    return [float(value) for value in np.percentile(values, list(quantiles))]
