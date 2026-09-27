"""Temporal reducers: collapsing a series of rasters into one.

Every reducer here is an ordinary function taking a time series and returning a
single :class:`~eeo.core.core.EEORasterDataset`;
:class:`~eeo.timeseries.core.EEOTimeSeries` exposes them as methods. They are
plain functions rather than decorator-bound operations because an operation is
bound onto a *dataset* — these take a series, and there is one type they can
belong to.

**Nodata across time is absent, not contagious.** The library's contract makes
nodata contagious when an operation *combines* operands: a pixel missing from
either input is missing from the output. A reducer is not combining operands, it
is computing a statistic, and the contract's first rule is that a statistic
treats nodata as absent. So a pixel clouded at two of five timesteps still gets
a median — of the three that saw it — and only a pixel missing at *every*
timestep comes out as nodata.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import rasterio as rio

from eeo.common import resolve_band_index
from eeo.core.adapters import RasterioAdapter
from eeo.core.blockwise import (
    DEFAULT_BLOCK_PIXELS,
    BlockSource,
    block_windows,
    resolve_block_shape,
)
from eeo.core.core import EEORasterDataset
from eeo.core.exceptions import ValidationError
from eeo.core.streaming import valid_mask
from eeo.core.types import StrPath

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime, types only
    from eeo.timeseries.core import EEOTimeSeries


def _neutral_for(kind: str, dtype: np.dtype) -> Any:
    """Return the value an invalid pixel takes so that ``kind`` ignores it.

    A minimum over a stack must not see a masked pixel, so masked pixels are
    filled with the largest value the dtype can hold, which can never win; a
    maximum fills with the smallest. This keeps the reduction in the input's own
    dtype, where replacing masked pixels with NaN would force it to float.
    """
    if np.issubdtype(dtype, np.floating):
        extreme = np.inf if kind == "min" else -np.inf
    else:
        info = np.iinfo(dtype)
        extreme = info.max if kind == "min" else info.min
    return np.array(extreme, dtype=dtype)


def _reduce_block(
    blocks: Sequence[np.ndarray], nodatas: Sequence[float | None], *, how: str
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce one window across timesteps, ignoring each timestep's nodata.

    Returns the reduced block and a boolean mask of the pixels that had at least
    one valid timestep. The two are separate because the caller decides how an
    all-invalid pixel is marked, which depends on the output dtype.
    """
    stack = np.stack(blocks)
    valid = np.stack(
        [valid_mask(block, nodata) for block, nodata in zip(blocks, nodatas, strict=True)]
    )
    any_valid = valid.any(axis=0)

    if how in ("min", "max"):
        filled = np.where(valid, stack, _neutral_for(how, stack.dtype))
        reduced = filled.min(axis=0) if how == "min" else filled.max(axis=0)
        return reduced, any_valid

    # mean and median run in float32 and skip NaN. A pixel with no valid
    # timestep at all is filled with zeros first: `numpy.nanmean` of an all-NaN
    # column is a warning and a NaN, and the caller is about to mark that pixel
    # nodata anyway, so there is nothing to learn from letting it happen.
    masked = np.where(valid, stack.astype(np.float32), np.float32(np.nan))
    feedable = np.where(any_valid, masked, np.float32(0))
    reduced = np.nanmedian(feedable, axis=0) if how == "median" else np.nanmean(feedable, axis=0)
    return reduced.astype(np.float32), any_valid


def _shared_attrs(datasets: Sequence[EEORasterDataset]) -> dict:
    """Return the attrs every timestep agrees on.

    A reduction is one raster made from many, so per-scene provenance (the STAC
    item id, the product name) does not describe it and is dropped. What every
    timestep shares does — the mission above all, which is what a later quality
    mask needs to know.
    """
    first, *rest = [ds.attrs for ds in datasets]
    shared = {}
    for key, value in first.items():
        if all(key in other and other[key] == value for other in rest):
            shared[key] = value
    return shared


def reduce_series(
    series: EEOTimeSeries,
    how: str,
    *,
    bands: Sequence[int | str] | None = None,
    save_path: StrPath | None = None,
) -> EEORasterDataset:
    """Reduce every pixel of a series across time, one window at a time.

    The shared implementation behind :func:`median`, :func:`mean`,
    :func:`minimum` and :func:`maximum`.

    Parameters
    ----------
    series : EEOTimeSeries
        Series to collapse. Its timesteps share a grid, which is what lets the
        same window be read from each of them and reduced in step.
    how : {"median", "mean", "min", "max"}
        Statistic to take across timesteps, per pixel and per band.
    bands : sequence of (int or str) or None, default None
        Which bands to reduce, as 1-based indices or band names; None reduces
        every band. A subset is what lets a composite leave the quality band
        out of its output, where a median of scene-class numbers would be
        meaningless.
    save_path : str or path-like or None, default None
        Write the result to this path instead of holding it in memory, so a
        reduction of a large series never has to fit in RAM twice.

    Returns
    -------
    EEORasterDataset
        Single raster on the series' grid with the series' bands and band names,
        holding the statistic over time. ``median`` and ``mean`` are float32
        with NaN for a pixel no timestep saw; ``min`` and ``max`` keep the
        timesteps' own dtype — they select a value rather than computing one —
        and mark such a pixel with the timesteps' nodata value. Carries no
        timestamp, since a reduction is not an acquisition, and records the
        reduction and the time span it covers in ``attrs``.

    Raises
    ------
    ValidationError
        If ``how`` is not one of the four statistics.

    Notes
    -----
    Streams window by window, and the block is scaled down by the number of
    timesteps, so peak memory is about one block's worth in total however long
    the series is — never a raster, let alone the series. Each timestep's own
    nodata value is what masks it, and a pixel is nodata in the result only when
    every timestep is missing it.
    """
    if how not in ("median", "mean", "min", "max"):
        raise ValidationError(f"how must be 'median', 'mean', 'min' or 'max'; got {how!r}")

    reference = series.reference
    if bands is None:
        positions = list(range(series.band_count))
    else:
        positions = [resolve_band_index(reference, band) - 1 for band in bands]
        if not positions:
            raise ValidationError("bands is empty; name at least one band to reduce")

    sources = [BlockSource.from_dataset(ds) for ds in series]
    nodatas = [source.nodata for source in sources]
    fractional = how in ("median", "mean")

    if fractional:
        out_dtype: Any = np.float32
        out_nodata: float | None = float("nan")
    else:
        out_dtype = np.result_type(*(ds.get_metadata()["dtype"] for ds in series))
        declared = next((value for value in nodatas if value is not None), None)
        out_nodata = declared

    meta = reference.get_metadata()
    meta.update(
        driver="GTiff",
        dtype=np.dtype(out_dtype).name,
        nodata=out_nodata,
        count=len(positions),
    )

    # A reduction holds one block of every timestep at once, so the per-block
    # budget is divided by their number: a longer series then reads more, smaller
    # blocks rather than holding more memory. It has to cover the reduction's own
    # working set too, which for a median is several times the stacked block —
    # `numpy.nanmedian` sorts through a masked array, and its int64 index array
    # alone is four times the width of the uint16 pixels it is indexing.
    budget = max(DEFAULT_BLOCK_PIXELS // len(series), 1)
    windows = block_windows(series.shape, resolve_block_shape(series.shape, target_pixels=budget))
    memfile: rio.io.MemoryFile | None = None
    dst: Any = None
    try:
        if save_path is None:
            memfile = rio.io.MemoryFile()
            dst = memfile.open(**meta)
        else:
            dst = rio.open(save_path, "w", **meta)

        for window in windows:
            blocks = [source.read(window)[positions] for source in sources]
            reduced, any_valid = _reduce_block(blocks, nodatas, how=how)
            if out_nodata is not None:
                reduced = np.where(any_valid, reduced, np.array(out_nodata, dtype=out_dtype))
            dst.write(reduced.astype(out_dtype), window=window)
    except Exception:
        # Cleanup only; the error is re-raised. A half-written output left open
        # would keep its blocks dirty in GDAL's cache, and on disk it would
        # leave a truncated raster looking like a finished one.
        if dst is not None:
            dst.close()
        if memfile is not None:
            memfile.close()
        raise

    dst.close()

    attrs = _shared_attrs(list(series))
    attrs["temporal_reduction"] = how
    attrs["timesteps"] = len(series)
    attrs["time_start"] = series.timestamps[0]
    attrs["time_end"] = series.timestamps[-1]

    if save_path is None:
        assert memfile is not None
        result = EEORasterDataset(adapter=RasterioAdapter.from_memory_file(memfile))
    else:
        result = EEORasterDataset.from_path(save_path)
    result.attrs = attrs
    result.band_names = [series.band_names[position] for position in positions]
    return result


def median(
    series: EEOTimeSeries,
    *,
    bands: Sequence[int | str] | None = None,
    save_path: StrPath | None = None,
) -> EEORasterDataset:
    """Median of every pixel across time. See :func:`reduce_series`."""
    return reduce_series(series, "median", bands=bands, save_path=save_path)


def mean(
    series: EEOTimeSeries,
    *,
    bands: Sequence[int | str] | None = None,
    save_path: StrPath | None = None,
) -> EEORasterDataset:
    """Mean of every pixel across time. See :func:`reduce_series`."""
    return reduce_series(series, "mean", bands=bands, save_path=save_path)


def minimum(
    series: EEOTimeSeries,
    *,
    bands: Sequence[int | str] | None = None,
    save_path: StrPath | None = None,
) -> EEORasterDataset:
    """Smallest value of every pixel across time. See :func:`reduce_series`."""
    return reduce_series(series, "min", bands=bands, save_path=save_path)


def maximum(
    series: EEOTimeSeries,
    *,
    bands: Sequence[int | str] | None = None,
    save_path: StrPath | None = None,
) -> EEORasterDataset:
    """Largest value of every pixel across time. See :func:`reduce_series`."""
    return reduce_series(series, "max", bands=bands, save_path=save_path)
