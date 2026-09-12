"""Normalization operations: min-max, percentile, and standardize."""

import numpy as np
import rasterio as rio

from eeo.common import get_nodata, mask_nodata
from eeo.core.blockwise import BlockSource, apply_blockwise, block_windows, resolve_block_shape
from eeo.core.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_op


def _write_normalized(ds: EEORasterDataset, out: np.ndarray, out_nodata) -> EEORasterDataset:
    """Write a float32 normalization result sharing ``ds``'s georeferencing."""
    meta = ds.get_metadata()
    meta.update(dtype="float32", nodata=out_nodata)
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(out)
    return EEORasterDataset.from_rasterio(out_ds)


def _valid_min_max(ds: EEORasterDataset) -> tuple[float, float]:
    """Return the minimum and maximum over ``ds``'s valid pixels, block by block.

    The whole-array equivalent is ``nanmin``/``nanmax`` over the nodata-masked
    raster, but taken a window at a time so a scene never has to be resident to
    be measured. Blocks holding no valid pixel are skipped rather than reduced,
    which is not merely an optimisation: ``nanmin`` over an all-nodata block
    warns and returns NaN, and on a partly-filled scene most edge blocks are
    exactly that. A raster with no valid pixel anywhere yields ``(nan, nan)``,
    which carries through the rescaling to an all-NaN result — the same answer
    the whole-array form gives, without the warning.
    """
    shape = ds.get_shape()
    low, high = np.inf, -np.inf
    seen = False
    for window in block_windows(shape, resolve_block_shape(shape)):
        # mask_nodata returns float64 whenever a nodata value is declared, so
        # the integer branch below is only reached when none is — and there
        # every pixel is valid by definition.
        masked = mask_nodata(ds, ds.read(window=window))
        if np.issubdtype(masked.dtype, np.floating):
            if not (~np.isnan(masked)).any():
                continue
            block_low, block_high = float(np.nanmin(masked)), float(np.nanmax(masked))
        else:
            block_low, block_high = float(masked.min()), float(masked.max())
        seen = True
        low, high = min(low, block_low), max(high, block_high)

    if not seen:
        return float("nan"), float("nan")
    return low, high


@eeo_raster_op
def standardize(ds: EEORasterDataset) -> EEORasterDataset:
    """Standardize a raster to zero mean and unit variance (z-score).

    Computes ``(x - mean) / std`` over the valid pixels.

    Parameters
    ----------
    ds : EEORasterDataset
        Input raster dataset.

    Returns
    -------
    EEORasterDataset
        New dataset in float32. The mean and standard deviation are computed
        over valid pixels only (nodata excluded), and nodata pixels are NaN in
        the output (``nodata=nan``); a raster with no declared nodata produces
        output with no nodata.

    Notes
    -----
    Reads the full array into memory and makes one statistics pass before
    writing, rather than streaming block-wise.

    Examples
    --------
    >>> z = ds.standardize()
    """
    ds_nodata = get_nodata(ds)
    masked = mask_nodata(ds, ds.read())

    mean_value = np.nanmean(masked)
    std_value = np.nanstd(masked)
    with np.errstate(divide="ignore", invalid="ignore"):
        standardized = (masked - mean_value) / std_value

    out_nodata = float("nan") if ds_nodata is not None else None
    return _write_normalized(ds, standardized.astype(np.float32), out_nodata)


@eeo_raster_op
def normalize_min_max(
    ds: EEORasterDataset, *, new_min: float | int = 0.0, new_max: float | int = 1.0
) -> EEORasterDataset:
    """Linearly rescale a raster to a new value range.

    Maps the raster's valid data range onto ``[new_min, new_max]``.

    Parameters
    ----------
    ds : EEORasterDataset
        Input raster dataset.
    new_min : float or int, default 0.0
        Lower bound of the output range.
    new_max : float or int, default 1.0
        Upper bound of the output range.

    Returns
    -------
    EEORasterDataset
        New dataset in float32 scaled to ``[new_min, new_max]``. The data
        minimum and maximum are computed over valid pixels only (nodata
        excluded), and nodata pixels are NaN in the output (``nodata=nan``); a
        raster with no declared nodata produces output with no nodata.

    Notes
    -----
    Streams block-wise, but reads every pixel twice: the data minimum and
    maximum are not knowable until the whole raster has been seen, so one pass
    measures the range and a second rescales against it. Memory stays bounded
    by the block in both.

    Examples
    --------
    >>> scaled = ds.normalize_min_max()
    >>> centred = ds.normalize_min_max(new_min=-1, new_max=1)
    """
    ds = ds.to_rasterio()
    old_min, old_max = _valid_min_max(ds)

    def rescale(block):
        masked = mask_nodata(ds, block)
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = (masked - old_min) / (old_max - old_min)
        return normalized * (new_max - new_min) + new_min

    return apply_blockwise(
        ds,
        rescale,
        sources=[BlockSource.from_dataset(ds)],
        fractional=True,
    )


@eeo_raster_op
def normalize_percentile(
    ds: EEORasterDataset,
    *,
    lower_percentile: float | int = 2,
    upper_percentile: float | int = 98,
) -> EEORasterDataset:
    """Normalize raster values using percentile thresholds.

    Values outside the percentile range are clipped; remaining values are
    scaled to [0, 1]. Robust to outliers compared to min-max normalization.

    Parameters
    ----------
    ds : EEORasterDataset
        Input raster dataset.
    lower_percentile : float, default 2
        Lower percentile threshold (0-100).
    upper_percentile : float, default 98
        Upper percentile threshold (0-100).

    Returns
    -------
    EEORasterDataset
        New dataset in float32 with values in [0, 1]. Percentiles are computed
        over valid pixels only (nodata excluded), and nodata pixels are NaN in
        the output (``nodata=nan``); a raster with no declared nodata produces
        output with no nodata.

    Raises
    ------
    ValueError
        If ``lower_percentile >= upper_percentile``, propagated from NumPy.

    Notes
    -----
    Reads the full array into memory and makes one statistics pass before
    writing, rather than streaming block-wise. Percentiles are computed with
    ``numpy.nanpercentile`` over the nodata-masked array.

    Examples
    --------
    >>> ds = load_array(np.random.rand(64, 64), crs=4326)
    >>> out = ds.normalize_percentile(lower_percentile=5, upper_percentile=95)
    """
    ds_nodata = get_nodata(ds)
    masked = mask_nodata(ds, ds.read())

    array_min, array_max = np.nanpercentile(masked, (lower_percentile, upper_percentile))
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.clip((masked - array_min) / (array_max - array_min), 0, 1)

    out_nodata = float("nan") if ds_nodata is not None else None
    return _write_normalized(ds, normalized.astype(np.float32), out_nodata)
