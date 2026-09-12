"""Normalization operations: min-max, percentile, and standardize.

All three need a statistic over every pixel before they can rescale any of
them, so all three make two passes: one streaming reduction to measure the
raster (:mod:`eeo.core.streaming`), then one streaming pass through the
block-wise engine to apply it. Memory stays bounded by the block in both,
except where :func:`~eeo.core.streaming.valid_percentiles` documents that it
cannot be.
"""

import numpy as np

from eeo.common import mask_nodata
from eeo.core.blockwise import BlockSource, apply_blockwise
from eeo.core.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_op
from eeo.core.exceptions import ValidationError
from eeo.core.streaming import valid_mean_std, valid_min_max, valid_percentiles


def _rescale_blockwise(ds: EEORasterDataset, rescale) -> EEORasterDataset:
    """Stream ``rescale`` over ``ds``, masking nodata before it is applied.

    The arithmetic runs on the nodata-masked block, so a sentinel never enters
    it; the engine then writes NaN back over those pixels and records
    ``nodata=nan``, or no nodata at all when the input declared none.

    The block is cast to float64 first, deliberately. ``mask_nodata`` promotes
    to float64 only when a nodata value is declared — it substitutes NaN, which
    is a Python float — so without the cast the same operation would rescale a
    float32 raster in float32 or in float64 depending on nothing more than
    whether it happened to declare nodata. The output is float32 either way;
    this only decides how much precision the intermediate arithmetic keeps.
    """
    return apply_blockwise(
        ds,
        lambda block: rescale(mask_nodata(ds, block).astype(np.float64)),
        sources=[BlockSource.from_dataset(ds)],
        fractional=True,
    )


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
    Streams block-wise, reading every pixel twice: the mean and deviation are
    not knowable until the whole raster has been seen, so one pass measures
    them and a second standardizes against them. Memory stays bounded by the
    block in both. Blocks are combined with Chan's parallel update, so the
    variance does not lose precision on a scene of a hundred million pixels.

    Examples
    --------
    >>> z = ds.standardize()
    """
    ds = ds.to_rasterio()
    mean_value, std_value = valid_mean_std(ds)

    def rescale(masked):
        with np.errstate(divide="ignore", invalid="ignore"):
            return (masked - mean_value) / std_value

    return _rescale_blockwise(ds, rescale)


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
    old_min, old_max = valid_min_max(ds)

    def rescale(masked):
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = (masked - old_min) / (old_max - old_min)
        return normalized * (new_max - new_min) + new_min

    return _rescale_blockwise(ds, rescale)


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
    ValidationError
        If either percentile falls outside ``[0, 100]``, or if
        ``lower_percentile >= upper_percentile``.

    Notes
    -----
    Streams block-wise, reading every pixel twice: the thresholds are not
    knowable until the whole raster has been seen, so one pass measures them
    and a second rescales against them.

    The measuring pass is itself streamed for integer rasters — every raw
    Sentinel-2 and Landsat band, and so every raster large enough for this to
    matter — from an exact histogram, one counter per distinct value. Floating
    -point rasters read the band instead, because an exact percentile of
    floating-point data cannot be computed in bounded memory; see
    :func:`eeo.core.streaming.valid_percentiles`. Either way the thresholds
    equal what ``numpy.percentile`` gives over the valid pixels.

    Examples
    --------
    >>> ds = load_array(np.random.rand(64, 64), crs=4326)
    >>> out = ds.normalize_percentile(lower_percentile=5, upper_percentile=95)
    """
    for name, value in (
        ("lower_percentile", lower_percentile),
        ("upper_percentile", upper_percentile),
    ):
        if not 0 <= value <= 100:
            raise ValidationError(f"{name} must be in the range [0, 100]; got {value}")
    if lower_percentile >= upper_percentile:
        raise ValidationError(
            "lower_percentile must be below upper_percentile; got "
            f"{lower_percentile} and {upper_percentile}"
        )

    ds = ds.to_rasterio()
    array_min, array_max = valid_percentiles(ds, (lower_percentile, upper_percentile))

    def rescale(masked):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.clip((masked - array_min) / (array_max - array_min), 0, 1)

    return _rescale_blockwise(ds, rescale)
