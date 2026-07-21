"""Normalization operations: min-max, percentile, and standardize."""

import numpy as np
import rasterio as rio

from eeo.common import get_nodata, mask_nodata
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
    Reads the full array into memory and makes one statistics pass before
    writing, rather than streaming block-wise.

    Examples
    --------
    >>> scaled = ds.normalize_min_max()
    >>> centred = ds.normalize_min_max(new_min=-1, new_max=1)
    """
    ds_nodata = get_nodata(ds)
    masked = mask_nodata(ds, ds.read())

    old_min, old_max = np.nanmin(masked), np.nanmax(masked)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = (masked - old_min) / (old_max - old_min)
    normalized = normalized * (new_max - new_min) + new_min

    out_nodata = float("nan") if ds_nodata is not None else None
    return _write_normalized(ds, normalized.astype(np.float32), out_nodata)


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
