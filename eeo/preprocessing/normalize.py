import numpy as np
import rasterio as rio

from eeo.core.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_op


@eeo_raster_op
def standardize(ds: EEORasterDataset) -> EEORasterDataset:
    """Standardize a raster to zero mean and unit variance (z-score).

    Computes ``(x - mean) / std`` over all pixels.

    Parameters
    ----------
    ds : EEORasterDataset
        Input raster dataset.

    Returns
    -------
    EEORasterDataset
        New dataset in the same dtype as ``ds``; results are truncated for
        integer inputs, so use a floating dtype to keep the standardized
        values. The nodata value is carried over unchanged.

    Notes
    -----
    Reads the full array into memory rather than streaming block-wise. The
    mean and standard deviation are computed over every pixel, including
    nodata: nodata pixels are not masked and skew the statistics.

    Examples
    --------
    >>> z = ds.standardize()
    """
    data = ds.read()
    mean_value = np.mean(data)
    std_value = np.std(data)
    standardized_data = (data - mean_value) / std_value
    meta = ds.get_metadata()
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(standardized_data)
    return EEORasterDataset.from_rasterio(out_ds)


@eeo_raster_op
def normalize_min_max(
    ds: EEORasterDataset, *, new_min: float | int = 0.0, new_max: float | int = 1.0
) -> EEORasterDataset:
    """Linearly rescale a raster to a new value range.

    Maps the raster's full data range onto ``[new_min, new_max]``.

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
        New dataset scaled to ``[new_min, new_max]``, in the same dtype as
        ``ds``; results are truncated for integer inputs, so use a floating
        dtype to keep the scaling. The nodata value is carried over unchanged.

    Notes
    -----
    Reads the full array into memory rather than streaming block-wise. The
    data minimum and maximum are computed over every pixel, including nodata:
    nodata pixels are not masked and an extreme sentinel (e.g. -9999) skews
    the rescaling.

    Examples
    --------
    >>> scaled = ds.normalize_min_max()
    >>> centred = ds.normalize_min_max(new_min=-1, new_max=1)
    """
    data = ds.read()
    old_min, old_max = np.min(data), np.max(data)
    normalized_data = (data - old_min) / (old_max - old_min)
    normalized_data = normalized_data * (new_max - new_min) + new_min
    meta = ds.get_metadata()
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(normalized_data)
    return EEORasterDataset.from_rasterio(out_ds)


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
        New dataset with values in [0, 1], in the same dtype as the input.
        Fractional results are truncated for integer inputs (e.g.
        uint8/uint16), so use a floating dtype to keep the [0, 1] scaling.

    Raises
    ------
    ValueError
        If ``lower_percentile >= upper_percentile``, propagated from NumPy.

    Notes
    -----
    Reads the full array into memory rather than streaming block-wise.
    Percentiles are computed with ``numpy.nanpercentile``, which ignores NaN
    but not the dataset's ``nodata`` sentinel, so nodata pixels currently
    take part in the percentile computation.

    Examples
    --------
    >>> ds = load_array(np.random.rand(64, 64), crs=4326)
    >>> out = ds.normalize_percentile(lower_percentile=5, upper_percentile=95)
    """
    data = ds.read()
    array_min, array_max = np.nanpercentile(data, (lower_percentile, upper_percentile))
    normalized_data = np.clip((data - array_min) / (array_max - array_min), 0, 1)
    meta = ds.get_metadata()
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(normalized_data)
    return EEORasterDataset.from_rasterio(out_ds)
