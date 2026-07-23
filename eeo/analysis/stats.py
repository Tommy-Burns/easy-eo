"""Per-pixel statistics and coordinate sampling."""

import numpy as np

from eeo.common import get_nodata, mask_nodata
from eeo.core.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_op
from eeo.core.exceptions import ValidationError

Coordinate = tuple[float, float] | list[float]


@eeo_raster_op
def extract_value_at_coordinate(
    ds: EEORasterDataset, coordinates: Coordinate, band_idx: int | str = 1
) -> int | float:
    """Sample a single pixel value at a world coordinate.

    Parameters
    ----------
    ds : EEORasterDataset
        Raster to sample. NumPy-backed inputs are promoted to rasterio.
    coordinates : tuple of float or list of float
        ``(x, y)`` position in the raster's CRS units. Must contain exactly
        two values and fall within the raster extent.
    band_idx : int or str, default 1
        Band to sample, as a 1-based index or a band name.

    Returns
    -------
    int or float
        The pixel value at ``coordinates`` for the selected band, in the
        band's own dtype. If the sampled pixel is nodata — equal to the
        raster's declared nodata value, or already NaN — ``float('nan')`` is
        returned instead of the raw sentinel, so a nodata fill is never
        mistaken for a real measurement.

    Raises
    ------
    IndexError
        If ``band_idx`` is an index outside the range of available bands.
    ValidationError
        If ``coordinates`` does not contain exactly two values, or
        ``band_idx`` is a name that is unknown or matches more than one band.

    Notes
    -----
    Reads the selected band into memory. Coordinates are ``(x, y)`` in CRS
    units, distinct from the ``(row, col)`` pixel indexing used elsewhere.

    Examples
    --------
    >>> value = ds.extract_value_at_coordinate((500000.0, 4200000.0))
    """
    if len(coordinates) != 2:
        raise ValidationError(
            f"coordinates must contain exactly 2 values (x, y); got {len(coordinates)}"
        )

    # No-op when the dataset is already rasterio-backed
    ds = ds.to_rasterio()
    backend = ds._adapter.backend

    x, y = coordinates
    # rasterio's DatasetReader.index returns ints on 1.5+ but floats on 1.4,
    # so coerce before indexing to stay correct across the supported range.
    row, col = backend.index(x, y)
    row, col = int(row), int(col)

    value = ds.get_band(band_idx)[row, col]

    # Report nodata as NaN rather than the raw pixel value, so a fill value that
    # sits near real measurements is never mistaken for one.
    nodata = get_nodata(ds)
    if nodata is not None and value == nodata:
        return float("nan")
    if np.issubdtype(value.dtype, np.floating) and np.isnan(value):
        return float("nan")

    return value


@eeo_raster_op
def get_maximum_pixel(
    ds: EEORasterDataset,
    band_idx: int | str = 1,
    *,
    return_position_as_pixel_coordinate: bool = False,
) -> dict:
    """Find the maximum pixel value in a band and its location.

    Parameters
    ----------
    ds : EEORasterDataset
        Input raster dataset.
    band_idx : int or str, default 1
        Band to analyse, as a 1-based index or a band name. The default
        selects the first band, which for a single-band raster is its only
        band; pass a band number or name to analyse a different band of a
        multi-band raster.
    return_position_as_pixel_coordinate : bool, default False
        If True, return the position as ``(row, col)`` pixel indices;
        otherwise as ``(x, y)`` world coordinates in the raster's CRS.

    Returns
    -------
    dict
        ``{"value": float, "position": tuple}`` — the maximum value and where
        it occurs. Nodata pixels are excluded from the search.

    Raises
    ------
    IndexError
        If ``band_idx`` is an index outside the range of available bands.
    ValidationError
        If ``band_idx`` is a name that is unknown or matches more than one
        band.

    Notes
    -----
    Reads the band into memory. Nodata pixels are masked to NaN and ignored.

    Examples
    --------
    >>> peak = ds.get_maximum_pixel()
    >>> peak["value"], peak["position"]
    """
    band = mask_nodata(ds, ds.get_band(band_idx))

    # get max value
    value = float(np.nanmax(band))
    row, col = np.unravel_index(np.nanargmax(band), band.shape)

    if return_position_as_pixel_coordinate:
        position = (row, col)
    else:
        transform = ds.get_transform()
        position = transform * (col, row)

    return {"value": value, "position": position}


@eeo_raster_op
def get_minimum_pixel(
    ds: EEORasterDataset,
    band_idx: int | str = 1,
    *,
    return_position_as_pixel_coordinate: bool = False,
) -> dict:
    """Find the minimum pixel value in a band and its location.

    Parameters
    ----------
    ds : EEORasterDataset
        Input raster dataset.
    band_idx : int or str, default 1
        Band to analyse, as a 1-based index or a band name. The default
        selects the first band, which for a single-band raster is its only
        band; pass a band number or name to analyse a different band of a
        multi-band raster.
    return_position_as_pixel_coordinate : bool, default False
        If True, return the position as ``(row, col)`` pixel indices;
        otherwise as ``(x, y)`` world coordinates in the raster's CRS.

    Returns
    -------
    dict
        ``{"value": float, "position": tuple}`` — the minimum value and where
        it occurs. Nodata pixels are excluded from the search.

    Raises
    ------
    IndexError
        If ``band_idx`` is an index outside the range of available bands.
    ValidationError
        If ``band_idx`` is a name that is unknown or matches more than one
        band.

    Notes
    -----
    Reads the band into memory. Nodata pixels are masked to NaN and ignored.

    Examples
    --------
    >>> low = ds.get_minimum_pixel()
    >>> low["value"], low["position"]
    """
    band = mask_nodata(ds, ds.get_band(band_idx))

    # get min value
    value = float(np.nanmin(band))
    row, col = np.unravel_index(np.nanargmin(band), band.shape)

    if return_position_as_pixel_coordinate:
        position = (row, col)
    else:
        transform = ds.get_transform()
        position = transform * (col, row)

    return {"value": value, "position": position}


@eeo_raster_op
def get_mean_pixel(
    ds: EEORasterDataset,
    band_idx: int | str = 1,
    *,
    return_position_as_pixel_coordinate: bool = False,
) -> dict:
    """Compute a band's mean and locate the pixel closest to it.

    Parameters
    ----------
    ds : EEORasterDataset
        Input raster dataset.
    band_idx : int or str, default 1
        Band to analyse, as a 1-based index or a band name. The default
        selects the first band, which for a single-band raster is its only
        band; pass a band number or name to analyse a different band of a
        multi-band raster.
    return_position_as_pixel_coordinate : bool, default False
        If True, return the position as ``(row, col)`` pixel indices;
        otherwise as ``(x, y)`` world coordinates in the raster's CRS.

    Returns
    -------
    dict
        ``{"value": float, "position": tuple}`` — ``value`` is the band mean
        (nodata excluded), and ``position`` locates the pixel whose value is
        nearest that mean.

    Raises
    ------
    IndexError
        If ``band_idx`` is an index outside the range of available bands.
    ValidationError
        If ``band_idx`` is a name that is unknown or matches more than one
        band.

    Notes
    -----
    Reads the band into memory. Nodata pixels are masked to NaN and ignored.

    Examples
    --------
    >>> centre = ds.get_mean_pixel()
    >>> centre["value"], centre["position"]
    """
    band = mask_nodata(ds, ds.get_band(band_idx))

    mean_value = float(np.nanmean(band))
    diff = np.abs(band - mean_value)

    row, col = np.unravel_index(np.nanargmin(diff), diff.shape)

    if return_position_as_pixel_coordinate:
        position = (row, col)
    else:
        transform = ds.get_transform()
        position = transform * (col, row)

    return {"value": mean_value, "position": position}


@eeo_raster_op
def get_percentile_pixel(
    ds: EEORasterDataset,
    percentile: float,
    band_idx: int | str = 1,
    *,
    return_position_as_pixel_coordinate: bool = False,
) -> dict:
    """Compute a band percentile and locate the pixel closest to it.

    Parameters
    ----------
    ds : EEORasterDataset
        Input raster dataset.
    percentile : float
        Percentile to compute, in the range ``[0, 100]``.
    band_idx : int or str, default 1
        Band to analyse, as a 1-based index or a band name. The default
        selects the first band, which for a single-band raster is its only
        band; pass a band number or name to analyse a different band of a
        multi-band raster.
    return_position_as_pixel_coordinate : bool, default False
        If True, return the position as ``(row, col)`` pixel indices;
        otherwise as ``(x, y)`` world coordinates in the raster's CRS.

    Returns
    -------
    dict
        ``{"value": float, "position": tuple}`` — ``value`` is the requested
        percentile of the band (nodata excluded), and ``position`` locates
        the pixel whose value is nearest that percentile.

    Raises
    ------
    IndexError
        If ``band_idx`` is an index outside the range of available bands.
    ValidationError
        If ``band_idx`` is a name that is unknown or matches more than one
        band.

    Notes
    -----
    Reads the band into memory. Nodata pixels are masked to NaN and ignored.

    Examples
    --------
    >>> p95 = ds.get_percentile_pixel(95)
    >>> p95["value"], p95["position"]
    """
    band = mask_nodata(ds, ds.get_band(band_idx))

    perc_value = float(np.nanpercentile(band, percentile))
    diff = np.abs(band - perc_value)

    row, col = np.unravel_index(np.nanargmin(diff), diff.shape)

    if return_position_as_pixel_coordinate:
        position = (row, col)
    else:
        transform = ds.get_transform()
        position = transform * (col, row)

    return {"value": perc_value, "position": position}
