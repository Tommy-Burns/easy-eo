"""Per-pixel statistics and coordinate sampling."""

import numpy as np

from eeo.common import mask_nodata
from eeo.core.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_op
from eeo.core.exceptions import ValidationError

Coordinate = tuple[float, float] | list[float]


@eeo_raster_op
def extract_value_at_coordinate(
    ds: EEORasterDataset, coordinates: Coordinate, band_idx: int = 1
) -> int | float:
    """Sample a single pixel value at a world coordinate.

    Parameters
    ----------
    ds : EEORasterDataset
        Raster to sample. NumPy-backed inputs are promoted to rasterio.
    coordinates : tuple of float or list of float
        ``(x, y)`` position in the raster's CRS units. Must contain exactly
        two values and fall within the raster extent.
    band_idx : int, default 1
        1-based band to sample.

    Returns
    -------
    int or float
        The pixel value at ``coordinates`` for the selected band, in the
        band's own dtype. The value is returned as-is with no nodata
        handling, so sampling a nodata pixel returns the sentinel value.

    Raises
    ------
    ValidationError
        If ``coordinates`` does not contain exactly two values.

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

    return ds.get_band(band_idx)[row, col]


@eeo_raster_op
def get_maximum_pixel(
    ds: EEORasterDataset,
    band_idx: int = 1,
    *,
    return_position_as_pixel_coordinate: bool = False,
) -> dict:
    """Find the maximum pixel value in a band and its location.

    Parameters
    ----------
    ds : EEORasterDataset
        Input raster dataset.
    band_idx : int, default 1
        1-based band to analyse.
    return_position_as_pixel_coordinate : bool, default False
        If True, return the position as ``(row, col)`` pixel indices;
        otherwise as ``(x, y)`` world coordinates in the raster's CRS.

    Returns
    -------
    dict
        ``{"value": float, "position": tuple}`` — the maximum value and where
        it occurs. Nodata pixels are excluded from the search.

    Notes
    -----
    Reads the band into memory. Nodata pixels are masked to NaN and ignored.

    Examples
    --------
    >>> peak = ds.get_maximum_pixel()
    >>> peak["value"], peak["position"]
    """
    band = ds.read() if ds.get_count() == 1 else ds.get_band(band_idx)
    band = mask_nodata(ds, band)

    # get max value
    value = float(np.nanmax(band))
    _, row, col = np.unravel_index(np.nanargmax(band), band.shape)

    if return_position_as_pixel_coordinate:
        position = (row, col)
    else:
        transform = ds.get_transform()
        position = transform * (col, row)

    return {"value": value, "position": position}


@eeo_raster_op
def get_minimum_pixel(
    ds: EEORasterDataset,
    band_idx: int = 1,
    *,
    return_position_as_pixel_coordinate: bool = False,
) -> dict:
    """Find the minimum pixel value in a band and its location.

    Parameters
    ----------
    ds : EEORasterDataset
        Input raster dataset.
    band_idx : int, default 1
        1-based band to analyse.
    return_position_as_pixel_coordinate : bool, default False
        If True, return the position as ``(row, col)`` pixel indices;
        otherwise as ``(x, y)`` world coordinates in the raster's CRS.

    Returns
    -------
    dict
        ``{"value": float, "position": tuple}`` — the minimum value and where
        it occurs. Nodata pixels are excluded from the search.

    Notes
    -----
    Reads the band into memory. Nodata pixels are masked to NaN and ignored.

    Examples
    --------
    >>> low = ds.get_minimum_pixel()
    >>> low["value"], low["position"]
    """
    band = ds.read() if ds.get_count() == 1 else ds.get_band(band_idx)
    band = mask_nodata(ds, band)

    # get min value
    value = float(np.nanmin(band))
    _, row, col = np.unravel_index(np.nanargmin(band), band.shape)

    if return_position_as_pixel_coordinate:
        position = (row, col)
    else:
        transform = ds.get_transform()
        position = transform * (col, row)

    return {"value": value, "position": position}


@eeo_raster_op
def get_mean_pixel(
    ds: EEORasterDataset,
    band_idx: int = 1,
    *,
    return_position_as_pixel_coordinate: bool = False,
) -> dict:
    """Compute a band's mean and locate the pixel closest to it.

    Parameters
    ----------
    ds : EEORasterDataset
        Input raster dataset.
    band_idx : int, default 1
        1-based band to analyse.
    return_position_as_pixel_coordinate : bool, default False
        If True, return the position as ``(row, col)`` pixel indices;
        otherwise as ``(x, y)`` world coordinates in the raster's CRS.

    Returns
    -------
    dict
        ``{"value": float, "position": tuple}`` — ``value`` is the band mean
        (nodata excluded), and ``position`` locates the pixel whose value is
        nearest that mean.

    Notes
    -----
    Reads the band into memory. Nodata pixels are masked to NaN and ignored.

    Examples
    --------
    >>> centre = ds.get_mean_pixel()
    >>> centre["value"], centre["position"]
    """
    band = ds.read() if ds.get_count() == 1 else ds.get_band(band_idx)
    band = mask_nodata(ds, band)

    mean_value = float(np.nanmean(band))
    diff = np.abs(band - mean_value)

    _, row, col = np.unravel_index(np.nanargmin(diff), diff.shape)

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
    band_idx: int = 1,
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
    band_idx : int, default 1
        1-based band to analyse.
    return_position_as_pixel_coordinate : bool, default False
        If True, return the position as ``(row, col)`` pixel indices;
        otherwise as ``(x, y)`` world coordinates in the raster's CRS.

    Returns
    -------
    dict
        ``{"value": float, "position": tuple}`` — ``value`` is the requested
        percentile of the band (nodata excluded), and ``position`` locates
        the pixel whose value is nearest that percentile.

    Notes
    -----
    Reads the band into memory. Nodata pixels are masked to NaN and ignored.

    Examples
    --------
    >>> p95 = ds.get_percentile_pixel(95)
    >>> p95["value"], p95["position"]
    """
    band = ds.read() if ds.get_count() == 1 else ds.get_band(band_idx)
    band = mask_nodata(ds, band)

    perc_value = float(np.nanpercentile(band, percentile))
    diff = np.abs(band - perc_value)

    _, row, col = np.unravel_index(np.nanargmin(diff), band.shape)

    if return_position_as_pixel_coordinate:
        position = (row, col)
    else:
        transform = ds.get_transform()
        position = transform * (col, row)

    return {"value": perc_value, "position": position}
