"""Per-pixel statistics and coordinate sampling.

Each of these reports a value *and* where it occurs, which is why they stream
rather than reduce: the value comes from a streaming reduction over the band
(:mod:`eeo.core.streaming`), and a second streaming pass finds the pixel that
carries it. Memory is bounded by the block in both, so a statistic can be taken
over a scene far larger than memory.

Ties are broken exactly as ``numpy.nanargmin`` breaks them — first in
row-major order — regardless of how the raster happens to be cut into blocks,
so a position never depends on the block shape.
"""

import numpy as np
from rasterio.windows import Window

from eeo.common import get_nodata, resolve_band_index
from eeo.core.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_op
from eeo.core.exceptions import ValidationError
from eeo.core.streaming import stream_windows, valid_mean_std, valid_percentiles

Coordinate = tuple[float, float] | list[float]

_NO_VALID_PIXELS = "band {band} has no valid pixels, so {what} is undefined"


def _locate(ds, band_idx, score, what):
    """Stream the band and return ``(value, row, col)`` minimising ``score``.

    ``score`` maps a block of pixels to a block of comparison values; the
    smallest wins. Because a block is a contiguous rectangle, block-local
    row-major order agrees with global row-major order inside it, so taking
    the first minimum per block and then breaking ties on ``(row, col)``
    reproduces ``numpy.nanargmin`` over the whole band for any block shape.

    Returns the *pixel's* value, not its score, so a caller searching for the
    pixel nearest a target gets the measurement rather than the distance.
    """
    band = resolve_band_index(ds, band_idx)
    best = None
    for window, block, valid in stream_windows(ds, band):
        if not valid.any():
            continue
        # Score in float64, never in the block's own dtype: negating a uint8
        # or uint16 block to turn a maximum into a minimum wraps instead of
        # changing sign, which silently returns the *smallest* pixel whenever
        # the band contains a zero.
        scored = np.where(valid, score(block.astype(np.float64)), np.inf)
        flat = int(np.argmin(scored))
        local_row, local_col = np.unravel_index(flat, scored.shape)
        row = int(window.row_off) + int(local_row)
        col = int(window.col_off) + int(local_col)
        candidate = (float(scored[local_row, local_col]), row, col)
        if best is None or candidate < best[:3]:
            best = (*candidate, block[local_row, local_col])
    if best is None:
        raise ValidationError(_NO_VALID_PIXELS.format(band=band_idx, what=what))
    _score, row, col, value = best
    return value, row, col


def _position(ds, row, col, as_pixel_coordinate):
    """Render a pixel location as ``(row, col)`` or as world coordinates."""
    if as_pixel_coordinate:
        return (row, col)
    return ds.get_transform() * (col, row)


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
    Reads a single pixel, not the band: sampling one location never costs the
    scene. Coordinates are ``(x, y)`` in CRS units, distinct from the
    ``(row, col)`` pixel indexing used elsewhere.

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

    # A 1x1 window, so sampling a point in a 10980x10980 scene reads one
    # pixel rather than the 241 MB band around it.
    window = Window(col, row, 1, 1)
    value = ds.read(resolve_band_index(ds, band_idx), window=window)[0, 0]

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
    Streams the band block by block, so memory is bounded by the block rather
    than the band. Nodata pixels are excluded.

    Examples
    --------
    >>> peak = ds.get_maximum_pixel()
    >>> peak["value"], peak["position"]
    """
    ds = ds.to_rasterio()
    # Negating turns the search for a maximum into the same minimisation the
    # other three do, tie-breaking included.
    value, row, col = _locate(ds, band_idx, lambda block: -block, "a maximum")
    return {
        "value": float(value),
        "position": _position(ds, row, col, return_position_as_pixel_coordinate),
    }


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
    Streams the band block by block in two passes — one to measure, one to
    locate — so memory is bounded by the block rather than the band. Nodata
    pixels are excluded from both.

    Examples
    --------
    >>> low = ds.get_minimum_pixel()
    >>> low["value"], low["position"]
    """
    ds = ds.to_rasterio()
    value, row, col = _locate(ds, band_idx, lambda block: block, "a minimum")
    return {
        "value": float(value),
        "position": _position(ds, row, col, return_position_as_pixel_coordinate),
    }


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
    Streams the band block by block in two passes — one to measure, one to
    locate — so memory is bounded by the block rather than the band. Nodata
    pixels are excluded from both.

    Examples
    --------
    >>> centre = ds.get_mean_pixel()
    >>> centre["value"], centre["position"]
    """
    ds = ds.to_rasterio()
    mean_value, _std = valid_mean_std(ds, resolve_band_index(ds, band_idx))
    _value, row, col = _locate(ds, band_idx, lambda block: np.abs(block - mean_value), "a mean")
    return {
        "value": mean_value,
        "position": _position(ds, row, col, return_position_as_pixel_coordinate),
    }


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
    Streams the band block by block in two passes — one to measure, one to
    locate — so memory is bounded by the block rather than the band. Nodata
    pixels are excluded from both.

    Examples
    --------
    >>> p95 = ds.get_percentile_pixel(95)
    >>> p95["value"], p95["position"]
    """
    ds = ds.to_rasterio()
    (perc_value,) = valid_percentiles(ds, (percentile,), resolve_band_index(ds, band_idx))
    _value, row, col = _locate(
        ds, band_idx, lambda block: np.abs(block - perc_value), "a percentile"
    )
    return {
        "value": perc_value,
        "position": _position(ds, row, col, return_position_as_pixel_coordinate),
    }
