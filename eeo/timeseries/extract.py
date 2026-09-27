"""Sampling a series at a location: the other half of temporal analysis.

Where the reducers collapse time into one raster, this collapses space into one
table — the per-pixel trajectory that answers "what happened here", and the
shape that pandas, matplotlib and every statistics library already understand.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd
from rasterio.transform import rowcol
from rasterio.warp import transform as warp_transform

from eeo.analysis.stats import extract_value_at_coordinate
from eeo.common import resolve_band_index
from eeo.core.exceptions import ValidationError

if TYPE_CHECKING:  # pragma: no cover - types only
    from eeo.timeseries.core import EEOTimeSeries


def _column_labels(series: EEOTimeSeries, positions: Sequence[int]) -> list[str]:
    """Name one column per sampled band, falling back to the band's position.

    A band name is what the caller addressed the band by, so it is what the
    column should be called; an unnamed band still needs a label a DataFrame can
    hold, and its 1-based index is the only other thing it is known by.
    """
    names = series.band_names
    return [names[position] or f"band_{position + 1}" for position in positions]


def _in_series_crs(
    series: EEOTimeSeries, coordinates: Sequence[float], crs: Any
) -> tuple[float, float]:
    """Return the sample point in the series' own CRS, transforming if asked."""
    if len(coordinates) != 2:
        raise ValidationError(
            f"coordinates must contain exactly 2 values (x, y); got {len(coordinates)}"
        )
    x, y = (float(value) for value in coordinates)
    if crs is None:
        return x, y

    target = series.crs
    if target is None:
        raise ValidationError(
            "crs= transforms the point onto the series' CRS, but this series declares "
            "none; pass coordinates in the rasters' own units instead"
        )
    xs, ys = warp_transform(crs, target, [x], [y])
    return float(xs[0]), float(ys[0])


def extract_at(
    series: EEOTimeSeries,
    coordinates: Sequence[float],
    *,
    bands: Sequence[int | str] | None = None,
    crs: Any = None,
) -> pd.DataFrame:
    """Sample one location at every timestep, as a table indexed by time.

    Reads a single pixel per timestep and band, so a trajectory over a season of
    Sentinel-2 tiles costs a few dozen pixels rather than a few gigabytes.

    Parameters
    ----------
    series : EEOTimeSeries
        Series to sample.
    coordinates : sequence of float
        ``(x, y)`` position, in the series' CRS unless ``crs`` says otherwise.
    bands : sequence of (int or str) or None, default None
        Which bands to sample, as 1-based indices or band names; None samples
        every band.
    crs : optional
        CRS the coordinates are given in — anything rasterio accepts, such as
        ``"EPSG:4326"`` or ``4326`` — when that is not the series' own. The
        point is transformed onto the series' CRS before sampling, which saves
        doing it by hand after a catalog search handed back lon/lat.

    Returns
    -------
    pandas.DataFrame
        One row per timestep, indexed by a ``DatetimeIndex`` named ``time``, and
        one float column per sampled band, named after the band (or
        ``band_<n>`` where the band has no name). A pixel that was nodata at a
        timestep is ``NaN`` there rather than its fill value, so a gap reads as a
        gap. ``attrs`` records the sampled point and the CRS it was sampled in.

    Raises
    ------
    ValidationError
        If ``coordinates`` does not hold exactly two values, if the point falls
        outside the series' extent, if ``crs`` is given for a series that
        declares none, or if ``bands`` names a band the series does not have.

    Notes
    -----
    Reads one pixel per timestep and band — never a band, never a scene.

    Examples
    --------
    >>> trajectory = ts.extract_at((11.1, 46.6), crs="EPSG:4326")  # doctest: +SKIP
    >>> trajectory  # doctest: +SKIP
                               B04     B08
    time
    2023-04-12 10:06:21+00:00  1043.0  2870.0
    2023-05-02 10:06:19+00:00   987.0  3211.0

    Several locations are a concat of several calls, which keeps each one's
    index intact:

    >>> import pandas as pd  # doctest: +SKIP
    >>> plots = {"north": (11.1, 46.6), "south": (11.15, 46.55)}  # doctest: +SKIP
    >>> table = pd.concat(  # doctest: +SKIP
    ...     {name: ts.extract_at(point, crs="EPSG:4326") for name, point in plots.items()},
    ...     names=["plot"],
    ... )
    """
    x, y = _in_series_crs(series, coordinates, crs)

    height, width = series.shape
    row, col = rowcol(series.transform, x, y)
    row, col = int(row), int(col)
    if not (0 <= row < height and 0 <= col < width):
        left, bottom, right, top = series.reference.get_bounds()
        raise ValidationError(
            f"({x}, {y}) falls outside the series' extent — x from {left} to {right}, "
            f"y from {bottom} to {top}, in the series' own CRS. Pass crs= if the "
            f"coordinates are in another one"
        )

    reference = series.reference
    if bands is None:
        positions = list(range(series.band_count))
    else:
        positions = [resolve_band_index(reference, band) - 1 for band in bands]
        if not positions:
            raise ValidationError("bands is empty; name at least one band to sample")

    rows = [
        [extract_value_at_coordinate(ds, (x, y), position + 1) for position in positions]
        for ds in series
    ]

    table = pd.DataFrame(
        rows,
        index=pd.DatetimeIndex(series.timestamps, name="time"),
        columns=_column_labels(series, positions),
        dtype="float64",
    )
    table.attrs = {"x": x, "y": y, "crs": str(series.crs)}
    return table
