"""Interoperate with the xarray ecosystem.

Easy-EO and xarray describe the same raster differently: an
:class:`~eeo.core.core.EEORasterDataset` wraps a raster and its
georeferencing, while xarray describes a labelled array and rioxarray carries
the georeferencing alongside it. This module translates between the two, so
data can be handed to the xarray/dask ecosystem and taken back.

The translation is a boundary crossing, not a backend: the pixels are
materialised in memory and the result is an ordinary
:class:`xarray.DataArray` with no further link to the dataset it came from. A
lazy, chunk-backed adapter is a separate feature.

The layout produced here follows :func:`rioxarray.open_rasterio`, so the
result is shaped exactly like a DataArray rioxarray opened itself:
dimensions ``("band", "y", "x")``, a 1-based ``band`` coordinate, pixel-centre
``y``/``x`` coordinates, and the CRS, geotransform, and nodata value written
through the ``.rio`` accessor.

This module needs the optional ``xarray`` extra::

    pip install "easy-eo[xarray]"
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any

import numpy as np

from eeo._optional import import_optional
from eeo.common import get_nodata, is_rasterio_backed

if TYPE_CHECKING:
    from eeo.core.core import EEORasterDataset

# DataArray attributes filled from the dataset's own metadata. A same-named
# entry in ``ds.attrs`` never overrides one of these, so georeferencing
# metadata cannot be shadowed by a free-form tag.
_MANAGED_ATTRS = frozenset({"_FillValue", "long_name", "grid_mapping"})

_PURPOSE = "xarray interop"


def _import_xarray() -> Any:
    """Import xarray, plus rioxarray for the side effect of registering ``.rio``."""
    xr = import_optional("xarray", extra="xarray", purpose=_PURPOSE)
    # rioxarray is imported purely for its side effect: importing it registers
    # the ``.rio`` accessor that carries CRS, transform, and nodata.
    import_optional("rioxarray", extra="xarray", purpose=_PURPOSE)
    return xr


def _pixel_centre_coords(transform, height: int, width: int) -> dict[str, Any]:
    """Build pixel-centre coordinates for a geotransform, as rioxarray does.

    An axis-aligned grid gets 1-D ``y``/``x`` dimension coordinates. A rotated
    or sheared grid cannot be described by 1-D axes, so it gets 2-D ``yc``/``xc``
    coordinates over ``("y", "x")`` instead — matching
    :func:`rioxarray.open_rasterio`.
    """
    cols = np.arange(width, dtype="float64") + 0.5
    rows = np.arange(height, dtype="float64") + 0.5

    if transform.b == 0 and transform.d == 0:
        return {
            "y": transform.f + transform.e * rows,
            "x": transform.c + transform.a * cols,
        }

    col_grid, row_grid = np.meshgrid(cols, rows)
    return {
        "yc": (("y", "x"), transform.f + transform.d * col_grid + transform.e * row_grid),
        "xc": (("y", "x"), transform.c + transform.a * col_grid + transform.b * row_grid),
    }


def _long_name(band_names: list[str | None]) -> str | tuple[str | None, ...] | None:
    """Render band names as rioxarray's ``long_name`` attribute, or None if unnamed.

    rioxarray uses a plain string for a single-band raster and a tuple (with
    ``None`` for an unnamed band) otherwise.
    """
    if not any(band_names):
        return None
    if len(band_names) == 1:
        return band_names[0]
    return tuple(band_names)


def _naive_utc(stamp: dt.datetime) -> dt.datetime:
    """Return a timestamp as naive UTC, since xarray datetimes carry no timezone."""
    if stamp.tzinfo is None:
        return stamp
    return stamp.astimezone(dt.timezone.utc).replace(tzinfo=None)


# Range representable by ``datetime64[ns]`` (1678-2262), conservatively bounded.
_NS_MIN = np.datetime64("1678-01-01", "us")
_NS_MAX = np.datetime64("2262-01-01", "us")


def _as_datetime64(stamp: dt.datetime) -> np.datetime64:
    """Convert a timestamp to the datetime64 unit xarray handles best.

    Microseconds match :class:`datetime.datetime`'s own resolution, so the
    conversion is always lossless. Values inside the nanosecond range are
    narrowed to ``datetime64[ns]``, because xarray before 2025 converts
    anything coarser and warns while doing it. A date outside that range keeps
    microseconds rather than being narrowed, since narrowing it would silently
    wrap to the wrong year.
    """
    value = np.datetime64(_naive_utc(stamp), "us")
    if _NS_MIN <= value <= _NS_MAX:
        return value.astype("datetime64[ns]")
    return value


def _independent_array(ds: EEORasterDataset) -> Any:
    """Read the pixels as an array that does not alias the dataset's own buffer.

    A rasterio read already returns a fresh array. The NumPy backend hands out
    the array it holds, so that one is copied — otherwise writing into the
    DataArray would reach back into the dataset.
    """
    array = ds.read()
    if is_rasterio_backed(ds):
        return array
    return array.copy()


def _to_dataarray(ds: EEORasterDataset) -> Any:
    """Build a georeferenced :class:`xarray.DataArray` from a dataset.

    Implements :meth:`eeo.core.core.EEORasterDataset.to_xarray`, which is the
    documented entry point; see its docstring for the full contract.
    """
    xr = _import_xarray()

    height, width = ds.get_shape()
    transform = ds.get_transform()

    coords: dict[str, Any] = {"band": np.arange(1, ds.get_count() + 1)}
    coords.update(_pixel_centre_coords(transform, height, width))
    if ds.timestamp is not None:
        coords["time"] = _as_datetime64(ds.timestamp)

    attrs = {key: value for key, value in ds.attrs.items() if key not in _MANAGED_ATTRS}
    long_name = _long_name(ds.band_names)
    if long_name is not None:
        attrs["long_name"] = long_name

    da = xr.DataArray(_independent_array(ds), dims=("band", "y", "x"), coords=coords, attrs=attrs)

    crs = ds.get_crs()
    if crs is not None:
        da = da.rio.write_crs(crs)
    # Written explicitly rather than left to be inferred from the coordinates,
    # so the affine round-trips exactly — including a rotated transform, whose
    # rotation terms the 2-D coordinates alone would not recover.
    da = da.rio.write_transform(transform)

    nodata = get_nodata(ds)
    if nodata is not None:
        da = da.rio.write_nodata(nodata)

    return da
