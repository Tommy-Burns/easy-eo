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
from affine import Affine

from eeo._optional import import_optional
from eeo.common import get_nodata, is_rasterio_backed
from eeo.core.exceptions import ValidationError

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


# --------------------------------------------------------------------------
# DataArray -> EEORasterDataset
# --------------------------------------------------------------------------


def _require_dataarray(obj: Any, xr: Any) -> None:
    """Reject anything that is not a DataArray, pointing at the right entry point."""
    if isinstance(obj, xr.DataArray):
        return
    if isinstance(obj, xr.Dataset):
        first = next(iter(obj.data_vars), None)
        hint = "one of its variables" if first is None else f"dataset[{first!r}]"
        raise ValidationError(
            f"from_xarray expects an xarray.DataArray; got a Dataset. Select a variable "
            f"({hint}) or merge them into an array with dataset.to_dataarray()."
        )
    raise ValidationError(
        f"from_xarray expects an xarray.DataArray; got {type(obj).__name__}. "
        "Use eeo.load_array() for a plain NumPy array, or eeo.load_raster() for a file."
    )


def _spatial_dims(da: Any) -> tuple[str, str]:
    """Return the DataArray's ``(y_dim, x_dim)`` names, as rioxarray identifies them."""
    exceptions = import_optional("rioxarray.exceptions", extra="xarray", purpose=_PURPOSE)
    try:
        return da.rio.y_dim, da.rio.x_dim
    except exceptions.DimensionError as err:
        raise ValidationError(
            "could not identify the spatial dimensions of the DataArray; got dimensions "
            f"{tuple(da.dims)}. Rename them to 'y' and 'x', or declare them with "
            "da.rio.set_spatial_dims(x_dim=..., y_dim=...)."
        ) from err


def _as_band_first(da: Any, y_dim: str, x_dim: str) -> Any:
    """Reduce a DataArray to ``(band, y, x)`` order, collapsing spare length-1 dims."""
    extra = [dim for dim in da.dims if dim not in (y_dim, x_dim)]

    if len(extra) > 1:
        # A single scene often arrives with spare length-1 dims (e.g. after
        # expand_dims); those collapse cleanly, anything longer cannot.
        collapsible = {dim: 0 for dim in extra if da.sizes[dim] == 1}
        da = da.isel(collapsible)
        extra = [dim for dim in da.dims if dim not in (y_dim, x_dim)]

    # Checked before the generic complaint below, since it is the more specific
    # mistake: bands and timesteps are different things, and silently loading
    # timesteps as bands would mislabel the data.
    if "time" in extra and da.sizes["time"] > 1:
        raise ValidationError(
            f"a 'time' dimension of length {da.sizes['time']} is a time series, not a band "
            "stack. Select a step with da.isel(time=0), or reduce it with da.mean('time')."
        )

    if len(extra) > 1:
        sizes = ", ".join(f"{dim}={da.sizes[dim]}" for dim in extra)
        raise ValidationError(
            f"a raster has one band dimension besides {y_dim!r} and {x_dim!r}; got {sizes}. "
            "Select or reduce the extra dimension(s) first, e.g. da.isel(time=0)."
        )

    if extra:
        return da.transpose(extra[0], y_dim, x_dim)
    return da.transpose(y_dim, x_dim)


def _axis_step(values: np.ndarray, dim: str) -> float:
    """Return an axis's constant spacing, rejecting an axis that has none."""
    steps = np.diff(values)
    step = float(steps[0])
    if step == 0 or not np.allclose(steps, step, rtol=1e-3, atol=0):
        raise ValidationError(
            f"the {dim!r} coordinate must be evenly spaced to describe a raster grid; got "
            f"spacing between {float(steps.min())} and {float(steps.max())}. Reindex or "
            "interpolate onto a regular grid first."
        )
    return step


def _axis_geometry(da: Any, dim: str) -> tuple[float, float] | None:
    """Return ``(resolution, origin)`` for one axis, or None if its coordinate cannot give it.

    The origin is the outer edge of the first pixel, half a pixel before its
    centre. Needs a 1-D coordinate with at least two values, since a single
    value carries no spacing.
    """
    if dim not in da.coords:
        return None
    values = np.asarray(da[dim].values, dtype="float64")
    if values.ndim != 1 or values.size < 2:
        return None
    step = _axis_step(values, dim)
    return step, float(values[0]) - step / 2


def _transform_from(da: Any, y_dim: str, x_dim: str) -> Affine:
    """Determine the geotransform of a DataArray from its axes, or its stored affine.

    The coordinates are the authority: slicing, sorting, or reversing a
    DataArray moves the pixels while the stored geotransform stays behind, and
    reading that stale affine would misplace the raster. The stored affine is
    used where the coordinates cannot speak — a rotated grid (described by 2-D
    coordinates), a single-pixel axis, or no coordinates at all — and is
    preferred whenever it already agrees with them, so an untouched DataArray
    round-trips exactly.
    """
    stored = da.rio.transform()
    if not stored.is_rectilinear:
        return stored

    x_geometry = _axis_geometry(da, x_dim)
    y_geometry = _axis_geometry(da, y_dim)
    x_res, x_origin = (stored.a, stored.c) if x_geometry is None else x_geometry
    y_res, y_origin = (stored.e, stored.f) if y_geometry is None else y_geometry
    derived = Affine.translation(x_origin, y_origin) * Affine.scale(x_res, y_res)

    if np.allclose(tuple(derived)[:6], tuple(stored)[:6], rtol=1e-9, atol=0):
        return stored
    return derived


def _band_names_from(da: Any, count: int) -> list[str | None] | None:
    """Read band names from rioxarray's ``long_name`` attribute, or None if unusable."""
    long_name = da.attrs.get("long_name")
    if isinstance(long_name, str):
        names: list[Any] = [long_name]
    elif isinstance(long_name, (list, tuple)):
        names = list(long_name)
    else:
        return None

    if len(names) != count:
        # Selecting one band of a stack leaves the whole stack's long_name
        # behind; a list that no longer fits the data would mislabel it.
        return None
    return [name if isinstance(name, str) else None for name in names]


def _timestamp_from(da: Any) -> dt.datetime | None:
    """Read an acquisition time from a scalar ``time`` coordinate, if there is one."""
    coord = da.coords.get("time")
    if coord is None or coord.ndim != 0:
        return None
    value = coord.values
    if not np.issubdtype(value.dtype, np.datetime64):
        return None
    # Microseconds match datetime's own resolution; .item() on NaT gives None.
    return value.astype("datetime64[us]").item()


def from_xarray(da: Any) -> EEORasterDataset:
    """Wrap a georeferenced xarray DataArray as an Easy-EO dataset.

    Reads the CRS, geotransform, and nodata value from rioxarray's ``.rio``
    accessor, so a DataArray from ``rioxarray.open_rasterio`` — or from
    :meth:`~eeo.core.core.EEORasterDataset.to_xarray` — comes back as a fully
    georeferenced dataset ready to chain. Needs the optional ``xarray`` extra
    (``pip install "easy-eo[xarray]"``).

    Parameters
    ----------
    da : xarray.DataArray
        Raster to wrap. Dimensions may be ``(band, y, x)`` in any order, or
        ``(y, x)`` for a single band; the spatial dimensions are the ones
        rioxarray identifies (``y``/``x`` by name, or whatever
        ``da.rio.set_spatial_dims()`` declared), and any remaining dimension is
        the band dimension. Spare length-1 dimensions are collapsed. Pixel
        values are taken as laid out, so the pixel order of the DataArray is
        the pixel order of the raster.

    Returns
    -------
    EEORasterDataset
        NumPy-backed dataset with the DataArray's own values and dtype —
        neither is converted — carrying its CRS, geotransform, and nodata
        value. Nodata pixels keep whatever marks them in the array
        (``NaN`` for a DataArray read with ``mask_and_scale=True``, the
        sentinel otherwise), and ``da.rio.nodata`` is recorded as the dataset's
        nodata so later operations mask on it.

    Raises
    ------
    ValidationError
        If ``da`` is not a DataArray, if its spatial dimensions cannot be
        identified, if more than one non-spatial dimension remains after
        length-1 dimensions are collapsed (including a ``time`` dimension,
        which is a time series rather than a band stack), or if a coordinate
        axis is not evenly spaced and so cannot describe a raster grid.
    MissingDependencyError
        If the ``xarray`` extra is not installed.

    See Also
    --------
    eeo.core.core.EEORasterDataset.to_xarray : The reverse conversion.

    Notes
    -----
    Nothing is copied on top of the DataArray's values, exactly as
    :func:`eeo.load_array` wraps an array as it is, so converting a large scene
    does not double its memory: the dataset wraps whatever buffer the DataArray
    hands over, and a lazily opened or dask-backed one is materialised as it is
    read. Treat the conversion as handing that buffer to Easy-EO — a later write
    to either side may show up on the other — and pass ``da.copy()`` if the two
    must stay fully separate.

    The geotransform comes from the coordinate axes when they can give it, so a
    sliced, sorted, or reversed DataArray is placed where its coordinates
    actually are rather than where its stored geotransform used to be. A
    DataArray whose ``y`` axis ascends therefore yields a south-up raster
    (a positive north-south term) rather than being flipped silently.

    Band names are read from rioxarray's ``long_name`` attribute, and are
    dropped rather than guessed at when the attribute no longer matches the
    band count. A scalar ``time`` coordinate becomes the dataset's
    ``timestamp`` (naive, in UTC, as xarray stores it), and the remaining
    ``attrs`` are copied across. A DataArray carrying no georeferencing yields
    an unreferenced dataset rather than an error.

    Examples
    --------
    >>> import eeo
    >>> import rioxarray
    >>> da = rioxarray.open_rasterio("scene.tif")
    >>> ds = eeo.from_xarray(da)
    >>> ndvi = ds.ndvi(red=1, nir=4)
    """
    from eeo.core.loader import load_array

    xr = _import_xarray()
    _require_dataarray(da, xr)

    y_dim, x_dim = _spatial_dims(da)
    ordered = _as_band_first(da, y_dim, x_dim)
    transform = _transform_from(ordered, y_dim, x_dim)

    array = ordered.values
    if array.ndim == 2:
        array = array[np.newaxis, ...]

    return load_array(
        array,
        transform=transform,
        crs=ordered.rio.crs,
        nodata=ordered.rio.nodata,
        timestamp=_timestamp_from(ordered),
        attrs={key: value for key, value in ordered.attrs.items() if key not in _MANAGED_ATTRS},
        band_names=_band_names_from(ordered, array.shape[0]),
    )
