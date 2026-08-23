"""Reading several georeferenced sources onto one common grid.

A scene arrives as one file per band — STAC assets over HTTP, or the JP2s and
GeoTIFFs inside a downloaded product — and turning a chosen subset of them into
a single multi-band raster is the same job either way. The rule this module
implements is:

* the **first** source defines the output grid (CRS, transform, size) and
  supplies the result's nodata value;
* every later source is read onto that grid — by a plain windowed read when it
  already shares it, which is exact and cheap, and through a
  :class:`~rasterio.vrt.WarpedVRT` otherwise, which covers a coarser
  resolution, a different projection, a sub-pixel offset, or a footprint
  smaller than the requested area;
* the arrays are concatenated in the order given, and each contributes its band
  names — the source's own name for a single-band file, its GDAL descriptions
  (or ``name_1..name_n``) for a stack.

Cropping is applied to the first source only, because the grid it produces then
constrains every other read.

Notes
-----
The GDAL configuration in :data:`_GDAL_HTTP_ENV` is tuned for cloud-optimized
GeoTIFFs over HTTP. ``GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR`` stops an object
store from listing a whole container to answer a directory probe, but it also
stops GDAL from finding a local file's siblings — which a multi-file product
format relies on. A local-product reader must revisit this setting rather than
inherit it unexamined.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
import rasterio as rio
from rasterio import windows as rio_windows
from rasterio.transform import array_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from eeo.core.exceptions import ValidationError

# GDAL settings for reading a remote COG efficiently. Object stores answer a
# directory probe by listing the whole container, which costs far more than the
# read itself; HTTP/2 multiplexing and the VSI cache keep the range requests for
# the tiles we actually want.
_GDAL_HTTP_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "VSI_CACHE": "TRUE",
}


class Grid(NamedTuple):
    """The output grid every source of one read is placed onto."""

    crs: Any
    transform: Any
    width: int
    height: int


def _crop_window(src: Any, bbox: Sequence[float]) -> Any:
    """Return the pixel window of ``src`` covering a WGS 84 lon/lat bbox."""
    if src.crs is None:
        raise ValidationError(
            "cropping needs a georeferenced asset; this asset declares no CRS. "
            "Pass crop=False to read it whole."
        )

    bounds = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
    window = rio_windows.from_bounds(*bounds, transform=src.transform)

    # Round outward to whole pixels so the AOI is fully covered.
    col_off = math.floor(window.col_off)
    row_off = math.floor(window.row_off)
    requested = rio_windows.Window(
        col_off,
        row_off,
        max(math.ceil(window.col_off + window.width) - col_off, 1),
        max(math.ceil(window.row_off + window.height) - row_off, 1),
    )

    scene = rio_windows.Window(0, 0, src.width, src.height)
    if not rio_windows.intersect(requested, scene):
        raise ValidationError(
            f"the requested area does not overlap this scene; got bbox {tuple(bbox)!r} "
            f"in WGS 84 lon/lat, while the scene covers "
            f"{transform_bounds(src.crs, 'EPSG:4326', *src.bounds, densify_pts=21)!r}"
        )
    return rio_windows.intersection(requested, scene)


def _aligned_window(src: Any, grid: Grid) -> Any | None:
    """Return an exact pixel window onto ``grid``, or None if a warp is needed.

    A source sharing the target grid's CRS and resolution can be read directly,
    which is both cheaper and exact; anything else goes through a warp.
    """
    if src.crs is None or src.crs != grid.crs:
        return None
    bounds = array_bounds(grid.height, grid.width, grid.transform)
    window = rio_windows.from_bounds(*bounds, transform=src.transform)

    tol = 1e-6
    if abs(window.width - grid.width) > tol or abs(window.height - grid.height) > tol:
        return None
    if abs(window.col_off - round(window.col_off)) > tol:
        return None
    if abs(window.row_off - round(window.row_off)) > tol:
        return None

    col_off, row_off = round(window.col_off), round(window.row_off)
    # A window running off the edge would come back short and break the grid;
    # let the warp path fill those pixels with nodata instead.
    if col_off < 0 or row_off < 0:
        return None
    if col_off + grid.width > src.width or row_off + grid.height > src.height:
        return None
    return rio_windows.Window(col_off, row_off, grid.width, grid.height)


def _band_names(src: Any, name: str) -> list[str | None]:
    """Name a single-band source after itself; number the bands of a stack."""
    if src.count == 1:
        return [name]
    descriptions = list(src.descriptions or ())
    names: list[str | None] = []
    for index in range(src.count):
        description = descriptions[index] if index < len(descriptions) else None
        names.append(description or f"{name}_{index + 1}")
    return names


def _read_first(href: str, name: str, bbox: Sequence[float] | None) -> tuple:
    """Read the leading source, which defines the grid the rest are read onto."""
    with rio.Env(**_GDAL_HTTP_ENV), rio.open(href) as src:
        window = None if bbox is None else _crop_window(src, bbox)
        array = src.read(window=window)
        transform = src.transform if window is None else src.window_transform(window)
        grid = Grid(src.crs, transform, array.shape[-1], array.shape[-2])
        return array, grid, src.nodata, _band_names(src, name)


def _read_onto_grid(href: str, name: str, grid: Grid, resampling: Any) -> tuple:
    """Read a further source onto ``grid``, warping only when it does not fit."""
    with rio.Env(**_GDAL_HTTP_ENV), rio.open(href) as src:
        window = _aligned_window(src, grid)
        if window is not None:
            return src.read(window=window), _band_names(src, name)

        # Different CRS, resolution, or sub-pixel offset: let GDAL resample
        # straight onto the target grid rather than reading and fixing up after.
        with WarpedVRT(
            src,
            crs=grid.crs,
            transform=grid.transform,
            width=grid.width,
            height=grid.height,
            resampling=resampling,
            src_nodata=src.nodata,
            nodata=src.nodata,
        ) as vrt:
            return vrt.read(), _band_names(src, name)


def read_onto_common_grid(
    sources: Sequence[tuple[str, str]],
    *,
    bbox: Sequence[float] | None,
    resampling: Any,
) -> tuple[np.ndarray, Grid, float | None, list[str | None]]:
    """Read several sources onto the grid of the first and stack them.

    Parameters
    ----------
    sources : sequence of tuple of str
        ``(href, name)`` pairs in the order they should become bands. The href
        is anything rasterio can open — a local path or a remote URL — and the
        name labels the bands that source contributes. Must not be empty.
    bbox : sequence of float or None
        Area to read as ``(minx, miny, maxx, maxy)`` in **WGS 84 lon/lat
        degrees**, applied when opening the first source. None reads it whole.
    resampling : rasterio.enums.Resampling
        Method used when a later source has to be warped onto the grid.

    Returns
    -------
    tuple
        ``(array, grid, nodata, band_names)`` — the stacked bands as one array
        of shape ``(bands, height, width)``, the grid they share, the first
        source's nodata value, and one name per band. The dtype is the
        sources' own; stacking sources of different dtypes promotes to their
        common NumPy dtype.

    Raises
    ------
    ValidationError
        If ``sources`` is empty, if a source needing a crop declares no CRS, or
        if ``bbox`` does not overlap the first source.

    Notes
    -----
    Holds the stacked result in memory; a bbox is what keeps that bounded.
    """
    if not sources:
        raise ValidationError("name at least one source to read; got an empty sequence")

    href, name = sources[0]
    array, grid, nodata, band_names = _read_first(href, name, bbox)
    arrays = [array]
    for href, name in sources[1:]:
        other, other_names = _read_onto_grid(href, name, grid, resampling)
        arrays.append(other)
        band_names.extend(other_names)

    stacked = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)
    return stacked, grid, nodata, band_names
