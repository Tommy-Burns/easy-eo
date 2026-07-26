"""Search SpatioTemporal Asset Catalogs (STAC) for imagery.

:func:`stac_search` queries a STAC API — Microsoft Planetary Computer by default —
and returns the matching scenes as a :class:`STACSearchResult`: an ordered,
timestamped sequence of :class:`STACItem` wrappers. Ordering is chronological
and every item carries its acquisition time, which is what lets a search
result stand in directly for a time series later on.

Searching is metadata-only. It performs HTTP requests against the catalog but
reads no pixel data, so a search over thousands of scenes stays cheap.

This module needs the optional ``stac`` extra::

    pip install "easy-eo[stac]"

Examples
--------
>>> import eeo
>>> results = eeo.stac_search(
...     "sentinel-2-l2a",
...     bbox=(11.0, 46.5, 11.2, 46.7),
...     datetime="2023-06-01/2023-08-31",
...     cloud_cover=20,
...     limit=5,
... )  # doctest: +SKIP
>>> [item.timestamp for item in results]  # doctest: +SKIP
[datetime.datetime(2023, 6, 5, 10, 6, 21, tzinfo=datetime.timezone.utc), ...]
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, overload

import numpy as np
import rasterio as rio
from rasterio import windows as rio_windows
from rasterio.transform import array_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from eeo._optional import import_optional
from eeo.common import normalize_resampling_method
from eeo.core.core import EEORasterDataset
from eeo.core.exceptions import ValidationError
from eeo.core.loader import load_array
from eeo.core.types import ResamplingMethod

#: Microsoft Planetary Computer's STAC API — the default ``catalog`` for
#: :func:`stac_search`, and the only catalog contacted unless another is given.
PLANETARY_COMPUTER_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Catalogs whose asset URLs must be signed before they can be read. Signing is
# applied automatically for these hosts unless the caller overrides ``sign``.
_SIGNED_CATALOG_HOSTS = ("planetarycomputer.microsoft.com",)

# Accepted forms for the ``datetime`` argument: a single instant, an
# RFC 3339 / ISO 8601 interval string ("start/end", open-ended with ".."), or a
# (start, end) pair. Passed through to pystac-client unchanged.
DatetimeSpec = str | dt.datetime | tuple[str | dt.datetime | None, str | dt.datetime | None] | list

_EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)


def _parse_timestamp(value: Any) -> dt.datetime | None:
    """Coerce a STAC datetime value to a datetime, or None if unusable."""
    if isinstance(value, dt.datetime):
        return value
    if not isinstance(value, str):
        return None
    try:
        # STAC timestamps are RFC 3339; Python < 3.11 cannot parse the "Z".
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _item_timestamp(item: Any) -> dt.datetime | None:
    """Return an item's acquisition time, falling back to its range start."""
    timestamp = _parse_timestamp(getattr(item, "datetime", None))
    if timestamp is not None:
        return timestamp
    properties = getattr(item, "properties", None) or {}
    return _parse_timestamp(properties.get("start_datetime"))


def _sort_key(item: STACItem) -> tuple[bool, dt.datetime]:
    """Order items chronologically, placing undated ones last."""
    timestamp = item.timestamp
    if timestamp is None:
        return (True, _EPOCH)
    if timestamp.tzinfo is None:
        # A catalog that omits the offset is reporting UTC by STAC convention.
        timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
    return (False, timestamp)


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


class _Grid(NamedTuple):
    """The output grid every asset of one load is read onto."""

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


def _aligned_window(src: Any, grid: _Grid) -> Any | None:
    """Return an exact pixel window onto ``grid``, or None if a warp is needed.

    An asset sharing the target grid's CRS and resolution can be read directly,
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


def _asset_band_names(src: Any, asset: str) -> list[str | None]:
    """Name a single-band asset after itself; number the bands of a stack."""
    if src.count == 1:
        return [asset]
    descriptions = list(src.descriptions or ())
    names: list[str | None] = []
    for index in range(src.count):
        description = descriptions[index] if index < len(descriptions) else None
        names.append(description or f"{asset}_{index + 1}")
    return names


def _read_first_asset(href: str, asset: str, bbox: Sequence[float] | None) -> tuple:
    """Read the leading asset, which defines the grid the rest are read onto."""
    with rio.Env(**_GDAL_HTTP_ENV), rio.open(href) as src:
        window = None if bbox is None else _crop_window(src, bbox)
        array = src.read(window=window)
        transform = src.transform if window is None else src.window_transform(window)
        grid = _Grid(src.crs, transform, array.shape[-1], array.shape[-2])
        return array, grid, src.nodata, _asset_band_names(src, asset)


def _read_onto_grid(href: str, asset: str, grid: _Grid, resampling: Any) -> tuple:
    """Read a further asset onto ``grid``, warping only when it does not fit."""
    with rio.Env(**_GDAL_HTTP_ENV), rio.open(href) as src:
        window = _aligned_window(src, grid)
        if window is not None:
            return src.read(window=window), _asset_band_names(src, asset)

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
            return vrt.read(), _asset_band_names(src, asset)


class STACItem:
    """One scene from a STAC search, with its acquisition time.

    Wraps the underlying ``pystac.Item`` with a stable, documented surface;
    the raw item stays reachable through :attr:`item` for anything this
    wrapper does not expose.

    Parameters
    ----------
    item : pystac.Item
        The item returned by the STAC client.
    search_bbox : sequence of float or None, default None
        Area of interest of the search that produced the item, in WGS 84
        lon/lat. :meth:`load` crops to it by default.
    """

    def __init__(self, item: Any, *, search_bbox: Sequence[float] | None = None) -> None:
        self._item = item
        self._timestamp = _item_timestamp(item)
        self._search_bbox = None if search_bbox is None else tuple(float(v) for v in search_bbox)

    @property
    def item(self) -> Any:
        """Return the underlying ``pystac.Item``.

        Returns
        -------
        pystac.Item
            The unwrapped item, for access to the full STAC API.
        """
        return self._item

    @property
    def id(self) -> str:
        """Return the item's catalog identifier.

        Returns
        -------
        str
            The STAC item id (e.g. a Sentinel-2 product name).
        """
        return str(getattr(self._item, "id", ""))

    @property
    def collection(self) -> str | None:
        """Return the collection the item belongs to.

        Returns
        -------
        str or None
            The collection id, or None if the item declares none.
        """
        collection = getattr(self._item, "collection_id", None)
        return None if collection is None else str(collection)

    @property
    def timestamp(self) -> dt.datetime | None:
        """Return the acquisition time of the scene.

        Returns
        -------
        datetime.datetime or None
            The item's ``datetime``, falling back to ``start_datetime`` for
            items that describe a range. None when the catalog reports neither
            or reports an unparseable value.
        """
        return self._timestamp

    @property
    def properties(self) -> Mapping[str, Any]:
        """Return the item's STAC properties.

        Returns
        -------
        Mapping
            The raw properties mapping (cloud cover, platform, projection
            metadata, and so on).
        """
        return getattr(self._item, "properties", None) or {}

    @property
    def assets(self) -> Mapping[str, Any]:
        """Return the item's assets keyed by name.

        Returns
        -------
        Mapping
            Mapping of asset name (e.g. ``"B04"``, ``"visual"``) to the STAC
            asset object, whose ``href`` points at the data.
        """
        return getattr(self._item, "assets", None) or {}

    @property
    def asset_names(self) -> list[str]:
        """Return the names of the assets this item offers.

        Returns
        -------
        list of str
            Asset keys in catalog order.
        """
        return list(self.assets)

    @property
    def bbox(self) -> tuple[float, ...] | None:
        """Return the item's bounding box in WGS 84 lon/lat.

        Returns
        -------
        tuple of float or None
            ``(minx, miny, maxx, maxy)`` in degrees, or None if the item
            declares no bounding box.
        """
        bbox = getattr(self._item, "bbox", None)
        return None if bbox is None else tuple(float(value) for value in bbox)

    @property
    def cloud_cover(self) -> float | None:
        """Return the scene's cloud cover percentage.

        Returns
        -------
        float or None
            The ``eo:cloud_cover`` property (0-100), or None when the item
            does not report it.
        """
        value = self.properties.get("eo:cloud_cover")
        return None if value is None else float(value)

    @property
    def search_bbox(self) -> tuple[float, ...] | None:
        """Return the area of interest :meth:`load` crops to by default.

        Returns
        -------
        tuple of float or None
            ``(minx, miny, maxx, maxy)`` in WGS 84 lon/lat degrees, taken from
            the search that produced this item, or None if the search set no
            spatial filter.
        """
        return self._search_bbox

    def load(
        self,
        assets: str | Sequence[str],
        *,
        bbox: Sequence[float] | None = None,
        crop: bool = True,
        resampling: ResamplingMethod | Any = "nearest",
    ) -> EEORasterDataset:
        """Read one or more of the item's assets into a raster dataset.

        Reads only the area of interest by default: the asset is opened
        remotely and only the pixels covering the AOI are fetched, as HTTP
        range requests against the cloud-optimized GeoTIFF, so a small AOI over
        a full Sentinel-2 tile transfers a fraction of the band. Several assets
        are stacked into one multi-band dataset, each band named after the
        asset it came from.

        Parameters
        ----------
        assets : str or sequence of str
            Asset key (e.g. ``"B04"``, ``"nir08"``) or several keys to stack,
            in the order they should become bands. See
            :attr:`STACItem.asset_names` for what this item offers.
        bbox : sequence of float or None, default None
            Area to read, as ``(minx, miny, maxx, maxy)`` in **WGS 84 lon/lat
            degrees**. None uses :attr:`search_bbox`, the AOI of the search
            that produced this item; if that is also None, the whole scene is
            read.
        crop : bool, default True
            Whether to crop at all. False reads the entire scene and cannot be
            combined with ``bbox``.
        resampling : str or rasterio.enums.Resampling, default "nearest"
            Method used when an asset has to be resampled onto the first
            asset's grid — a Sentinel-2 20 m band stacked with a 10 m one, for
            instance. Nearest neighbour by default so values are never blended.

        Returns
        -------
        EEORasterDataset
            Rasterio-backed dataset holding the requested window, with the
            asset's CRS, a transform matching the cropped window, the first
            asset's nodata value, band names taken from the asset keys, the
            item's acquisition time as ``timestamp``, and the item id,
            collection, and asset list in ``attrs``. The dtype is the asset's
            own; stacking assets of different dtypes promotes to their common
            NumPy dtype.

        Raises
        ------
        ValidationError
            If ``assets`` is empty or names an asset the item does not have,
            if ``bbox`` is not four ordered lon/lat values, if ``bbox`` is
            combined with ``crop=False``, or if the AOI does not overlap the
            scene.

        Notes
        -----
        Holds the requested window in memory — the AOI crop is what keeps that
        small, so reading a full scene with ``crop=False`` costs a full band in
        RAM. Assets after the first are read onto the first asset's grid, by a
        plain windowed read when they already share it and a warp otherwise.

        Reading a signed Planetary Computer URL must happen while the signature
        is still valid; search, then load, rather than storing items for later.

        Examples
        --------
        >>> results = eeo.stac_search(
        ...     "sentinel-2-l2a", bbox=(11.0, 46.5, 11.2, 46.7), limit=1
        ... )  # doctest: +SKIP
        >>> scene = results[0].load(["B04", "B08"])  # doctest: +SKIP
        >>> ndvi = scene.ndvi(red="B04", nir="B08")  # doctest: +SKIP
        """
        keys = self._resolve_assets(assets)

        if crop is False and bbox is not None:
            raise ValidationError(
                "crop=False reads the whole scene and cannot be combined with a bbox; "
                "pass one or the other"
            )
        aoi: Sequence[float] | None = None
        if crop:
            aoi = _validate_bbox(bbox) if bbox is not None else self._search_bbox

        resampling_enum = normalize_resampling_method(resampling)

        array, grid, nodata, band_names = _read_first_asset(self._href(keys[0]), keys[0], aoi)
        arrays = [array]
        for key in keys[1:]:
            other, other_names = _read_onto_grid(self._href(key), key, grid, resampling_enum)
            arrays.append(other)
            band_names.extend(other_names)

        stacked = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)

        dataset = load_array(
            stacked,
            transform=grid.transform,
            crs=grid.crs,
            nodata=nodata,
            timestamp=self._timestamp,
            attrs={
                "stac_item": self.id,
                "stac_collection": self.collection,
                "stac_assets": list(keys),
            },
            band_names=band_names,
        )
        return dataset.to_rasterio()

    def _resolve_assets(self, assets: str | Sequence[str]) -> list[str]:
        """Normalize the asset argument to a list of keys this item has."""
        if isinstance(assets, str):
            keys = [assets]
        elif isinstance(assets, Sequence):
            keys = [str(key) for key in assets]
        else:
            raise ValidationError(
                f"assets must be an asset key or a sequence of keys; got {assets!r}"
            )
        if not keys:
            raise ValidationError("name at least one asset to load; got an empty sequence")

        available = self.asset_names
        missing = [key for key in keys if key not in available]
        if missing:
            raise ValidationError(
                f"item {self.id!r} has no asset {missing[0]!r}; "
                f"available assets are {', '.join(available) or 'none'}"
            )
        return keys

    def _href(self, asset: str) -> str:
        """Return the URL or path of one of the item's assets."""
        return str(self.assets[asset].href)

    def __repr__(self) -> str:
        """Return a one-line summary: id, acquisition time, and asset count."""
        when = "no date" if self._timestamp is None else self._timestamp.isoformat()
        return f"<STACItem {self.id} {when} ({len(self.assets)} assets)>"


class STACSearchResult(Sequence[STACItem]):
    """An ordered, timestamped sequence of :class:`STACItem` results.

    Behaves like a list of items sorted from oldest to newest (undated items
    last), so it can be indexed, sliced, iterated, and measured with ``len()``.
    The chronological ordering and per-item timestamps are what allow a search
    result to be handed straight to a time-series layer.

    Parameters
    ----------
    items : iterable of STACItem
        Items to hold; sorted chronologically on construction.
    collections : sequence of str
        Collection ids the search covered.
    catalog : str
        URL of the catalog that was searched.
    bbox : sequence of float or None, default None
        The area of interest the search was restricted to, in WGS 84 lon/lat.
        Retained so that loading an asset can crop to it.
    """

    def __init__(
        self,
        items: Sequence[STACItem],
        *,
        collections: Sequence[str],
        catalog: str,
        bbox: Sequence[float] | None = None,
    ) -> None:
        self._items: list[STACItem] = sorted(items, key=_sort_key)
        self._collections = list(collections)
        self._catalog = catalog
        self._bbox = None if bbox is None else tuple(float(value) for value in bbox)

    @property
    def bbox(self) -> tuple[float, ...] | None:
        """Return the area of interest the search was restricted to.

        Returns
        -------
        tuple of float or None
            ``(minx, miny, maxx, maxy)`` in WGS 84 lon/lat degrees, or None if
            the search set no spatial filter.
        """
        return self._bbox

    @property
    def collections(self) -> list[str]:
        """Return the collections the search covered.

        Returns
        -------
        list of str
            Collection ids passed to :func:`stac_search`.
        """
        return list(self._collections)

    @property
    def catalog(self) -> str:
        """Return the catalog URL that was searched.

        Returns
        -------
        str
            The STAC API endpoint.
        """
        return self._catalog

    @property
    def timestamps(self) -> list[dt.datetime | None]:
        """Return the acquisition times of the results, in order.

        Returns
        -------
        list of (datetime.datetime or None)
            One timestamp per item, oldest first; None for an undated item.
        """
        return [item.timestamp for item in self._items]

    def __len__(self) -> int:
        """Return the number of items in the result.

        Returns
        -------
        int
            Count of matching items.
        """
        return len(self._items)

    @overload
    def __getitem__(self, index: int) -> STACItem: ...

    @overload
    def __getitem__(self, index: slice) -> STACSearchResult: ...

    def __getitem__(self, index: int | slice) -> STACItem | STACSearchResult:
        """Return the item at ``index``, or a sliced result for a slice.

        Parameters
        ----------
        index : int or slice
            Position of a single item, or a slice of positions.

        Returns
        -------
        STACItem or STACSearchResult
            The item at ``index``, or a new result holding the sliced items
            and the same collection/catalog metadata.
        """
        if isinstance(index, slice):
            return STACSearchResult(
                self._items[index],
                collections=self._collections,
                catalog=self._catalog,
                bbox=self._bbox,
            )
        return self._items[index]

    def __repr__(self) -> str:
        """Return a one-line summary: item count, time span, and collections."""
        collections = ", ".join(self._collections) or "no collection"
        if not self._items:
            return f"<STACSearchResult: 0 items ({collections})>"
        dated = [ts for ts in self.timestamps if ts is not None]
        span = ""
        if dated:
            span = f" from {dated[0].date()} to {dated[-1].date()}"
        return f"<STACSearchResult: {len(self._items)} items{span} ({collections})>"


def _validate_collection(collection: str | Sequence[str]) -> list[str]:
    """Normalize the collection argument to a non-empty list of ids."""
    if isinstance(collection, str):
        collections = [collection]
    elif isinstance(collection, Sequence):
        collections = [str(name) for name in collection]
    else:
        raise ValidationError(
            f"collection must be a collection id or a sequence of ids; got {collection!r}"
        )
    if not collections or any(not name for name in collections):
        raise ValidationError(
            f"collection must name at least one non-empty collection id; got {collection!r}"
        )
    return collections


def _validate_bbox(bbox: Sequence[float]) -> list[float]:
    """Validate a WGS 84 lon/lat bounding box and return it as a list."""
    if isinstance(bbox, str) or not isinstance(bbox, Sequence):
        raise ValidationError(
            f"bbox must be (minx, miny, maxx, maxy) in WGS 84 lon/lat degrees; got {bbox!r}"
        )
    if len(bbox) != 4:
        raise ValidationError(
            "bbox must be (minx, miny, maxx, maxy) in WGS 84 lon/lat degrees; "
            f"got {len(bbox)} values"
        )
    minx, miny, maxx, maxy = (float(value) for value in bbox)
    if minx >= maxx or miny >= maxy:
        raise ValidationError(
            "bbox must have minx < maxx and miny < maxy in WGS 84 lon/lat degrees; "
            f"got {tuple(bbox)!r}"
        )
    return [minx, miny, maxx, maxy]


def _search_parameters(
    collections: list[str],
    bbox: Sequence[float] | None,
    datetime: DatetimeSpec | None,
    cloud_cover: float | None,
    limit: int | None,
) -> dict[str, Any]:
    """Validate the query arguments and build the pystac-client parameters."""
    params: dict[str, Any] = {"collections": collections}

    if bbox is not None:
        params["bbox"] = _validate_bbox(bbox)

    if datetime is not None:
        params["datetime"] = datetime

    if cloud_cover is not None:
        if not 0 <= float(cloud_cover) <= 100:
            raise ValidationError(
                f"cloud_cover must be a percentage between 0 and 100; got {cloud_cover!r}"
            )
        # STAC's query extension; "lte" makes cloud_cover=20 mean "at most 20%".
        params["query"] = {"eo:cloud_cover": {"lte": float(cloud_cover)}}

    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValidationError(f"limit must be a positive integer; got {limit!r}")
        # pystac-client's `limit` is a page size; `max_items` caps the total.
        params["max_items"] = limit

    return params


def _open_catalog(catalog: str, sign: bool | None) -> Any:
    """Open a STAC client for ``catalog``, signing asset URLs when required."""
    pystac_client = import_optional("pystac_client", extra="stac", purpose="STAC search")

    if sign is None:
        sign = any(host in catalog for host in _SIGNED_CATALOG_HOSTS)

    modifier = None
    if sign:
        planetary_computer = import_optional(
            "planetary_computer",
            extra="stac",
            purpose="signing Planetary Computer asset URLs",
        )
        modifier = planetary_computer.sign_inplace

    return pystac_client.Client.open(catalog, modifier=modifier)


def stac_search(
    collection: str | Sequence[str],
    *,
    bbox: Sequence[float] | None = None,
    datetime: DatetimeSpec | None = None,
    cloud_cover: float | None = None,
    limit: int | None = None,
    catalog: str = PLANETARY_COMPUTER_STAC_URL,
    sign: bool | None = None,
) -> STACSearchResult:
    """Search a STAC catalog for scenes matching a place, time, and cloudiness.

    Queries the catalog's search endpoint and returns the matching items in
    chronological order, each carrying its acquisition timestamp. Only
    metadata is fetched — no pixels are read and nothing is downloaded.

    Parameters
    ----------
    collection : str or sequence of str
        Collection id to search (e.g. ``"sentinel-2-l2a"``,
        ``"landsat-c2-l2"``), or several ids to search at once.
    bbox : sequence of float or None, default None
        Area of interest as ``(minx, miny, maxx, maxy)`` in **WGS 84 lon/lat
        degrees** (EPSG:4326), which is what the STAC API expects regardless of
        the imagery's own CRS. None searches the whole catalog extent.
    datetime : str or datetime.datetime or tuple, default None
        Time filter, passed through to the catalog: a single instant, an
        interval string such as ``"2023-06-01/2023-08-31"`` (``".."`` leaves an
        end open), or a ``(start, end)`` pair. A bare date covers its whole
        day, so a closing date is inclusive. None searches all times.
    cloud_cover : float or None, default None
        Maximum scene cloud cover in percent (0-100), applied by the catalog
        against each scene's ``eo:cloud_cover``. None applies no cloud filter.
        How scenes that do not report cloud cover are treated is up to the
        catalog; see the note below on the filter's exactness.
    limit : int or None, default None
        Maximum number of items to return. None returns every match, which
        may be many pages of results.
    catalog : str, default :data:`PLANETARY_COMPUTER_STAC_URL`
        STAC API endpoint to search. Defaults to Microsoft Planetary Computer
        (``https://planetarycomputer.microsoft.com/api/stac/v1``), which is the
        only catalog contacted unless another endpoint is given here. Any STAC
        API works, e.g. Earth Search.
    sign : bool or None, default None
        Whether to sign asset URLs with ``planetary-computer``. None signs
        automatically when ``catalog`` is a Planetary Computer endpoint and
        leaves other catalogs untouched.

    Returns
    -------
    STACSearchResult
        Matching items, ordered oldest to newest (undated items last), each
        exposing its ``timestamp``, ``assets``, and STAC ``properties``. The
        result also retains the search's ``bbox`` as its area of interest.
        Empty when nothing matches — that is not an error.

    Raises
    ------
    MissingDependencyError
        If the ``stac`` extra is not installed.
    ValidationError
        If ``collection`` is empty, ``bbox`` is not four ordered lon/lat
        values, ``cloud_cover`` is outside 0-100, or ``limit`` is not a
        positive integer.

    Notes
    -----
    Performs network requests against the catalog and holds only item metadata
    in memory; no pixel data is read (see :meth:`STACItem.assets` for the URLs
    a later read would use). Errors raised by the catalog itself — an
    unreachable endpoint, an unknown collection — propagate from pystac-client
    as ``pystac_client.exceptions.APIError``.

    Signed Planetary Computer asset URLs are time-limited, so a search result
    is best used soon after it is created rather than stored and reused.

    ``cloud_cover`` is evaluated by the catalog, not by Easy-EO, and some
    catalogs (Planetary Computer among them) match against their own indexed
    value rather than the item's reported ``eo:cloud_cover``. A returned scene
    can therefore sit a little above the threshold. Filter the result yourself
    when an exact cut-off matters::

        clear = [item for item in results if item.cloud_cover <= 20]

    Catalogs may also return several near-identical scenes for one acquisition
    (the same tile reprocessed under different baselines); they share a
    timestamp and are kept as separate items.

    Examples
    --------
    >>> import eeo
    >>> results = eeo.stac_search(
    ...     "sentinel-2-l2a",
    ...     bbox=(11.0, 46.5, 11.2, 46.7),
    ...     datetime="2023-06-01/2023-08-31",
    ...     cloud_cover=20,
    ...     limit=5,
    ... )  # doctest: +SKIP
    >>> len(results), results[0].id  # doctest: +SKIP
    (5, 'S2A_MSIL2A_20230605T100031_R122_T32TPS_20230605T173940')
    """
    collections = _validate_collection(collection)
    params = _search_parameters(collections, bbox, datetime, cloud_cover, limit)

    client = _open_catalog(catalog, sign)
    found = client.search(**params)

    return STACSearchResult(
        [STACItem(item, search_bbox=params.get("bbox")) for item in found.items()],
        collections=collections,
        catalog=catalog,
        bbox=params.get("bbox"),
    )
