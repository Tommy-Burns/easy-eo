"""Ordered, timestamped collections of rasters.

Holds :class:`EEOTimeSeries` and the :func:`time_series` entry point that
builds one. A time series is deliberately a separate type from a multi-band
dataset: bands are what a sensor measured at one moment, timesteps are the same
measurement repeated, and collapsing the two would make a spectral stack and a
temporal stack indistinguishable.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import os
import tempfile
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, cast, overload

from rasterio import CRS
from rasterio.transform import Affine

from eeo.core.adapters.xarray import validate_chunks
from eeo.core.core import EEORasterDataset
from eeo.core.exceptions import ValidationError
from eeo.core.loader import load_raster
from eeo.core.types import ChunkSpec, ResamplingMethod, StrPath
from eeo.io.stac import STACItem, STACSearchResult

_UTC = dt.timezone.utc

_NO_TIMESTAMP = (
    "timestep {index} carries no timestamp, and a time series is ordered by "
    "time. Give every dataset one at load time (load_raster(..., "
    "timestamp=...); the Sentinel-2, Landsat and STAC loaders do it for you), "
    "or pass timestamps=[...] with one entry per dataset."
)

_FOLDER_NOT_IMPLEMENTED = (
    "building a time series from a folder is not implemented yet. Load the "
    "scenes yourself and hand them over, which is the same thing with the "
    "timestamps made explicit:\n\n"
    "    paths = sorted(pathlib.Path({folder!r}).glob({pattern!r}))\n"
    "    ts = eeo.time_series([eeo.load_raster(p, timestamp=...) for p in paths])\n\n"
    "A STAC search needs none of that — every item states its acquisition "
    "time — so eeo.time_series(eeo.stac_search(...), assets=[...]) is the "
    "supported path today."
)


def _as_utc(value: object, *, index: int) -> dt.datetime:
    """Coerce one timestamp to a timezone-aware UTC datetime.

    A naive datetime is read as UTC rather than rejected, matching what the
    Sentinel-2 and Landsat metadata parsers already do. Without this, a series
    mixing a loader's aware timestamp with a hand-written naive one could not
    even be sorted: comparing the two raises ``TypeError``.
    """
    if not isinstance(value, dt.datetime):
        raise ValidationError(
            f"timestamp {index} must be a datetime.datetime; got {type(value).__name__}"
        )
    return value if value.tzinfo is not None else value.replace(tzinfo=_UTC)


def _resolve_timestamps(
    datasets: Sequence[EEORasterDataset], timestamps: Sequence[dt.datetime] | None
) -> list[dt.datetime]:
    """Return one UTC timestamp per dataset, from ``timestamps`` or the datasets."""
    if timestamps is None:
        resolved = []
        for index, ds in enumerate(datasets):
            if ds.timestamp is None:
                raise ValidationError(_NO_TIMESTAMP.format(index=index))
            resolved.append(_as_utc(ds.timestamp, index=index))
        return resolved

    given = list(timestamps)
    if len(given) != len(datasets):
        raise ValidationError(
            f"timestamps has {len(given)} entries but there are {len(datasets)} "
            f"datasets; give exactly one timestamp per dataset"
        )
    return [_as_utc(value, index=index) for index, value in enumerate(given)]


def _through_file(
    scene: EEORasterDataset,
    directory: StrPath,
    index: int,
    chunks: ChunkSpec | None,
    *,
    timestamp: dt.datetime,
) -> EEORasterDataset:
    """Write one in-memory raster into ``directory`` and reopen it from there.

    Reopening is what bounds a series' memory: a file-backed dataset reads no
    pixels until an operation asks for a window, while an in-memory one — what a
    catalog load and every operation produce — holds its whole array. The
    timestamp and attrs are carried over explicitly because they live in Python,
    not in the GeoTIFF; band names survive in the file's band descriptions and
    are passed anyway so the reopened dataset is identical either way.
    """
    path = Path(os.fspath(directory)) / f"{index:04d}_{timestamp:%Y%m%dT%H%M%S}.tif"
    scene.save_raster(path)
    attrs = dict(scene.attrs)
    band_names = scene.band_names
    scene.close()
    return load_raster(path, chunks=chunks, timestamp=timestamp, attrs=attrs, band_names=band_names)


def _apply(op: Callable[..., Any], ds: EEORasterDataset, kwargs: dict[str, Any]) -> Any:
    """Run one operation on one dataset, exactly as calling it on the dataset would.

    An Easy-EO operation is written as a free function and bound onto
    ``EEORasterDataset`` by ``@eeo_raster_op``, and it is the bound wrapper — not
    the function — that carries the timestamp, attrs and band names of the input
    onto the result. Calling the free function directly would quietly drop all
    three, so a registered operation is invoked through its bound method, found
    by the identity ``functools.wraps`` records. Anything else, including a
    user's own function or a lambda, is called directly.
    """
    name = getattr(op, "__name__", None)
    if name is not None:
        bound = getattr(ds, name, None)
        if bound is not None and getattr(bound, "__wrapped__", None) is op:
            return bound(**kwargs)
    return op(ds, **kwargs)


class EEOTimeSeries(Sequence[EEORasterDataset]):
    """An ordered, timestamped collection of rasters covering one area.

    Behaves like a list of :class:`~eeo.core.core.EEORasterDataset` sorted from
    oldest to newest, so it can be indexed, sliced, iterated, and measured with
    ``len()``. Slicing returns another time series; indexing returns the
    dataset at that timestep, which is an ordinary dataset with every operation
    available on it.

    Every timestep must carry a timestamp: the ordering is the point of the
    type, and a trajectory or composite indexed by nothing is not a time
    series. The STAC, Sentinel-2 and Landsat loaders all record one, so this is
    only a constraint on a hand-assembled series.

    Construct one with :func:`eeo.time_series`, or with
    :meth:`from_stac` when the scenes come from a catalog search.

    Parameters
    ----------
    datasets : iterable of EEORasterDataset
        Rasters making up the series, in any order — they are sorted by
        timestamp. Datasets are held by reference, not copied.
    timestamps : sequence of datetime.datetime or None, default None
        One acquisition time per dataset, overriding whatever the datasets
        carry. None takes each dataset's own ``timestamp``. A naive datetime is
        read as UTC. The datasets themselves are left untouched either way, so
        supplying timestamps here does not mutate the caller's objects.

    Raises
    ------
    ValidationError
        If ``datasets`` is empty, holds anything that is not an
        ``EEORasterDataset``, or has a timestep with no timestamp and no
        ``timestamps`` entry; or if ``timestamps`` is given with a length that
        does not match.

    Notes
    -----
    Holds whatever its datasets hold: a series of file-backed datasets keeps
    one GDAL handle per timestep and no pixels, while a series of in-memory
    datasets (what a catalog load returns) keeps every window in memory. This
    is why :meth:`from_stac` caches scenes to disk by default.

    Consistency of CRS, grid and band structure across timesteps is not
    enforced here yet; the grid properties describe the earliest timestep.

    Examples
    --------
    >>> import datetime as dt
    >>> import numpy as np
    >>> import eeo
    >>> scenes = [
    ...     eeo.load_array(
    ...         np.full((4, 4), i, dtype="uint16"),
    ...         crs=32633,
    ...         timestamp=dt.datetime(2023, i, 1),
    ...     )
    ...     for i in (3, 1, 2)
    ... ]
    >>> ts = eeo.time_series(scenes)
    >>> len(ts)
    3
    >>> [stamp.month for stamp in ts.timestamps]
    [1, 2, 3]
    """

    def __init__(
        self,
        datasets: Iterable[EEORasterDataset],
        *,
        timestamps: Sequence[dt.datetime] | None = None,
    ) -> None:
        items = list(datasets)
        if not items:
            raise ValidationError(
                "a time series needs at least one dataset; got an empty collection"
            )
        for index, item in enumerate(items):
            if not isinstance(item, EEORasterDataset):
                raise ValidationError(
                    f"timestep {index} is a {type(item).__name__}, not an "
                    f"EEORasterDataset; a time series holds datasets"
                )

        stamps = _resolve_timestamps(items, timestamps)
        # A stable sort, so reprocessed duplicates of one acquisition — which
        # catalogs do return — keep the order they arrived in.
        order = sorted(range(len(items)), key=lambda index: stamps[index])
        self._datasets: list[EEORasterDataset] = [items[index] for index in order]
        self._timestamps: list[dt.datetime] = [stamps[index] for index in order]
        # Set by from_stac when it owns a temporary scene cache.
        self._cache: tempfile.TemporaryDirectory | None = None
        # Set by from_stac when its scenes were opened on the lazy backend, so
        # that a series which started lazy stays lazy through map(save_dir=).
        self._chunks: ChunkSpec | None = None

    # ========================
    # Constructors
    # ========================
    @classmethod
    def from_stac(
        cls,
        result: STACSearchResult | Iterable[STACItem],
        assets: str | Sequence[str],
        *,
        bbox: Sequence[float] | None = None,
        crop: bool = True,
        mask: bool = False,
        resampling: ResamplingMethod | Any = "nearest",
        cache: bool | StrPath = True,
        chunks: ChunkSpec | None = None,
    ) -> EEOTimeSeries:
        """Build a series from a STAC search, reading the same assets from each item.

        A search result is already a time series in all but type: its items are
        ordered oldest-first and each states its acquisition time, so nothing
        has to be parsed out of filenames. Each item's assets are read with
        :meth:`eeo.io.STACItem.load`, which crops to the search area by
        default, and the scenes become the timesteps.

        Parameters
        ----------
        result : STACSearchResult or iterable of STACItem
            Items to read, typically straight from :func:`eeo.stac_search`.
        assets : str or sequence of str
            Asset key, or keys to stack into bands, read from every item — the
            same set for each, since a series with different bands per timestep
            could not be reduced. See :attr:`eeo.io.STACItem.asset_names`.
        bbox : sequence of float or None, default None
            Area to read from every item, as ``(minx, miny, maxx, maxy)`` in
            WGS 84 lon/lat degrees. None uses each item's search area.
        crop : bool, default True
            Whether to crop at all. False reads whole scenes, which for a
            series multiplies a full tile by the number of timesteps.
        mask : bool, default False
            Set pixels outside the search geometry to nodata, following its
            outline rather than its bounding box. Requires a search made with
            ``intersects``.
        resampling : str or rasterio.enums.Resampling, default "nearest"
            Method used where an asset must be resampled onto the first
            asset's grid. Nearest by default so values are never blended.
        cache : bool or str or path-like, default True
            Where the scenes live once read. True writes each scene to a
            temporary directory and reopens it from there, so the series holds
            file handles instead of arrays and :meth:`close` removes the files.
            A path does the same in a directory you keep, which also makes the
            read reusable: signed catalog URLs expire, cached GeoTIFFs do not.
            False keeps every scene in memory, which is faster for a small area
            and unbounded for a large one.
        chunks : str or int or dict or None, default None
            Chunk sizes for reopening cached scenes on the lazy, dask-chunked
            backend (see :func:`eeo.load_raster`), which needs the ``lazy``
            extra. None reopens them with rasterio, which already defers reads
            — the lazy backend adds dask on top of that, it is not what makes
            the series bounded. Cannot be combined with ``cache=False``: an
            in-memory scene has no file to open lazily.

        Returns
        -------
        EEOTimeSeries
            Series of one timestep per item, oldest first, each carrying the
            item's acquisition time, band names taken from the asset keys, and
            the item id, collection and asset list in ``attrs``.

        Raises
        ------
        ValidationError
            If ``result`` holds no items, if any item has no acquisition time,
            if ``chunks`` is combined with ``cache=False``, if ``chunks`` is
            not a valid chunk specification, or for anything
            :meth:`eeo.io.STACItem.load` rejects (an unknown asset, a bbox
            that is not four ordered lon/lat values, ``mask`` without a search
            geometry).
        MissingDependencyError
            If ``chunks`` is given without the ``lazy`` extra installed.

        Notes
        -----
        Reads every item eagerly, because a signed Planetary Computer URL is
        only valid for a while: a series that deferred its reads would fail
        hours later, in the middle of a computation. With the default cache,
        peak memory is one scene's window rather than the whole series'.

        Examples
        --------
        >>> import eeo
        >>> results = eeo.stac_search(
        ...     "sentinel-2-l2a",
        ...     bbox=(11.0, 46.5, 11.2, 46.7),
        ...     datetime="2023-04-01/2023-09-30",
        ...     cloud_cover=20,
        ... )  # doctest: +SKIP
        >>> ts = eeo.EEOTimeSeries.from_stac(results, ["B04", "B08"])  # doctest: +SKIP
        >>> ts.timestamps[0].date()  # doctest: +SKIP
        datetime.date(2023, 4, 12)
        """
        if chunks is not None:
            if cache is False:
                raise ValidationError(
                    "chunks= opens a cached scene on the lazy backend, so it needs a "
                    "cache: pass cache=True (a temporary directory) or a path, or drop "
                    "chunks= to keep the scenes in memory"
                )
            validate_chunks(chunks)

        items = list(result)
        if not items:
            raise ValidationError(
                "a time series needs at least one scene, and this search returned no "
                "items; widen the search area, the date range, or the cloud filter"
            )
        # Checked before any asset is read: discovering an undated item after
        # forty reads would waste all of them.
        dated: list[tuple[dt.datetime, STACItem]] = []
        for index, item in enumerate(items):
            stamp = item.timestamp
            if stamp is None:
                raise ValidationError(
                    f"item {index} ({item.id}) states no acquisition time, so it "
                    f"cannot take a place in a time series. Drop it from the result "
                    f"before building the series"
                )
            dated.append((stamp, item))
        # Read oldest-first, whatever order the items arrived in: a search
        # result is already chronological, but a hand-built list need not be,
        # and the cache is numbered by read order — so sorting here is what
        # makes the cached files read in time order too. Sorted on the
        # timestamp alone: two items of one acquisition would otherwise be
        # compared to each other, and a STACItem has no ordering.
        dated.sort(key=lambda pair: pair[0])

        holder: tempfile.TemporaryDirectory | None = None
        directory: Path | None = None
        if cache is not False:
            if cache is True:
                holder = tempfile.TemporaryDirectory(prefix="eeo-timeseries-")
                directory = Path(holder.name)
            else:
                directory = Path(os.fspath(cache))
                directory.mkdir(parents=True, exist_ok=True)

        scenes: list[EEORasterDataset] = []
        try:
            # Iterated as the (timestamp, item) pairs built above, so the
            # timestamp is known to be present rather than rechecked here.
            for index, (stamp, item) in enumerate(dated):
                scene = item.load(assets, bbox=bbox, crop=crop, mask=mask, resampling=resampling)
                if directory is not None:
                    scene = _through_file(scene, directory, index, chunks, timestamp=stamp)
                scenes.append(scene)
            series = cls(scenes)
        except BaseException:
            for scene in scenes:
                with contextlib.suppress(Exception):
                    scene.close()
            if holder is not None:
                with contextlib.suppress(Exception):
                    holder.cleanup()
            raise

        series._cache = holder
        series._chunks = chunks
        return series

    @classmethod
    def from_folder(
        cls,
        folder: StrPath,
        pattern: str = "*.tif",
        *,
        chunks: ChunkSpec | None = None,
    ) -> EEOTimeSeries:
        """Build a series from rasters on disk — not implemented yet.

        The signature is here so the shape of the eventual call is fixed and
        documented, and so this path names the code that does the same thing
        today rather than failing as a missing attribute.

        When it lands, it will glob ``folder`` with ``pattern`` (so
        ``"**/*.tif"`` walks subdirectories), open each hit with
        :func:`eeo.load_raster`, and take each timestamp from the filename's
        date, with a hook for files that carry theirs elsewhere.

        Parameters
        ----------
        folder : str or path-like
            Directory holding the rasters.
        pattern : str, default "*.tif"
            Glob pattern selecting them.
        chunks : str or int or dict or None, default None
            Chunk sizes for opening each raster on the lazy backend.

        Returns
        -------
        EEOTimeSeries
            Never returns.

        Raises
        ------
        NotImplementedError
            Always, with the code that does the same thing today.

        Examples
        --------
        >>> import eeo
        >>> eeo.EEOTimeSeries.from_folder("scenes/")  # doctest: +SKIP
        Traceback (most recent call last):
        NotImplementedError: building a time series from a folder is not implemented yet ...
        """
        raise NotImplementedError(
            _FOLDER_NOT_IMPLEMENTED.format(folder=os.fspath(folder), pattern=pattern)
        )

    # ========================
    # Sequence protocol
    # ========================
    def __len__(self) -> int:
        """Return the number of timesteps.

        Returns
        -------
        int
            Count of datasets in the series.
        """
        return len(self._datasets)

    @overload
    def __getitem__(self, index: int) -> EEORasterDataset: ...

    @overload
    def __getitem__(self, index: slice) -> EEOTimeSeries: ...

    def __getitem__(self, index: int | slice) -> EEORasterDataset | EEOTimeSeries:
        """Return the dataset at ``index``, or a series for a slice.

        Parameters
        ----------
        index : int or slice
            Position of one timestep, or a slice of positions.

        Returns
        -------
        EEORasterDataset or EEOTimeSeries
            The dataset at ``index``, or a new series over the sliced
            timesteps. A slice shares its datasets with this series rather than
            copying them, and does not own the scene cache, so closing either
            one affects both.

        Raises
        ------
        ValidationError
            If a slice selects no timesteps. Unlike a list, a series is never
            empty — the grid it reports would have nothing to describe — so an
            empty slice is refused rather than returned.
        IndexError
            If an int index is out of range.
        """
        if isinstance(index, slice):
            return self._derive(self._datasets[index], self._timestamps[index])
        return self._datasets[index]

    def __repr__(self) -> str:
        """Return a one-line summary: timestep count, time span, and grid."""
        count = len(self._datasets)
        span = f"{self._timestamps[0].date()} to {self._timestamps[-1].date()}"
        try:
            first = self._datasets[0]
            height, width = first.get_shape()
            dtype = first.get_metadata().get("dtype", "?")
            crs = first.get_crs()
            epsg = crs.to_epsg() if crs is not None else None
            grid = f", {first.get_count()}×{height}×{width} {dtype} "
            grid += f"EPSG:{epsg}" if epsg else "no CRS"
        except Exception:
            grid = ""
        return f"<EEOTimeSeries: {count} timesteps from {span}{grid}>"

    # ========================
    # Derivation
    # ========================
    def _derive(
        self, datasets: Sequence[EEORasterDataset], timestamps: Sequence[dt.datetime]
    ) -> EEOTimeSeries:
        """Build a series from this one's timesteps, keeping its backend choice.

        The scene cache is deliberately not carried over: the series that opened
        it owns it, and two owners would delete it twice.
        """
        derived = type(self)(datasets, timestamps=timestamps)
        derived._chunks = self._chunks
        return derived

    def map(
        self,
        op: Callable[..., EEORasterDataset],
        /,
        *,
        save_dir: StrPath | None = None,
        **kwargs: Any,
    ) -> EEOTimeSeries:
        """Apply one operation to every timestep, returning a new series.

        Any operation that takes a dataset first and returns a dataset works,
        unchanged — the spectral indices, the algebra, clipping, resampling,
        masking, or a function of your own. The same keyword arguments go to
        every timestep, which is what makes a trajectory comparable across
        time: one recipe, applied identically.

        Parameters
        ----------
        op : callable
            The operation itself, not its name: ``eeo.ndvi``, not ``"ndvi"``.
            Called as ``op(dataset, **kwargs)`` once per timestep, and must
            return an ``EEORasterDataset``.
        save_dir : str or path-like or None, default None
            Write each result to this directory and return a series reading
            those files, instead of holding every result in memory. Peak memory
            is then one result rather than all of them, which is what makes a
            long series mappable. Files are named by position and acquisition
            time, and an existing file of the same name is overwritten, as
            :meth:`~eeo.core.core.EEORasterDataset.save_raster` does. The
            directory is created if it does not exist.
        **kwargs
            Passed to ``op`` unchanged, for every timestep.

        Returns
        -------
        EEOTimeSeries
            New series with one result per timestep, in the same order and
            carrying this series' timestamps — including any that were supplied
            with ``timestamps=`` rather than read from the datasets.

        Raises
        ------
        ValidationError
            If ``op`` is a string (pass the function), is not callable, or
            returns anything other than an ``EEORasterDataset`` for some
            timestep.

        Notes
        -----
        Applies the operation immediately, timestep by timestep. Without
        ``save_dir`` the results are held in memory, so peak memory is the
        whole series' worth of results — fine for an area of interest, not for
        whole scenes. ``save_dir`` bounds it to one result, and a series opened
        on the lazy backend reopens its saved results there too, so the backend
        survives a chain.

        This series is left untouched: its datasets are the operation's inputs,
        never its outputs.

        Examples
        --------
        >>> ndvi_series = ts.map(eeo.ndvi, red="B04", nir="B08")  # doctest: +SKIP
        >>> masked = ts.map(eeo.mask_clouds).map(  # doctest: +SKIP
        ...     eeo.ndvi, red="B04", nir="B08", save_dir="ndvi/"
        ... )

        A function of your own is just as welcome:

        >>> doubled = ts.map(lambda ds: ds.multiply(2))  # doctest: +SKIP
        """
        if isinstance(op, str):
            raise ValidationError(
                f"map takes the operation itself, not its name: pass eeo.{op} rather "
                f"than {op!r} (or any function taking a dataset and returning one)"
            )
        if not callable(op):
            raise ValidationError(
                f"map needs a callable taking a dataset first and returning one; got "
                f"{type(op).__name__}"
            )

        directory: Path | None = None
        if save_dir is not None:
            directory = Path(os.fspath(save_dir))
            directory.mkdir(parents=True, exist_ok=True)

        results: list[EEORasterDataset] = []
        try:
            for index, ds in enumerate(self._datasets):
                result = _apply(op, ds, kwargs)
                if not isinstance(result, EEORasterDataset):
                    raise ValidationError(
                        f"map builds a series, so every result must be an "
                        f"EEORasterDataset; {getattr(op, '__name__', op)!r} returned "
                        f"{type(result).__name__} for timestep {index}. For an "
                        f"operation that returns a value rather than a raster, read "
                        f"the timesteps directly: [{getattr(op, '__name__', 'f')}(ds) "
                        f"for ds in ts]"
                    )
                if directory is not None:
                    result = _through_file(
                        result,
                        directory,
                        index,
                        self._chunks,
                        timestamp=self._timestamps[index],
                    )
                results.append(result)
        except BaseException:
            for result in results:
                with contextlib.suppress(Exception):
                    result.close()
            raise

        return self._derive(results, self._timestamps)

    # ========================
    # Series metadata
    # ========================
    @property
    def timestamps(self) -> list[dt.datetime]:
        """Acquisition times of the timesteps, in order.

        Returns
        -------
        list of datetime.datetime
            One timezone-aware UTC timestamp per timestep, oldest first.
        """
        return list(self._timestamps)

    @property
    def crs(self) -> CRS:
        """Coordinate reference system of the earliest timestep.

        Returns
        -------
        rasterio.crs.CRS
            The CRS the series is read in.
        """
        return self._datasets[0].get_crs()

    @property
    def transform(self) -> Affine:
        """Affine transform of the earliest timestep.

        Returns
        -------
        affine.Affine
            Mapping from pixel to world coordinates.
        """
        return self._datasets[0].get_transform()

    @property
    def shape(self) -> tuple[int, int]:
        """Pixel dimensions of the earliest timestep.

        Returns
        -------
        tuple of int
            ``(height, width)``.
        """
        return self._datasets[0].get_shape()

    @property
    def band_count(self) -> int:
        """Band count of the earliest timestep.

        Named ``band_count`` rather than ``count``, which a sequence already
        uses to count occurrences of a value.

        Returns
        -------
        int
            Number of bands each timestep is expected to hold.
        """
        return self._datasets[0].get_count()

    @property
    def band_names(self) -> list[str | None]:
        """Band names of the earliest timestep.

        Returns
        -------
        list of (str or None)
            One entry per band, ``None`` for an unnamed band.
        """
        return self._datasets[0].band_names

    # ========================
    # Lifecycle
    # ========================
    def close(self) -> None:
        """Release every timestep's resources, and the scene cache if owned.

        Returns
        -------
        None

        Notes
        -----
        Safe to call more than once. A series built by :meth:`from_stac` with
        the default ``cache=True`` owns a temporary directory, which this
        deletes — so a slice taken from it, which shares those files, stops
        working too. A cache directory you named yourself is left alone.
        """
        for ds in self._datasets:
            with contextlib.suppress(Exception):
                ds.close()
        if self._cache is not None:
            with contextlib.suppress(Exception):
                self._cache.cleanup()
            self._cache = None

    def __del__(self):
        """Best-effort close on garbage collection; errors are suppressed."""
        with contextlib.suppress(Exception):
            self.close()


def time_series(
    source: STACSearchResult | Iterable[STACItem] | Iterable[EEORasterDataset] | StrPath,
    assets: str | Sequence[str] | None = None,
    **kwargs: Any,
) -> EEOTimeSeries:
    """Build a time series from whatever holds the scenes.

    The one call to reach for: it reads what it was handed and delegates to the
    matching :class:`EEOTimeSeries` constructor, which stays available for
    anyone who prefers to name it.

    Parameters
    ----------
    source : STACSearchResult, iterable of STACItem, iterable of EEORasterDataset, or path
        The scenes. A search result or its items go to
        :meth:`EEOTimeSeries.from_stac`; datasets go to the
        :class:`EEOTimeSeries` constructor; a directory path goes to
        :meth:`EEOTimeSeries.from_folder`, which is not implemented yet.
    assets : str or sequence of str or None, default None
        Assets to read from each item. Required for a STAC source, rejected for
        datasets, which already hold their bands.
    **kwargs
        Keyword arguments of the constructor the source selects — ``bbox``,
        ``crop``, ``mask``, ``resampling``, ``cache`` and ``chunks`` for a STAC
        source, ``timestamps`` for datasets.

    Returns
    -------
    EEOTimeSeries
        Series over the scenes, oldest first.

    Raises
    ------
    ValidationError
        If ``source`` is not one of the accepted forms, is empty, mixes items
        and datasets, or is a STAC source without ``assets`` (or datasets
        with them). Also whatever the selected constructor rejects.
    NotImplementedError
        If ``source`` is a path: building a series from a folder is not
            implemented yet.

    Examples
    --------
    From a catalog search — the search result is already ordered and dated:

    >>> import eeo
    >>> results = eeo.stac_search(
    ...     "sentinel-2-l2a", bbox=(11.0, 46.5, 11.2, 46.7), limit=5
    ... )  # doctest: +SKIP
    >>> ts = eeo.time_series(results, assets=["B04", "B08"])  # doctest: +SKIP

    From datasets you loaded yourself:

    >>> import datetime as dt
    >>> import numpy as np
    >>> scenes = [
    ...     eeo.load_array(
    ...         np.full((4, 4), month, dtype="uint16"),
    ...         crs=32633,
    ...         timestamp=dt.datetime(2023, month, 1),
    ...     )
    ...     for month in (5, 4)
    ... ]
    >>> ts = eeo.time_series(scenes)
    >>> len(ts)
    2
    >>> ts.timestamps[0].date()
    datetime.date(2023, 4, 1)
    """
    if isinstance(source, (str, os.PathLike)):
        if assets is not None:
            raise ValidationError(
                "assets= names catalog assets to read and means nothing for a folder "
                "of rasters; drop it"
            )
        return EEOTimeSeries.from_folder(source, **kwargs)

    try:
        items = list(source)
    except TypeError as err:
        raise ValidationError(
            f"time_series takes a STAC search result, an iterable of STACItems, an "
            f"iterable of EEORasterDatasets, or a folder path; got "
            f"{type(source).__name__}"
        ) from err

    if not items:
        raise ValidationError("a time series needs at least one scene; got an empty collection")

    if all(isinstance(item, STACItem) for item in items):
        if assets is None:
            raise ValidationError(
                "reading from a catalog needs to know which assets to read, and there "
                "is no safe default — a Sentinel-2 item offers gigabytes across its "
                "bands. Pass assets=['B04', 'B08'] (or whatever the items offer, see "
                "STACItem.asset_names)"
            )
        # Narrowed by the isinstance check above; mypy cannot see that
        # through a list comprehension over `object`.
        return EEOTimeSeries.from_stac(cast("list[STACItem]", items), assets, **kwargs)

    if all(isinstance(item, EEORasterDataset) for item in items):
        if assets is not None:
            raise ValidationError(
                "assets= names catalog assets to read; these datasets are already "
                "loaded, so select their bands with band names instead"
            )
        return EEOTimeSeries(cast("list[EEORasterDataset]", items), **kwargs)

    kinds = sorted({type(item).__name__ for item in items})
    raise ValidationError(
        f"a time series is built from STACItems or from EEORasterDatasets, not a "
        f"mixture; got {', '.join(kinds)}"
    )
