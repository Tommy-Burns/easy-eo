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
import logging
import os
import tempfile
import warnings
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, cast, overload

import rasterio as rio
from rasterio import CRS
from rasterio.transform import Affine
from rasterio.warp import reproject

from eeo.common import get_nodata, normalize_resampling_method, resolve_band_index
from eeo.core.adapters import RasterioAdapter, XarrayAdapter
from eeo.core.adapters.xarray import validate_chunks
from eeo.core.core import EEORasterDataset
from eeo.core.exceptions import AlignmentError, CRSMismatchError, ValidationError
from eeo.core.loader import load_raster
from eeo.core.types import ChunkSpec, ResamplingMethod, StrPath
from eeo.io.stac import STACItem, STACSearchResult
from eeo.preprocessing.masking import _find_quality_band, mask_clouds
from eeo.preprocessing.quality import QA_PIXEL_DEFAULT_MIN_CLOUD_CONFIDENCE
from eeo.timeseries import extract, reducers

_UTC = dt.timezone.utc

_LOGGER = logging.getLogger(__name__)

# Sentinel-2 processing baseline 04.00 was deployed on this date, and from it an
# L2A product carries BOA_ADD_OFFSET (-1000 DN for every band in practice), so
# the same ground reads about 1000 DN higher after it than before. Until
# Easy-EO decodes reflectance, a series spanning the date mixes two radiometric
# conventions, which is worth a warning rather than a silent bias.
# https://sentinels.copernicus.eu/-/imminent-deployment-of-sentinel-2-processing-baseline-04.00-on-25-january-2022
_BASELINE_04_00 = dt.datetime(2022, 1, 25, tzinfo=_UTC)

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


def _crs_of(ds: EEORasterDataset) -> CRS | None:
    """Return a dataset's CRS as a :class:`rasterio.crs.CRS`, or None if it has none.

    A NumPy-backed dataset hands back whatever was passed to
    :func:`eeo.load_array` — an EPSG int or a string, not necessarily a ``CRS``
    — so comparing two datasets' CRSs directly can report a mismatch between
    two spellings of the same system. Everything here compares this instead.
    """
    crs = ds.get_crs()
    if crs is None or isinstance(crs, CRS):
        return crs
    return CRS.from_user_input(crs)


def _same_grid(ds: EEORasterDataset, reference: EEORasterDataset) -> bool:
    """Report whether two datasets share a pixel grid exactly."""
    return (
        ds.get_shape() == reference.get_shape() and ds.get_transform() == reference.get_transform()
    )


def _onto_grid(
    ds: EEORasterDataset, reference: EEORasterDataset, *, method: str
) -> EEORasterDataset:
    """Warp one dataset onto another's exact grid: its CRS, transform and shape.

    Alignment across time has to land on the reference grid *exactly* —
    timesteps of one area differ in origin and extent, not merely in size, and a
    composite of rasters that merely share a shape would average different
    ground. So this warps to an explicit destination grid rather than resampling
    to a shape.

    The dtype and nodata value are the input's; the timestamp, attrs and band
    names are carried over, since this is called directly rather than through
    the operation decorator that would otherwise do it.
    """
    source = ds.to_rasterio()
    target_crs = _crs_of(reference)
    transform = reference.get_transform()
    height, width = reference.get_shape()
    nodata = get_nodata(source)
    resampling = normalize_resampling_method(method)

    meta = source.get_metadata()
    meta.update(crs=target_crs, transform=transform, width=width, height=height)

    def warp_bands(destination):
        for band in range(1, source.get_count() + 1):
            reproject(
                source=rio.band(source.ds, band),
                destination=rio.band(destination, band),
                src_transform=source.get_transform(),
                src_crs=_crs_of(source),
                dst_transform=transform,
                dst_crs=target_crs,
                src_nodata=nodata,
                dst_nodata=nodata,
                resampling=resampling,
            )

    return EEORasterDataset(
        adapter=RasterioAdapter.write_in_memory(meta, warp_bands),
        timestamp=ds.timestamp,
        attrs=ds.attrs,
        band_names=ds.band_names,
    )


def _check_bands(datasets: Sequence[EEORasterDataset], reference: int) -> None:
    """Refuse a series whose timesteps do not hold the same bands.

    Band count is never fixed automatically: a timestep with a different number
    of bands is a different measurement, not a misaligned one. Names are checked
    because they are how bands are addressed — if band 1 is "red" in one
    timestep and "nir" in another, every mapped index would be silently wrong.
    """
    ref = datasets[reference]
    ref_count = ref.get_count()
    ref_names = ref.band_names
    for index, ds in enumerate(datasets):
        if index == reference:
            continue
        if ds.get_count() != ref_count:
            raise ValidationError(
                f"timestep {index} has {ds.get_count()} bands but timestep {reference} "
                f"has {ref_count}; every timestep of a series must hold the same bands, "
                f"so load the same ones for each"
            )
        for position, (theirs, ours) in enumerate(
            zip(ds.band_names, ref_names, strict=True), start=1
        ):
            if theirs and ours and theirs != ours:
                raise ValidationError(
                    f"band {position} is {ours!r} in timestep {reference} but {theirs!r} "
                    f"in timestep {index}; a band must mean the same thing at every "
                    f"timestep, or an index computed across them addresses different "
                    f"data. Load the bands in the same order, or rename them to agree"
                )


def _on_one_grid(
    datasets: Sequence[EEORasterDataset],
    reference: int,
    *,
    auto_align: bool,
    auto_reproject: bool,
    method: str,
) -> list[EEORasterDataset]:
    """Return the timesteps on the reference's grid, aligning them if allowed.

    A CRS mismatch needs ``auto_reproject``, a grid mismatch ``auto_align``;
    either way the fix is the same warp onto the reference grid, because a
    reprojection that did not land on that grid would leave the series
    unaligned. Neither happens silently: without permission the mismatch is an
    error naming the flag that would fix it.
    """
    ref = datasets[reference]
    ref_crs = _crs_of(ref)
    aligned = list(datasets)
    reprojected: list[int] = []
    resampled: list[int] = []

    for index, ds in enumerate(datasets):
        if index == reference:
            continue
        if _crs_of(ds) != ref_crs:
            if not auto_reproject:
                raise CRSMismatchError(
                    f"timestep {index} is in {_crs_of(ds)} but timestep {reference} is "
                    f"in {ref_crs}; a series must be in one CRS. Pass "
                    f"auto_reproject=True to warp the others onto the reference's grid"
                )
            aligned[index] = _onto_grid(ds, ref, method=method)
            reprojected.append(index)
        elif not _same_grid(ds, ref):
            if not auto_align:
                raise AlignmentError(
                    f"timestep {index} is {ds.get_shape()} pixels on a different grid "
                    f"from timestep {reference}, which is {ref.get_shape()}; a series "
                    f"must share one grid so its timesteps can be compared pixel by "
                    f"pixel. Pass auto_align=True to resample the others onto the "
                    f"reference's grid"
                )
            aligned[index] = _onto_grid(ds, ref, method=method)
            resampled.append(index)

    # CODE_STYLE: an alignment that happens automatically says so.
    if reprojected:
        _LOGGER.info(
            "reprojected timesteps %s onto timestep %d's grid (%s) with method=%s",
            reprojected,
            reference,
            ref_crs,
            method,
        )
    if resampled:
        _LOGGER.info(
            "aligned timesteps %s onto timestep %d's grid with method=%s",
            resampled,
            reference,
            method,
        )
    return aligned


def _baseline_side(ds: EEORasterDataset, timestamp: dt.datetime) -> str | None:
    """Say which side of Sentinel-2 baseline 04.00 a timestep sits on.

    The product's own recorded baseline decides it where there is one, because
    that is a fact rather than an inference: the reprocessed archive carries
    baseline 04.00 or later on acquisitions from well before the switch, so the
    date alone would put those on the wrong side. Only when nothing is recorded
    does the acquisition date stand in. Returns None for anything that is not a
    Sentinel-2 scene, or whose mission is not recorded.
    """
    mission = ds.attrs.get("mission")
    if not (isinstance(mission, str) and mission.strip().casefold().startswith("sentinel-2")):
        return None

    recorded = ds.attrs.get("processing_baseline")
    if recorded is not None:
        try:
            value = float(str(recorded).strip())
        except ValueError:
            value = float("nan")
        if value == value:  # not NaN
            return "post" if value >= 4.0 else "pre"

    return "post" if timestamp >= _BASELINE_04_00 else "pre"


def _warn_on_mixed_baseline(
    datasets: Sequence[EEORasterDataset], timestamps: Sequence[dt.datetime]
) -> None:
    """Warn when a Sentinel-2 series straddles the baseline 04.00 change."""
    sides = [_baseline_side(ds, stamp) for ds, stamp in zip(datasets, timestamps, strict=True)]
    before = sides.count("pre")
    after = sides.count("post")
    if not (before and after):
        return
    warnings.warn(
        f"this series spans the Sentinel-2 processing baseline 04.00 change of "
        f"{_BASELINE_04_00.date()}: {before} timestep(s) sit before it and {after} "
        f"after. From baseline 04.00 a Level-2A product shifts its stored values by "
        f"BOA_ADD_OFFSET, -1000 DN for every band, so the same ground reads about "
        f"1000 DN higher after the change than before it. Easy-EO reads stored values "
        f"and does not decode reflectance, so a composite over this series is biased "
        f"by however many scenes fall on each side, and an index trajectory shows a "
        f"step at the boundary that is not in the ground. Keep the series on one side "
        f"of that date, or use scenes reprocessed to a single baseline.",
        UserWarning,
        stacklevel=4,
    )


def _chunks_of(datasets: Sequence[EEORasterDataset]) -> ChunkSpec | None:
    """Return the chunking a series' timesteps are already split into, if any.

    A series built on the lazy backend should stay there through a chain: when
    ``map(save_dir=...)`` writes a result out, the file is reopened with these
    chunk sizes rather than with rasterio, which would silently drop the backend
    the caller chose. Read from the timesteps rather than remembered from the
    call that built them, so it is right however the series was assembled.
    """
    for ds in datasets:
        adapter = ds._adapter
        if isinstance(adapter, XarrayAdapter):
            sizes = adapter.chunk_sizes
            if sizes is not None:
                # dict values are invariant to a type checker, so a
                # dict[str, int] is not a dict[str, int | Literal["auto"]]
                # even though every value it holds is valid in one.
                return cast("ChunkSpec", sizes)
            # Lazily backed but not chunked: dask still owns the reads, so keep
            # the backend and let it choose the sizes.
            return "auto"
    return None


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
    auto_align : bool, default False
        Whether a timestep on a different pixel grid from the reference may be
        resampled onto it. False refuses the series with an
        :class:`~eeo.AlignmentError` instead, because resampling a series is
        expensive and changes the pixels: opting in says you want that.
    auto_reproject : bool, default False
        Whether a timestep in a different CRS may be warped onto the
        reference's. False refuses with a :class:`~eeo.CRSMismatchError`. Both
        fixes are the same warp onto the reference grid; this flag is the
        permission for the CRS part, as ``mosaic`` spells it.
    method : str, default "nearest"
        Resampling method used when either flag triggers. Nearest by default
        rather than bilinear: a series built for cloud masking or compositing
        carries a quality band whose values are class numbers, and blending
        class numbers invents classes.
    reference : int, default 0
        Which timestep's CRS, grid and bands the others must match, indexed in
        time order — 0 is the earliest, -1 the latest. The default follows
        ``mosaic`` and ``stack``, where the first input sets the grid; a
        chronological series cannot be reordered to put a different scene
        first, which is what this is for.

    Raises
    ------
    ValidationError
        If ``datasets`` is empty, holds anything that is not an
        ``EEORasterDataset``, or has a timestep with no timestamp and no
        ``timestamps`` entry; if ``timestamps`` is given with a length that
        does not match; if ``reference`` is not a timestep; or if the timesteps
        do not hold the same number of bands, or disagree about what a band is
        called.
    CRSMismatchError
        If timesteps are in different CRSs and ``auto_reproject`` is False.
    AlignmentError
        If timesteps are on different pixel grids and ``auto_align`` is False.

    Warns
    -----
    UserWarning
        If a Sentinel-2 series spans the processing baseline 04.00 change of
        25 January 2022, across which the same ground reads about 1000 DN
        apart. Stored values are not decoded to reflectance, so a composite or
        trajectory over such a series carries a step that is not in the ground.

    Notes
    -----
    Holds whatever its datasets hold: a series of file-backed datasets keeps
    one GDAL handle per timestep and no pixels, while a series of in-memory
    datasets (what a catalog load returns) keeps every window in memory. This
    is why :meth:`from_stac` caches scenes to disk by default.

    Validation reads metadata only — no pixels — so refusing a mismatched
    series costs nothing. Alignment, when opted into, does read: each
    mismatched timestep is warped onto the reference grid there and then.

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
        auto_align: bool = False,
        auto_reproject: bool = False,
        method: str = "nearest",
        reference: int = 0,
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
        ordered = [items[index] for index in order]
        self._timestamps: list[dt.datetime] = [stamps[index] for index in order]

        if not -len(ordered) <= reference < len(ordered):
            raise ValidationError(
                f"reference={reference} is not a timestep of a {len(ordered)}-step "
                f"series; it indexes the series in time order, so 0 is the earliest"
            )
        reference %= len(ordered)
        self._reference = reference
        _check_bands(ordered, reference)
        self._datasets: list[EEORasterDataset] = _on_one_grid(
            ordered,
            reference,
            auto_align=auto_align,
            auto_reproject=auto_reproject,
            method=method,
        )
        _warn_on_mixed_baseline(self._datasets, self._timestamps)
        # Set by from_stac when it owns a temporary scene cache.
        self._cache: tempfile.TemporaryDirectory | None = None
        # A series on the lazy backend stays there through map(save_dir=),
        # which reopens each saved result with these chunk sizes.
        self._chunks: ChunkSpec | None = _chunks_of(self._datasets)

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
        auto_align: bool = False,
        auto_reproject: bool = False,
        method: str = "nearest",
        reference: int = 0,
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
        auto_align, auto_reproject, method, reference
            Grid consistency across timesteps, as :class:`EEOTimeSeries`
            documents them. Worth knowing for a catalog search: items covering
            one area can land in different UTM zones, and a search wide enough
            to cross a zone boundary needs ``auto_reproject=True``.

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
        CRSMismatchError
            If the items are in different CRSs and ``auto_reproject`` is False.
        AlignmentError
            If the items land on different grids and ``auto_align`` is False.
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
            series = cls(
                scenes,
                auto_align=auto_align,
                auto_reproject=auto_reproject,
                method=method,
                reference=reference,
            )
        except BaseException:
            for scene in scenes:
                with contextlib.suppress(Exception):
                    scene.close()
            if holder is not None:
                with contextlib.suppress(Exception):
                    holder.cleanup()
            raise

        series._cache = holder
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
    def reference(self) -> EEORasterDataset:
        """The timestep whose CRS, grid and bands the others match.

        Returns
        -------
        EEORasterDataset
            The timestep chosen by ``reference`` at construction — the earliest
            by default. Every other timestep was checked against it, and
            aligned onto it where that was allowed.
        """
        return self._datasets[self._reference]

    @property
    def crs(self) -> CRS | None:
        """Coordinate reference system of the series.

        Returns
        -------
        rasterio.crs.CRS or None
            The one CRS every timestep is in, or None if the reference timestep
            declares none. Always a ``CRS``, even where a timestep was built
            from an EPSG code or a string.
        """
        return _crs_of(self.reference)

    @property
    def transform(self) -> Affine:
        """Affine transform of the series' grid.

        Returns
        -------
        affine.Affine
            Mapping from pixel to world coordinates, shared by every timestep.
        """
        return self.reference.get_transform()

    @property
    def shape(self) -> tuple[int, int]:
        """Pixel dimensions of the series' grid.

        Returns
        -------
        tuple of int
            ``(height, width)``, shared by every timestep.
        """
        return self.reference.get_shape()

    @property
    def band_count(self) -> int:
        """Number of bands each timestep holds.

        Named ``band_count`` rather than ``count``, which a sequence already
        uses to count occurrences of a value.

        Returns
        -------
        int
            Band count, the same at every timestep.
        """
        return self.reference.get_count()

    @property
    def band_names(self) -> list[str | None]:
        """Band names of the series.

        Returns
        -------
        list of (str or None)
            One entry per band, ``None`` for a band the reference timestep does
            not name. No two timesteps may disagree about a name, so these
            describe the series.
        """
        return self.reference.band_names

    # ========================
    # Temporal reducers
    # ========================
    def median(self, *, save_path: StrPath | None = None) -> EEORasterDataset:
        """Collapse the series to the median of every pixel across time.

        The reducer to reach for on a stack of scenes: a median over time is
        what turns repeat coverage into one clean image, because a cloud, a
        shadow or a sensor artefact at one timestep is an outlier among the
        others rather than a vote.

        Parameters
        ----------
        save_path : str or path-like or None, default None
            Write the result to this path instead of holding it in memory.

        Returns
        -------
        EEORasterDataset
            Single float32 raster on the series' grid, with its bands and band
            names, holding each pixel's median over time. A pixel that no
            timestep saw is NaN, which is the result's recorded nodata value.

        Notes
        -----
        Streams window by window, with the block divided by the number of
        timesteps, so peak memory is about one block's worth in total however
        long the series is. Each timestep's nodata pixels are absent from the median
        rather than counted, so a pixel missing at two of five timesteps is the
        median of the other three; only a pixel missing everywhere is nodata.
        float32 because a median over an even number of timesteps averages the
        two middle values.

        Examples
        --------
        >>> composite = ts.median()  # doctest: +SKIP
        >>> ts.map(eeo.mask_clouds).median(save_path="composite.tif")  # doctest: +SKIP
        """
        return reducers.median(self, save_path=save_path)

    def mean(self, *, save_path: StrPath | None = None) -> EEORasterDataset:
        """Collapse the series to the mean of every pixel across time.

        Parameters
        ----------
        save_path : str or path-like or None, default None
            Write the result to this path instead of holding it in memory.

        Returns
        -------
        EEORasterDataset
            Single float32 raster on the series' grid, with its bands and band
            names, holding each pixel's mean over time. A pixel that no timestep
            saw is NaN, the result's recorded nodata value.

        Notes
        -----
        Streams window by window, with the block divided by the number of
        timesteps, so peak memory does not grow with the series' length. Nodata
        pixels are absent from the mean rather than counted as zero, so each
        pixel is averaged over however many timesteps actually saw it. Prefer
        :meth:`median` over a series that may hold cloud: a mean is pulled by
        outliers, a median is not.

        Examples
        --------
        >>> average = ts.mean()  # doctest: +SKIP
        """
        return reducers.mean(self, save_path=save_path)

    def min(self, *, save_path: StrPath | None = None) -> EEORasterDataset:
        """Collapse the series to the smallest value of every pixel across time.

        Parameters
        ----------
        save_path : str or path-like or None, default None
            Write the result to this path instead of holding it in memory.

        Returns
        -------
        EEORasterDataset
            Single raster on the series' grid, with its bands and band names, in
            the timesteps' own dtype — a minimum selects a value that was
            measured rather than computing a new one. A pixel that no timestep
            saw takes the timesteps' nodata value, or none if they declare none,
            in which case no pixel can be missing.

        Notes
        -----
        Streams window by window, with the block divided by the number of
        timesteps, so peak memory does not grow with the series' length. Nodata
        pixels are absent from the comparison, so a fill value can never win it.

        Examples
        --------
        >>> darkest = ts.min()  # doctest: +SKIP
        """
        return reducers.minimum(self, save_path=save_path)

    def max(self, *, save_path: StrPath | None = None) -> EEORasterDataset:
        """Collapse the series to the largest value of every pixel across time.

        Parameters
        ----------
        save_path : str or path-like or None, default None
            Write the result to this path instead of holding it in memory.

        Returns
        -------
        EEORasterDataset
            Single raster on the series' grid, with its bands and band names, in
            the timesteps' own dtype — a maximum selects a value that was
            measured rather than computing a new one. A pixel that no timestep
            saw takes the timesteps' nodata value, or none if they declare none.

        Notes
        -----
        Streams window by window, with the block divided by the number of
        timesteps, so peak memory does not grow with the series' length. Nodata
        pixels are absent from the comparison. A maximum over an index series is
        the usual way to ask "how green did this ever get", one reason the
        reducers return a plain dataset that the rest of the library can chain.

        Examples
        --------
        >>> peak_greenness = ts.map(eeo.ndvi, red="red", nir="nir").max()  # doctest: +SKIP
        """
        return reducers.maximum(self, save_path=save_path)

    def composite(
        self,
        *,
        how: str = "median",
        mask_band: int | str | None = None,
        classes: Iterable[Any] | None = None,
        flags: Iterable[Any] | None = None,
        min_cloud_confidence: Any = QA_PIXEL_DEFAULT_MIN_CLOUD_CONFIDENCE,
        mission: int | None = None,
        nodata: int | float | None = None,
        mask_dir: StrPath | None = None,
        save_path: StrPath | None = None,
    ) -> EEORasterDataset:
        """Build one cloud-free raster from the series: mask, then reduce.

        The reason a time series is worth having. Each timestep is masked with
        its own quality band — Sentinel-2's ``SCL`` or Landsat's ``QA_PIXEL``,
        whichever it carries — and the masked timesteps are then reduced pixel
        by pixel across time. Where one scene was clouded another usually was
        not, so the result is a view of the ground assembled from whichever
        timestep saw it, rather than any single acquisition.

        The quality band is **not** in the result. It has done its work, and a
        median of scene-class numbers would be a number no classifier ever
        assigned.

        Parameters
        ----------
        how : {"median", "mean", "min", "max"}, default "median"
            Statistic taken across the masked timesteps. Median by default:
            a cloud edge or a missed cloud at one timestep is an outlier among
            the others, which a median discards and a mean averages in.
        mask_band : int or str, optional
            Which band holds the quality layer, as a 1-based index or a band
            name. Defaults to the one band named ``"scl"`` or ``"qa_pixel"``;
            having none, or more than one, is an error rather than a guess, and
            is raised before any pixel is read.
        classes : iterable of SCLClass or int or str, optional
            For an ``SCL`` band: which scene classes to mask, defaulting to
            :data:`~eeo.preprocessing.quality.SCL_DEFAULT_MASKED`.
        flags : iterable of QAPixelFlag or int or str, optional
            For a ``QA_PIXEL`` band: which flags to mask on, defaulting to
            :data:`~eeo.preprocessing.quality.QA_PIXEL_DEFAULT_MASKED`.
        min_cloud_confidence : QAConfidence or int or None, optional
            For a ``QA_PIXEL`` band: also mask pixels whose cloud confidence
            reaches this level, Medium by default.
        mission : int, optional
            For a ``QA_PIXEL`` band: which Landsat took the scenes, when the
            loaders did not record it.
        nodata : int or float, optional
            Value a masked pixel is set to in each timestep, defaulting to the
            raster's own. It marks pixels as absent, so it never reaches the
            result: what the result records is the reducer's own nodata.
        mask_dir : str or path-like or None, default None
            Write the masked timesteps to this directory instead of holding
            them in memory. Masking reads a whole scene, so without this the
            series' worth of masked scenes is held at once; with it, one is.
        save_path : str or path-like or None, default None
            Write the composite to this path instead of holding it in memory.

        Returns
        -------
        EEORasterDataset
            Single raster on the series' grid holding every band except the
            quality band, named as the series names them. float32 with NaN
            where no timestep saw the ground clear for ``"median"`` and
            ``"mean"``; the timesteps' own dtype, with their nodata value
            there, for ``"min"`` and ``"max"``. Carries no timestamp, and
            records the reduction and the time span it covers in ``attrs``.

        Raises
        ------
        ValidationError
            If no quality band can be found or more than one is present; if the
            series holds nothing but a quality band; if ``classes`` is given
            for a ``QA_PIXEL`` band or ``flags`` for an ``SCL`` band; if a
            ``QA_PIXEL`` series records no mission and none is given; or if the
            timesteps are an integer type declaring no nodata, leaving no value
            a masked pixel could take.

        Notes
        -----
        Equivalent to ``ts.map(eeo.mask_clouds, ...)`` followed by the reducer,
        minus the quality band — spelled as one call because it is the workflow
        a series exists for. Do it by hand when a timestep's mask lives in a
        separate raster, which this does not cover.

        A pixel clouded at every timestep is the one the composite cannot fill;
        it comes back as nodata rather than as whatever the cloud looked like.

        Examples
        --------
        >>> results = eeo.stac_search(  # doctest: +SKIP
        ...     "sentinel-2-l2a", bbox=AOI, datetime="2023-04-01/2023-09-30"
        ... )
        >>> ts = eeo.time_series(results, assets=["B04", "B08", "SCL"])  # doctest: +SKIP
        >>> clear = ts.composite()  # doctest: +SKIP
        >>> clear.band_names  # doctest: +SKIP
        ['B04', 'B08']
        >>> ndvi = clear.ndvi(red="B04", nir="B08")  # doctest: +SKIP
        """
        quality = (
            _find_quality_band(self.reference)[0]
            if mask_band is None
            else resolve_band_index(self.reference, mask_band)
        )
        data_bands = [band for band in range(1, self.band_count + 1) if band != quality]
        if not data_bands:
            raise ValidationError(
                f"this series holds only its quality band (band {quality}), so there "
                f"is nothing to composite. Load the spectral bands alongside it, e.g. "
                f"assets=['B04', 'B08', 'SCL']"
            )

        masked = self.map(
            mask_clouds,
            mask_band=quality,
            classes=classes,
            flags=flags,
            min_cloud_confidence=min_cloud_confidence,
            mission=mission,
            nodata=nodata,
            save_dir=mask_dir,
        )
        try:
            return reducers.reduce_series(masked, how, bands=data_bands, save_path=save_path)
        finally:
            # The composite is its own raster, so the masked timesteps have
            # done their work; closing frees them now rather than at collection.
            # Files written to mask_dir are the caller's and are left alone.
            masked.close()

    # ========================
    # Extraction
    # ========================
    def extract_at(
        self,
        coordinates: Sequence[float],
        *,
        bands: Sequence[int | str] | None = None,
        crs: Any = None,
    ) -> Any:
        """Sample one location at every timestep, as a table indexed by time.

        The counterpart to the reducers: where they collapse time into one
        raster, this collapses space into one trajectory — what happened *here*,
        in the shape pandas and matplotlib already understand.

        Parameters
        ----------
        coordinates : sequence of float
            ``(x, y)`` position, in the series' CRS unless ``crs`` says
            otherwise.
        bands : sequence of (int or str) or None, default None
            Which bands to sample, as 1-based indices or band names; None
            samples every band.
        crs : optional
            CRS the coordinates are given in — anything rasterio accepts, such
            as ``"EPSG:4326"`` — when that is not the series' own. Saves
            transforming lon/lat by hand after a catalog search.

        Returns
        -------
        pandas.DataFrame
            One row per timestep, indexed by a ``DatetimeIndex`` named ``time``,
            with one float column per sampled band named after that band
            (``band_<n>`` for an unnamed one). A pixel that was nodata at a
            timestep is ``NaN`` there rather than its fill value, so a gap in
            the trajectory reads as a gap. ``attrs`` records the point sampled.

        Raises
        ------
        ValidationError
            If ``coordinates`` does not hold exactly two values, if the point
            falls outside the series' extent, if ``crs`` is given for a series
            that declares none, or if ``bands`` names a band the series does not
            have.

        Notes
        -----
        Reads one pixel per timestep and band — never a band, never a scene, so
        a trajectory over a season of full tiles costs a few dozen pixels.

        Examples
        --------
        >>> trajectory = ts.extract_at((11.1, 46.6), crs="EPSG:4326")  # doctest: +SKIP
        >>> ndvi = ts.map(eeo.ndvi, red="B04", nir="B08")  # doctest: +SKIP
        >>> ndvi.extract_at((11.1, 46.6), crs="EPSG:4326").plot()  # doctest: +SKIP
        """
        return extract.extract_at(self, coordinates, bands=bands, crs=crs)

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
