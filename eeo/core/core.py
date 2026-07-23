"""Core functionality for Easy-EO."""

from __future__ import annotations

import contextlib
from datetime import datetime

import numpy as np
import rasterio as rio
from rasterio import CRS
from rasterio.coords import BoundingBox
from rasterio.transform import Affine

from eeo.common import is_rasterio_backed, mask_nodata, resolve_band_index
from eeo.core.adapters import BaseRasterAdapter, NumpyRasterioAdapter, RasterioAdapter
from eeo.core.exceptions import ValidationError

# Approximate (decimated) statistics never read more than this many pixels per
# side; a larger raster is decimated to fit, served from overviews when present.
_STATS_DECIMATION_CAP = 1024


# IO helper
def _save_raster(dataset: rio.DatasetReader, path: str, driver: str = "GTiff") -> None:
    """Write a rasterio dataset to ``path`` using its own profile."""
    profile = dataset.profile.copy()
    if driver != "GTiff":
        profile.update(driver=driver)
    with rio.open(path, mode="w", **profile) as dst:
        dst.write(dataset.read())


def _resolve_stats_mode(stats: bool | str) -> str | None:
    """Map the ``describe`` stats argument to a mode, or None for no stats."""
    if stats is False:
        return None
    if stats is True or stats == "approx":
        return "approx"
    if stats == "exact":
        return "exact"
    raise ValidationError(f"stats must be False, True, 'approx', or 'exact'; got {stats!r}")


def _format_crs(crs) -> str:
    """Render a CRS as ``EPSG:<code> — <name>`` when possible."""
    if crs is None:
        return "none"
    epsg = crs.to_epsg()
    try:
        import pyproj

        name = pyproj.CRS.from_user_input(crs).name
    except Exception:
        name = None
    if epsg and name:
        return f"EPSG:{epsg} — {name}"
    if epsg:
        return f"EPSG:{epsg}"
    return name or str(crs)


def _num(value: float) -> str:
    """Format a coordinate/resolution without scientific notation or trailing zeros."""
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _decimated_stats_shape(
    shape: tuple[int, int], cap: int = _STATS_DECIMATION_CAP
) -> tuple[int, int] | None:
    """Return a decimated ``(height, width)`` capped at ``cap`` per side.

    Returns None when the raster already fits within the cap and should be read
    at full resolution.
    """
    height, width = shape
    scale = min(cap / height, cap / width, 1.0)
    if scale >= 1.0:
        return None
    return max(1, round(height * scale)), max(1, round(width * scale))


def _band_label(ds: EEORasterDataset, band_idx: int) -> str:
    """Label a band by number, appending its name when it has one."""
    name = ds.band_names[band_idx - 1]
    return f"band {band_idx}" if name is None else f"band {band_idx} ({name})"


def _band_names_row(ds: EEORasterDataset) -> str:
    """Render the band-name list for ``describe``, or "none" when unnamed."""
    names = ds.band_names
    if not any(names):
        return "none"
    return ", ".join(f"{i}: {name or '—'}" for i, name in enumerate(names, start=1))


def _band_stats_line(
    ds: EEORasterDataset, band_idx: int, array, approximate: bool, width: int = 11
) -> str:
    """Build one nodata-aware per-band statistics line for ``describe``."""
    label = _band_label(ds, band_idx)
    masked = mask_nodata(ds, array)
    if np.issubdtype(masked.dtype, np.floating):
        valid = ~np.isnan(masked)
    else:  # no nodata declared -> every pixel is valid
        valid = np.ones(masked.shape, dtype=bool)

    n_valid = int(valid.sum())
    if n_valid == 0:
        return f"  {label:<{width}} : all nodata"

    with np.errstate(all="ignore"):
        vmin, vmax = float(np.nanmin(masked)), float(np.nanmax(masked))
        vmean, vstd = float(np.nanmean(masked)), float(np.nanstd(masked))
    pct = 100.0 * n_valid / masked.size

    m = "~" if approximate else ""

    def f(value: float) -> str:
        return f"{value:.6g}"

    line = (
        f"  {label:<{width}} : min{m} {f(vmin)}   max{m} {f(vmax)}   "
        f"mean{m} {f(vmean)}   std{m} {f(vstd)}   valid{m} {pct:.1f}%"
    )
    if not approximate:
        line += f" ({masked.size - n_valid:,} nodata)"
    return line


def _stats_lines(ds: EEORasterDataset, mode: str) -> list[str]:
    """Build the statistics block of ``describe`` (may read pixel data)."""
    out_shape = None
    if mode == "approx" and is_rasterio_backed(ds):
        out_shape = _decimated_stats_shape(ds.get_shape(), _STATS_DECIMATION_CAP)
    approximate = out_shape is not None

    if approximate:
        height, width = out_shape
        header = f"approximate — decimated read at {height} × {width} (set stats='exact' for exact)"
    else:
        header = "exact — full read"

    # Named bands make the labels longer, so size the label column to the
    # widest one and keep the ` : ` separators aligned.
    labels = [_band_label(ds, i) for i in range(1, ds.get_count() + 1)]
    width = max(11, *(len(label) for label in labels)) if labels else 11

    lines = ["", f"  {'statistics':<{width}} : {header}"]
    for band_idx in range(1, ds.get_count() + 1):
        array = ds.read(band_idx, out_shape=out_shape) if approximate else ds.get_band(band_idx)
        lines.append(_band_stats_line(ds, band_idx, array, approximate, width))
    return lines


def _describe_text(ds: EEORasterDataset, stats: bool | str) -> str:
    """Build the human-readable description printed by ``describe``."""
    mode = _resolve_stats_mode(stats)
    meta = ds.get_metadata()

    def row(label: str, value: object) -> str:
        return f"  {label:<11} : {value}"

    height, width = ds.get_shape()
    transform = ds.get_transform()
    # Positional unpack: the rasterio adapter returns a BoundingBox namedtuple,
    # the NumPy adapter a plain (left, bottom, right, top) tuple.
    left, bottom, right, top = ds.get_bounds()
    nodata = meta.get("nodata")
    attrs = ", ".join(f"{k}={v}" for k, v in ds.attrs.items()) if ds.attrs else "none"

    lines = [
        "EEORasterDataset",
        row("source", ds.path or "<in-memory>"),
        row("driver", meta.get("driver", "unknown")),
        row("bands", ds.get_count()),
        row("band names", _band_names_row(ds)),
        row("size", f"{height} × {width}  (height × width)"),
        row("dtype", meta.get("dtype", "unknown")),
        row("crs", _format_crs(ds.get_crs())),
        row("pixel size", f"{_num(abs(transform.a))} × {_num(abs(transform.e))}  (CRS units)"),
        row(
            "extent",
            f"{_num(left)}, {_num(bottom)}, {_num(right)}, {_num(top)}  (minx, miny, maxx, maxy)",
        ),
        row("nodata", "none" if nodata is None else nodata),
        row("timestamp", ds.timestamp.isoformat() if ds.timestamp is not None else "none"),
        row("attrs", attrs),
    ]

    if mode is not None:
        lines.extend(_stats_lines(ds, mode))
    return "\n".join(lines)


def _normalize_band_name(name: str | None) -> str | None:
    """Normalize one band name: strip whitespace, treat blank as ``None``."""
    if name is None:
        return None
    if isinstance(name, str):
        return name.strip() or None
    raise ValidationError(f"band name must be a str or None; got {type(name).__name__}")


def _normalize_band_names(names, count: int) -> list[str | None]:
    """Validate a band-name sequence against ``count`` and normalize each entry."""
    names = list(names)
    if len(names) != count:
        raise ValidationError(
            f"band_names must have one entry per band; expected {count}, got {len(names)}"
        )
    return [_normalize_band_name(n) for n in names]


def _resolve_initial_band_names(adapter: BaseRasterAdapter, band_names) -> list[str | None]:
    """Seed band names from an explicit list, else from the backend descriptions."""
    count = adapter.get_count()
    if band_names is None:
        return _normalize_band_names(adapter.get_band_descriptions(), count)
    return _normalize_band_names(band_names, count)


# Core class
class EEORasterDataset:
    """A chainable raster dataset backed by a swappable adapter.

    Wraps a raster (rasterio- or NumPy-backed through ``BaseRasterAdapter``)
    and exposes metadata accessors plus the chainable operations bound by the
    ``@eeo_raster_op`` / ``@eeo_raster_viz`` decorators. Construct one with
    :func:`eeo.load_raster`, :func:`eeo.load_array`, or the ``from_*``
    classmethods rather than calling ``__init__`` directly.

    Alongside the pixel data, a dataset carries two optional provenance fields
    that survive every operation: ``timestamp`` (an acquisition time) and
    ``attrs`` (a free-form tags dict). Chainable operations copy them onto
    their result, so metadata set once at load time follows the data through a
    whole processing chain.
    """

    def __init__(
        self,
        adapter: BaseRasterAdapter,
        path: str | None = None,
        *,
        timestamp: datetime | None = None,
        attrs: dict | None = None,
        band_names: list[str | None] | None = None,
    ):
        """Wrap a backend adapter.

        Parameters
        ----------
        adapter : BaseRasterAdapter
            Backend providing pixel access and metadata. Prefer the ``from_*``
            constructors or :func:`eeo.load_raster` / :func:`eeo.load_array`
            over calling this directly.
        path : str or None, default None
            Source path, when the dataset came from a file.
        timestamp : datetime.datetime or None, default None
            Optional acquisition time carried with the dataset and preserved
            through operations.
        attrs : dict or None, default None
            Optional free-form tags dict carried with the dataset and
            preserved through operations. Copied on assignment so datasets do
            not share a mutable dict.
        band_names : list of (str or None) or None, default None
            Optional per-band names, one entry per band (``None`` for an
            unnamed band). When omitted, names are seeded from the backend's
            band descriptions (all ``None`` for the NumPy backend). Must match
            the band count.
        """
        self._adapter = adapter
        self.path = path
        self.timestamp = timestamp
        self.attrs: dict = {} if attrs is None else dict(attrs)
        self._band_names: list[str | None] = _resolve_initial_band_names(adapter, band_names)

    def __repr__(self) -> str:
        """Return a concise one-line summary for REPLs and logs."""
        try:
            count = self.get_count()
            height, width = self.get_shape()
            dtype = self.get_metadata().get("dtype", "?")
            crs = self.get_crs()
            epsg = crs.to_epsg() if crs is not None else None
            crs_str = f"EPSG:{epsg}" if epsg else "no CRS"
            names = self._band_names_summary()
            return f"<EEORasterDataset {count}×{height}×{width} {dtype} {crs_str}{names}>"
        except Exception:
            return "<EEORasterDataset (unavailable)>"

    def _band_names_summary(self, limit: int = 4) -> str:
        """Render band names for ``__repr__``, elided past ``limit`` entries."""
        names = self._band_names
        if not any(names):
            return ""
        shown = [name or "—" for name in names[:limit]]
        if len(names) > limit:
            shown.append("…")
        return " [" + ", ".join(shown) + "]"

    # ========================
    # Constructors
    # ========================
    @classmethod
    def from_path(cls, path: str) -> EEORasterDataset:
        """Open a raster file as a rasterio-backed dataset.

        Parameters
        ----------
        path : str
            Path to a GDAL-readable raster.

        Returns
        -------
        EEORasterDataset
            Rasterio-backed dataset; pixels are read lazily.
        """
        adapter = RasterioAdapter.from_path(path)
        return cls(adapter=adapter, path=path)

    @classmethod
    def from_rasterio(cls, dataset: rio.DatasetReader) -> EEORasterDataset:
        """Wrap an already-open rasterio dataset.

        Parameters
        ----------
        dataset : rasterio.io.DatasetReader
            Open rasterio dataset to wrap.

        Returns
        -------
        EEORasterDataset
            Rasterio-backed dataset.
        """
        return cls(adapter=RasterioAdapter(dataset))

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        transform: Affine,
        crs: CRS | str | int,
        driver: str = "GTiff",
        nodata=None,
        *,
        timestamp: datetime | None = None,
        attrs: dict | None = None,
        band_names: list[str | None] | None = None,
    ) -> EEORasterDataset:
        """Build a NumPy-backed dataset from an array and georeferencing.

        Parameters
        ----------
        array : numpy.ndarray
            Raster values, ``(height, width)`` or ``(bands, height, width)``.
        transform : affine.Affine
            Affine geotransform (pixel-to-world mapping).
        crs : rasterio.crs.CRS or str or int
            Coordinate reference system.
        driver : str, default "GTiff"
            Driver recorded for when the dataset is later written or promoted.
        nodata : float or int or None, default None
            Value marking nodata pixels.
        timestamp : datetime.datetime or None, default None
            Optional acquisition time carried with the dataset.
        attrs : dict or None, default None
            Optional free-form tags dict carried with the dataset.
        band_names : list of (str or None) or None, default None
            Optional per-band names, one entry per band. Must match the band
            count.

        Returns
        -------
        EEORasterDataset
            NumPy-backed dataset.
        """
        adapter = NumpyRasterioAdapter(
            array=array,
            transform=transform,
            crs=crs,
            nodata=nodata,
            driver=driver,
        )
        return cls(adapter=adapter, timestamp=timestamp, attrs=attrs, band_names=band_names)

    # ========================
    # Conversion between adapters
    # ========================

    def to_rasterio(self) -> EEORasterDataset:
        """Return an equivalent rasterio-backed dataset.

        Returns
        -------
        EEORasterDataset
            ``self`` if it is already rasterio-backed, otherwise a new
            in-memory rasterio dataset with the same pixels and metadata.

        Notes
        -----
        Promoting a NumPy-backed dataset reads its full array into an
        in-memory rasterio ``MemoryFile``. Band names, ``timestamp``, and
        ``attrs`` are carried onto the promoted dataset.

        Examples
        --------
        >>> rio_ds = ds.to_rasterio()
        """
        # Detect by adapter type, not backend class: an op result's backend is
        # a rasterio DatasetWriter, which a DatasetReader isinstance check
        # would wrongly re-promote (full read + copy).
        if isinstance(self._adapter, RasterioAdapter):
            return self

        array = self.read()
        transform = self.get_transform()
        crs = self.get_crs()
        nodata = self._adapter.get_nodata()

        adapter = RasterioAdapter.from_array(
            array=array,
            transform=transform,
            crs=crs,
            nodata=nodata,
        )
        return EEORasterDataset(
            adapter=adapter,
            timestamp=self.timestamp,
            attrs=self.attrs,
            band_names=self.band_names,
        )

    def to_array(self) -> np.ndarray:
        """Read the raster into a NumPy array.

        Returns
        -------
        numpy.ndarray
            Array shaped ``(bands, height, width)``. Nodata pixels are
            returned as their stored sentinel value.

        Examples
        --------
        >>> arr = ds.to_array()
        """
        return self.read()

    # ========================
    # Metadata
    # ========================

    def read(self, *args, **kwargs) -> np.ndarray:
        """Read pixel data, forwarding all arguments to the backend.

        For the rasterio backend, the arguments are
        ``rasterio.DatasetReader.read`` options (band indexes, ``out_shape``,
        ``window``, ...). The NumPy backend returns its stored array.

        Returns
        -------
        numpy.ndarray
            The requested pixels, ``(bands, height, width)`` by default.
        """
        return self._adapter.read(*args, **kwargs)

    def get_crs(self) -> CRS:
        """Return the coordinate reference system.

        Returns
        -------
        rasterio.crs.CRS
            The raster's CRS.
        """
        return self._adapter.get_crs()

    def get_transform(self) -> Affine:
        """Return the affine geotransform.

        Returns
        -------
        affine.Affine
            Pixel-to-world transform.
        """
        return self._adapter.get_transform()

    def get_shape(self) -> tuple[int, int]:
        """Return the raster shape.

        Returns
        -------
        tuple of int
            ``(height, width)`` in pixels.
        """
        return self._adapter.get_shape()

    def get_bounds(self) -> BoundingBox:
        """Return the spatial bounds.

        Returns
        -------
        rasterio.coords.BoundingBox
            ``(left, bottom, right, top)`` in CRS units.
        """
        return self._adapter.get_bounds()

    def get_metadata(self) -> dict:
        """Return the raster metadata profile.

        Returns
        -------
        dict
            Metadata including driver, dtype, nodata, transform, crs, count,
            width, and height.
        """
        return self._adapter.get_metadata()

    def describe(self, *, stats: bool | str = False) -> None:
        """Print a human-readable description of the raster.

        Always shows structural metadata (source, driver, band count, band
        names, size, dtype, CRS, pixel size, extent, nodata, and the
        ``timestamp`` / ``attrs`` provenance fields) without reading any
        pixels. Optionally appends per-band statistics, whose rows are labelled
        ``band 4 (red)`` for a named band.

        Parameters
        ----------
        stats : bool or str, default False
            Controls the optional statistics block:

            - ``False`` — structural metadata only; no pixel data is read.
            - ``"approx"`` (or ``True``) — per-band statistics from a decimated
              read (served from overviews when present), fast and memory-safe
              on large scenes. Values are marked with ``~`` and are
              approximate: a decimated read can miss the true extremes.
            - ``"exact"`` — per-band statistics from a full read; exact but
              reads every pixel.

            A raster small enough to sit under the decimation cap, and every
            NumPy-backed dataset, is read in full even for ``"approx"`` and the
            block is then labelled exact.

        Returns
        -------
        None
            The description is printed to standard output.

        Raises
        ------
        ValidationError
            If ``stats`` is not one of ``False``, ``True``, ``"approx"``, or
            ``"exact"``.

        Notes
        -----
        Statistics exclude nodata pixels. ``stats="exact"`` reads the whole
        raster; ``stats="approx"`` reads a decimated array capped at
        ``1024`` pixels per side.

        Examples
        --------
        >>> ds.describe()
        >>> ds.describe(stats="approx")
        """
        print(_describe_text(self, stats))

    def get_width(self) -> int:
        """Return the raster width in pixels.

        Returns
        -------
        int
            Number of columns.
        """
        return self._adapter.get_width()

    def get_height(self) -> int:
        """Return the raster height in pixels.

        Returns
        -------
        int
            Number of rows.
        """
        return self._adapter.get_height()

    def get_count(self) -> int:
        """Return the number of bands.

        Returns
        -------
        int
            Band count.
        """
        return self._adapter.get_count()

    def get_index(self):
        """Return the backend's coordinate-to-pixel index method.

        Returns
        -------
        callable
            The underlying rasterio dataset's ``index`` method, mapping
            ``(x, y)`` world coordinates to ``(row, col)``. Available only on
            rasterio-backed datasets.
        """
        return self.ds.index

    def get_band(self, idx: int | str) -> np.ndarray:
        """Read a single band, addressed by index or by name.

        Parameters
        ----------
        idx : int or str
            1-based band index, or a band name declared in
            :attr:`band_names`. A string is always a name: ``"4"`` matches a
            band literally named ``"4"``, never band 4.

        Returns
        -------
        numpy.ndarray
            The band as a 2D array. Nodata pixels keep their sentinel value.

        Raises
        ------
        IndexError
            If ``idx`` is outside the range of available bands.
        ValidationError
            If ``idx`` is a name that is unknown or declared on more than one
            band.

        Examples
        --------
        >>> nir = ds.get_band(8)
        >>> nir = ds.get_band("nir")
        """
        return self._adapter.read_band(resolve_band_index(self, idx))

    @property
    def band_names(self) -> list[str | None]:
        """Per-band names, one entry per band (``None`` for an unnamed band).

        Seeded from the backend's band descriptions at load time and preserved
        in memory. Assigning a new list replaces all names at once (it must
        match the band count); assign ``None`` to clear every name. Use
        :meth:`set_band_name` to rename a single band. Names are written to the
        raster's GDAL band descriptions when the dataset is saved.

        Returns
        -------
        list of (str or None)
            A copy of the band names; mutate via assignment or
            :meth:`set_band_name`, not by editing the returned list in place.
        """
        return list(self._band_names)

    @band_names.setter
    def band_names(self, names: list[str | None] | None) -> None:
        """Replace all band names, or clear them when ``names`` is ``None``."""
        if names is None:
            self._band_names = [None] * self.get_count()
        else:
            self._band_names = _normalize_band_names(names, self.get_count())

    def set_band_name(self, band: int, new_name: str | None) -> None:
        """Rename a single band by its 1-based index.

        Parameters
        ----------
        band : int
            1-based index of the band to rename.
        new_name : str or None
            New name for the band; a blank or whitespace-only string, or
            ``None``, clears the name.

        Raises
        ------
        IndexError
            If ``band`` is outside the range of available bands.
        ValidationError
            If ``new_name`` is neither a string nor ``None``.

        Examples
        --------
        >>> ds.set_band_name(4, "red")
        """
        count = self.get_count()
        if isinstance(band, bool) or not isinstance(band, int) or band < 1 or band > count:
            raise IndexError(
                f"band index {band!r} out of range; dataset has {count} band(s) (valid 1..{count})"
            )
        self._band_names[band - 1] = _normalize_band_name(new_name)

    # ========================
    # Saving
    # ========================
    def save_raster(self, path: str, driver: str = "GTiff") -> None:
        """Write the raster to disk.

        Parameters
        ----------
        path : str
            Output file path.
        driver : str, default "GTiff"
            GDAL driver name for the output format.

        Returns
        -------
        None

        Notes
        -----
        Band names are written to the output's GDAL band descriptions, so they
        are read back automatically by :func:`eeo.load_raster`. Formats that
        cannot store band descriptions simply drop them.

        Examples
        --------
        >>> ds.save_raster("out.tif")
        """
        self._adapter.write(path=path, driver=driver, band_names=self.band_names)

    # ========================
    # Lifecycle
    # ========================
    def close(self) -> None:
        """Release the backend's file handles and resources.

        Returns
        -------
        None

        Notes
        -----
        Safe to call more than once. A dataset created from an in-memory
        ``MemoryFile`` (e.g. any operation result) cannot be reopened after
        closing.
        """
        self._adapter.close()

    def __del__(self):
        """Best-effort close on garbage collection; errors are suppressed."""
        with contextlib.suppress(Exception):
            self.close()

    # ========================
    # Constructors
    # ========================
    def _bind(self, func):
        """Wrap an external function as a bound, chainable method."""

        def method(*args, **kwargs):
            result = func(*args, **kwargs)
            return self if result is None else result

        return method

    # ========================
    # Adapter access
    # ========================
    @property
    def ds(self):
        """Underlying backend object (rasterio dataset or NumPy array).

        Returns
        -------
        rasterio.io.DatasetReader or numpy.ndarray
            The raw backend. Accessing it bypasses Easy-EO's abstractions; use
            the typed accessors where possible.
        """
        return self._adapter.backend

    # ========================
    # Arithmetic Operators
    # ========================
    def __add__(self, other: EEORasterDataset | int | float) -> EEORasterDataset:
        """Return ``self + other`` (delegates to :meth:`add`)."""
        return self.add(other)

    def __radd__(self, other: int | float) -> EEORasterDataset:
        """Return ``other + self`` (delegates to :meth:`add`)."""
        return self.add(other)

    def __sub__(self, other: EEORasterDataset | int | float) -> EEORasterDataset:
        """Return ``self - other`` (delegates to :meth:`subtract`)."""
        return self.subtract(other)

    def __rsub__(self, other: int | float) -> EEORasterDataset:
        """Return ``other - self`` for scalar ``other``."""
        # implement for only raster - scalar
        if isinstance(other, (int, float)):
            return self.multiply(-1).add(other)
        return NotImplemented

    def __mul__(self, other: EEORasterDataset | int | float) -> EEORasterDataset:
        """Return ``self * other`` (delegates to :meth:`multiply`)."""
        return self.multiply(other)

    def __rmul__(self, other: int | float) -> EEORasterDataset:
        """Return ``other * self`` (delegates to :meth:`multiply`)."""
        return self.multiply(other)

    def __truediv__(self, other: EEORasterDataset | int | float) -> EEORasterDataset:
        """Return ``self / other`` (delegates to :meth:`divide`)."""
        return self.divide(other)

    def __rtruediv__(self, other: int | float) -> EEORasterDataset:
        """Return ``other / self`` for scalar ``other``."""
        # implement scalar / raster
        if isinstance(other, (int, float)):
            return self.power(-1).multiply(other)
        return NotImplemented

    def __pow__(self, exponent: int | float) -> EEORasterDataset:
        """Return ``self ** exponent`` (delegates to :meth:`power`)."""
        return self.power(exponent)
