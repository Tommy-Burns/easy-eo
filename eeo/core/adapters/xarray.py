"""Lazy, dask-chunked raster adapter backed by xarray."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
import rasterio as rio
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.transform import Affine, array_bounds
from rasterio.windows import Window

from eeo._optional import import_optional
from eeo.core.exceptions import BackendError, ValidationError
from eeo.core.remote import open_env
from eeo.core.types import ChunkSpec, StrPath

from .base import BaseRasterAdapter

_PURPOSE = "the lazy backend"
_DIMS = ("band", "y", "x")
_CHUNK_DIMS = frozenset(_DIMS)


def _import_rioxarray() -> Any:
    """Import rioxarray, which also registers the ``.rio`` accessor."""
    return import_optional("rioxarray", extra="lazy", purpose=_PURPOSE)


def validate_chunks(chunks: object) -> ChunkSpec:
    """Check a chunk specification before it reaches dask.

    Parameters
    ----------
    chunks : object
        Candidate chunk specification.

    Returns
    -------
    str or int or dict
        ``chunks`` unchanged, once validated.

    Raises
    ------
    ValidationError
        If ``chunks`` is not ``"auto"``, a positive int, or a dict mapping
        some of ``"band"``, ``"y"``, ``"x"`` to a positive int or ``"auto"``.
    """

    def valid_size(value: object) -> bool:
        if value == "auto":
            return True
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    if isinstance(chunks, dict):
        unknown = set(chunks) - _CHUNK_DIMS
        if not unknown and all(valid_size(v) for v in chunks.values()):
            return chunks
    elif valid_size(chunks):
        return chunks  # type: ignore[return-value]
    raise ValidationError(
        "chunks must be 'auto', a positive int, or a dict mapping 'band', 'y' "
        f"and/or 'x' to a positive int or 'auto'; got {chunks!r}"
    )


class XarrayAdapter(BaseRasterAdapter):
    """Lazy raster adapter over a dask-chunked :class:`xarray.DataArray`.

    Opening a raster through this adapter reads metadata only; pixels stay in
    the file until something asks for them, and then only the requested bands
    and window are computed. Needs the optional ``lazy`` extra
    (``pip install "easy-eo[lazy]"``).

    Prefer :meth:`from_path` over calling the constructor directly.

    Parameters
    ----------
    dataarray : xarray.DataArray
        Georeferenced array with dimensions ``("band", "y", "x")``, laid out
        as :func:`rioxarray.open_rasterio` returns it, holding raw stored values
        (not masked or scaled).
    driver : str, default "GTiff"
        GDAL driver of the source, reported in the metadata.
    source_path : str or path-like or None, default None
        File the array was opened from, when it was opened from one. It is
        what lets a lazy dataset be promoted to the rasterio backend by
        reopening the file rather than by reading every pixel.

    Raises
    ------
    MissingDependencyError
        If the ``lazy`` extra is not installed.
    ValidationError
        If ``dataarray`` does not have dimensions ``("band", "y", "x")``.
    """

    def __init__(
        self, dataarray: Any, *, driver: str = "GTiff", source_path: StrPath | None = None
    ) -> None:
        _import_rioxarray()
        if tuple(dataarray.dims) != _DIMS:
            raise ValidationError(
                f"dataarray must have dimensions {_DIMS}; got {tuple(dataarray.dims)}"
            )
        self._da = dataarray
        self._driver = driver
        self._source_path = source_path

    # ========================
    # Factories
    # ========================
    @classmethod
    def from_path(cls, path: StrPath, *, chunks: ChunkSpec = "auto") -> XarrayAdapter:
        """Open a raster file lazily, split into dask chunks.

        Parameters
        ----------
        path : str or path-like
            Path to a GDAL-readable raster.
        chunks : str or int or dict, default "auto"
            Chunk sizes, as :func:`rioxarray.open_rasterio` takes them.
            ``"auto"`` lets dask pick sizes aligned with the file's internal
            blocks.

        Returns
        -------
        XarrayAdapter
            Adapter over the unread raster.

        Raises
        ------
        MissingDependencyError
            If the ``lazy`` extra is not installed.
        ValidationError
            If ``chunks`` is not a valid chunk specification.
        BackendError
            If the file cannot be opened as a raster.
        """
        chunks = validate_chunks(chunks)
        rioxarray = _import_rioxarray()
        # dask is not imported by rioxarray itself; check it up front so a
        # missing package is reported as the extra, not as an xarray error.
        import_optional("dask.array", extra="lazy", purpose=_PURPOSE)
        try:
            with open_env(path):
                # rasterio supplies the driver, which rioxarray does not record.
                with rio.open(path) as src:
                    driver = src.driver
                dataarray = rioxarray.open_rasterio(path, chunks=chunks)
        except Exception as e:
            raise BackendError(f"failed to open raster lazily: {path}") from e
        return cls(dataarray, driver=driver, source_path=path)

    @property
    def source_path(self) -> StrPath | None:
        """File this array was opened from, or None if it was not opened from one."""
        return self._source_path

    # ========================
    # Metadata
    # ========================
    def get_crs(self) -> CRS:
        return self._da.rio.crs

    def get_transform(self) -> Affine:
        return self._da.rio.transform()

    def get_bounds(self) -> BoundingBox:
        height, width = self.get_shape()
        return BoundingBox(*array_bounds(height, width, self.get_transform()))

    def get_shape(self) -> tuple[int, int]:
        return self.get_height(), self.get_width()

    def get_width(self) -> int:
        return int(self._da.sizes["x"])

    def get_height(self) -> int:
        return int(self._da.sizes["y"])

    def get_count(self) -> int:
        return int(self._da.sizes["band"])

    def get_nodata(self) -> float | None:
        # rioxarray reports nodata as a scalar of the band dtype; rasterio
        # reports a float, and the rest of the library expects that.
        nodata = self._da.rio.nodata
        return None if nodata is None else float(nodata)

    def get_metadata(self) -> dict:
        return {
            "driver": self._driver,
            "dtype": str(self._da.dtype),
            "nodata": self.get_nodata(),
            "width": self.get_width(),
            "height": self.get_height(),
            "count": self.get_count(),
            "crs": self.get_crs(),
            "transform": self.get_transform(),
        }

    def get_band_descriptions(self) -> list[str | None]:
        # rioxarray stores GDAL band descriptions as ``long_name``: a plain
        # string for one band, a tuple otherwise, absent when none are set.
        long_name = self._da.attrs.get("long_name")
        if long_name is None:
            return [None] * self.get_count()
        if isinstance(long_name, str):
            long_name = (long_name,)
        return [(name or None) for name in long_name]

    # ========================
    # Data Access
    # ========================
    def read(
        self,
        indexes: int | Sequence[int] | None = None,
        *,
        window: Window | tuple | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Compute the requested bands and window as a NumPy array.

        Follows :meth:`rasterio.io.DatasetReader.read` for the arguments it
        supports: only the selected bands and window are computed, so reading
        a window of a scene larger than memory stays bounded.

        Parameters
        ----------
        indexes : int or sequence of int or None, default None
            1-based band index (returns a 2D array), a sequence of them, or
            ``None`` for every band (both return a 3D array).
        window : rasterio.windows.Window or tuple or None, default None
            Integer pixel window to read, or ``((row_start, row_stop),
            (col_start, col_stop))``. ``None`` reads the whole raster.
        **kwargs
            Any other rasterio read option. None is supported.

        Returns
        -------
        numpy.ndarray
            The requested pixels, as stored.

        Raises
        ------
        BackendError
            If a rasterio read option this backend does not support is given.
        IndexError
            If a band index is out of range.
        ValidationError
            If ``window`` has fractional offsets or sizes, or does not lie
            within the raster.
        """
        if kwargs:
            raise BackendError(
                f"the lazy backend's read() supports indexes and window only; "
                f"got {', '.join(sorted(kwargs))}"
            )
        bands: int | list[int] | slice
        if indexes is None:
            bands = slice(None)
        elif isinstance(indexes, (int, np.integer)):
            bands = self._band_position(int(indexes))
        else:
            bands = [self._band_position(int(i)) for i in indexes]

        rows, cols = self._window_slices(window)
        return self._compute(self._da.data[bands, rows, cols])

    def read_band(self, idx: int) -> np.ndarray:
        return self.read(idx)

    def _band_position(self, idx: int) -> int:
        """Map a 1-based band index to its 0-based array position."""
        count = self.get_count()
        if idx < 1 or idx > count:
            raise IndexError(
                f"band index {idx} out of range; dataset has {count} band(s) (valid 1..{count})"
            )
        return idx - 1

    def _window_slices(self, window: Window | tuple | None) -> tuple[slice, slice]:
        """Validate a pixel window and return its row and column slices."""
        if window is None:
            return slice(None), slice(None)
        win = window if isinstance(window, Window) else Window.from_slices(*window)
        bounds = (win.row_off, win.col_off, win.height, win.width)
        if any(float(v) != int(v) for v in bounds):
            raise ValidationError(f"window must have integer offsets and sizes; got {win}")
        row, col, height, width = (int(v) for v in bounds)
        if row < 0 or col < 0 or row + height > self.get_height() or col + width > self.get_width():
            raise ValidationError(
                f"window {win} does not lie within the raster "
                f"({self.get_height()} x {self.get_width()} pixels)"
            )
        return slice(row, row + height), slice(col, col + width)

    @staticmethod
    def _compute(data: Any) -> np.ndarray:
        """Materialise a selection as a NumPy array that owns its memory."""
        if hasattr(data, "compute"):
            return np.asarray(data.compute())
        # An in-memory DataArray hands out views of its own buffer; copy, so a
        # caller writing into the result cannot reach back into the dataset.
        return np.array(data, copy=True)

    # ========================
    # Persistence
    # ========================
    def write(
        self, path: StrPath, driver: str = "GTiff", band_names: list[str | None] | None = None
    ) -> None:
        # One dask chunk is computed and written at a time, so peak memory
        # follows the chunk size, not the raster size.
        meta = self.get_metadata()
        meta.update(driver=driver)
        data = self._da.data
        chunks = getattr(data, "chunks", None) or tuple((n,) for n in data.shape)

        with rio.open(path, "w", **meta) as dst:
            for band_s, row_s, col_s in itertools.product(*(_spans(c) for c in chunks)):
                block = self._compute(data[band_s, row_s, col_s])
                dst.write(
                    block,
                    indexes=list(range(band_s.start + 1, band_s.stop + 1)),
                    window=Window.from_slices(row_s, col_s),
                )
            # Unnamed bands are left alone so the file records no description.
            for i, name in enumerate(band_names or [], start=1):
                if name:
                    dst.set_band_description(i, name)

    def close(self) -> None:
        self._da.close()

    # ========================
    # Backend Access
    # ========================
    @property
    def backend(self) -> Any:
        return self._da


def _spans(sizes: Sequence[int]) -> list[slice]:
    """Turn one axis's chunk sizes into consecutive slices."""
    stops = np.cumsum(sizes).tolist()
    starts = [0, *stops[:-1]]
    return [slice(start, stop) for start, stop in zip(starts, stops, strict=True)]
