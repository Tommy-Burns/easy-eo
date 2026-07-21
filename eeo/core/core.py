"""Core functionality for Easy-EO."""

from __future__ import annotations

import contextlib

import numpy as np
import rasterio as rio
from rasterio import CRS
from rasterio.coords import BoundingBox
from rasterio.transform import Affine

from eeo.core.adapters import BaseRasterAdapter, NumpyRasterioAdapter, RasterioAdapter


# IO helper
def _save_raster(dataset: rio.DatasetReader, path: str, driver: str = "GTiff") -> None:
    """Write a rasterio dataset to ``path`` using its own profile."""
    profile = dataset.profile.copy()
    if driver != "GTiff":
        profile.update(driver=driver)
    with rio.open(path, mode="w", **profile) as dst:
        dst.write(dataset.read())


# Core class
class EEORasterDataset:
    """A chainable raster dataset backed by a swappable adapter.

    Wraps a raster (rasterio- or NumPy-backed through ``BaseRasterAdapter``)
    and exposes metadata accessors plus the chainable operations bound by the
    ``@eeo_raster_op`` / ``@eeo_raster_viz`` decorators. Construct one with
    :func:`eeo.load_raster`, :func:`eeo.load_array`, or the ``from_*``
    classmethods rather than calling ``__init__`` directly.
    """

    def __init__(self, adapter: BaseRasterAdapter, path: str | None = None):
        """Wrap a backend adapter.

        Parameters
        ----------
        adapter : BaseRasterAdapter
            Backend providing pixel access and metadata. Prefer the ``from_*``
            constructors or :func:`eeo.load_raster` / :func:`eeo.load_array`
            over calling this directly.
        path : str or None, default None
            Source path, when the dataset came from a file.
        """
        self._adapter = adapter
        self.path = path

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
        return cls(adapter=adapter)

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
        in-memory rasterio ``MemoryFile``.

        Examples
        --------
        >>> rio_ds = ds.to_rasterio()
        """
        backend = self._adapter.backend

        # already a rasterio backend
        if isinstance(backend, rio.DatasetReader):
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
        return EEORasterDataset(adapter=adapter)

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

    def get_band(self, idx: int) -> np.ndarray:
        """Read a single band.

        Parameters
        ----------
        idx : int
            1-based band index.

        Returns
        -------
        numpy.ndarray
            The band as a 2D array. Nodata pixels keep their sentinel value.

        Raises
        ------
        IndexError
            If ``idx`` is outside the range of available bands.
        """
        return self._adapter.read_band(idx)

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

        Examples
        --------
        >>> ds.save_raster("out.tif")
        """
        self._adapter.write(path=path, driver=driver)

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
