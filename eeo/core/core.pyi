from typing import Union, Optional, Iterable, Sequence

import geopandas as gpd
import numpy as np
import rasterio as rio
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine

from eeo.analysis.stats import Coordinate
from eeo.core.adapters import BaseRasterAdapter
from eeo.core.types import ResamplingMethod


def _save_raster(dataset: rio.DatasetReader, path: str, driver: str = "GTiff") -> None:
    """
    Save a rasterio dataset to a file.

    Args:
        dataset (rio.DatasetReader): The raster dataset to save.
        path (str): The file path where the dataset will be saved.
        driver (str, optional): The raster driver format. Defaults to "GTiff".
    """
    ...

class EEORasterDataset:
    """
    Core class for handling raster datasets with preprocessing capabilities.
    """

    def __init__(self, adapter: BaseRasterAdapter, path: str | None = None) -> None:
        """
        Initialize an EEORasterDataset from a file path.

        Args:
            path (str): Path to the raster file.
        """
        self._adapter = adapter
        ...

    @classmethod
    def from_path(cls, path: str) -> "EEORasterDataset":
        ...

    @classmethod
    def from_rasterio(cls, dataset: rio.DatasetReader) -> "EEORasterDataset":
        ...

    @classmethod
    def from_array(
            cls,
            array: np.ndarray,
            transform: Affine,
            crs: Union[CRS, str, int],
            driver: str = "GTiff",
            nodata=None,
    ) -> "EEORasterDataset":
        ...

    def to_rasterio(self) -> "EEORasterDataset":
        """
        Convert the dataset to a Rasterio-backed EEORasterDataset.

        If the dataset is already backed by rasterio, it is returned.
        NumPy-backed datasets are promoted to an in-memory rasterio dataset.
        """
        ...

    def to_array(self) -> np.ndarray:
        """
        Return the raster data as a NumPy array.

        For multiband rasters, the returned array has shape
        ``(bands, height, width)``.
        """
        ...

    def get_crs(self) -> rio.crs.CRS:
        """
        Get the coordinate reference system of the raster.

        Returns:
            rio.crs.CRS: The raster's CRS.
        """
        ...

    def get_transform(self) -> rio.Affine:
        """
        Get the affine transform of the raster.

        Returns:
            rio.Affine: The raster's affine transform.
        """
        ...

    def get_shape(self) -> tuple[int, int]:
        """
        Get the shape (height, width) of the raster.

        Returns:
            tuple[int, int]: The raster's shape.
        """
        ...

    def get_bounds(self) -> rio.coords.BoundingBox:
        """
        Get the bounds of the raster.

        Returns:
            rio.coords.BoundingBox: The raster's bounding box.
        """
        ...

    def get_metadata(self) -> dict:
        """
        Get the metadata of the raster dataset.

        Returns:
            dict: The raster's metadata.
        """
        ...

    def get_band(self, idx: int) -> np.ndarray:
        """
        Read a specific band from the raster dataset.

        Args:
            idx (int): The index of the band to read (1-based).

        Returns:
            np.ndarray: The raster band as a NumPy array.
        """
        ...

    def get_width(self) -> int:
        """
        Get the width of the raster dataset.

        Returns:
            int: The width of the raster dataset.
        """
        ...

    def get_height(self) -> int:
        """
        Get the height of the raster dataset.

        Returns:
            int: The height of the raster dataset.
        """
        ...

    def get_count(self) -> int:
        """
        Get the number of bands of the raster dataset.

        Returns:
            int: The number of bands of the raster dataset.
        """
        ...

    def get_index(self):
        """
        Get the index
        """
        ...

    def save_raster(self, path: str, driver: str = "GTiff") -> None:
        """
            Save a rasterio dataset to a file.

            Args:
                path (str): The file path where the dataset will be saved.
                driver (str, optional): The raster driver format. Defaults to "GTiff".
            """
        ...

    def close(self) -> None:
        """
        Close the underlying Rasterio dataset.

        This method explicitly releases file handles, GDAL resources, and any
        associated memory held by the underlying ``DatasetReader``. Calling it
        is recommended once the dataset is no longer needed, particularly when
        opening many files in a workflow.

        Notes
        -----
        - Safe to call multiple times.
        - If the dataset was created from a ``MemoryFile`` (via
          RasterDatasetFromRasterio), closing it makes it unavailable for
          reopening.
        """

    def __del__(self):
        """
        Attempt to close the dataset during object destruction.

        This acts as a safety fallback if ``close()`` was not called explicitly.
        Because ``__del__`` may run during interpreter shutdown, and the order of
        garbage collection is not guaranteed, all exceptions must be suppressed.

        Notes
        -----
        • ``__del__`` is not guaranteed to run immediately.
        • You should not rely on this as the primary cleanup mechanism. Calling ``close()`` is safer instead.
        """

    # ========================
    # Adapter access
    # ========================
    @property
    def ds(self):
        return self._adapter.backend


    def read(self, *args, **kwargs):
        """Forward rasterio.read"""
        ...


    # Arithmetic Dunder Overloads
    def __add__(
            self,
            other: Union["EEORasterDataset", int, float]
    ) -> "EEORasterDataset":
        """Pixel-wise raster addition."""

    def __radd__(
            self,
            other: Union[int, float]
    ) -> "EEORasterDataset":
        """Reflected pixel-wise addition."""

    def __sub__(
            self,
            other: Union["EEORasterDataset", int, float]
    ) -> "EEORasterDataset":
        """Pixel-wise raster subtraction."""

    def __rsub__(
            self,
            other: Union[int, float]
    ) -> "EEORasterDataset":
        """Scalar minus raster."""

    def __mul__(
            self,
            other: Union["EEORasterDataset", int, float]
    ) -> "EEORasterDataset":
        """Pixel-wise raster multiplication."""

    def __rmul__(
            self,
            other: Union[int, float]
    ) -> "EEORasterDataset":
        """Reflected pixel-wise multiplication."""

    def __truediv__(
            self,
            other: Union["EEORasterDataset", int, float]
    ) -> "EEORasterDataset":
        """Pixel-wise raster division."""

    def __rtruediv__(
            self,
            other: Union[int, float]
    ) -> "EEORasterDataset":
        """Scalar divided by raster."""

    def __pow__(
            self,
            exponent: Union[int, float]
    ) -> "EEORasterDataset":
        """Pixel-wise exponentiation."""


    # The following methods are add to this .pyi file for IDE autocompletion
    # They are not strictly needed for the functioning of the methods.
    # The eeo_raster_op decorator already binds all the functions to this class

    # ===================================
    # ANALYSIS MODULE
    # ===================================
    def normalized_difference(
            self,
            other: EEORasterDataset,
            *,
            auto_align: bool = True,
            method: str = "bilinear",
            return_as_ndarray: bool = False
    ) -> Union[np.ndarray, EEORasterDataset]:
        """
            Compute the normalized difference index: (ds - other) / (ds + other)


            :param ds: EEORasterDataset
                First raster (e.g. NIR).
            :param other: EEORasterDataset
                Second raster (e.g. Red).
            :param auto_align: bool, optional
                Automatically resample `other` to match `ds`.
            :param method: str, optional
                Resampling method used if auto-aligning.
            :param return_as_ndarray: bool, optional
                If True, return a NumPy array instead of an EEORasterDataset.

            :return: Union[np.ndarray, EEORasterDataset]
        """
        ...

    def extract_value_at_coordinate(
            self,
            coordinates: Coordinate,
            band_idx: int = 1,
    ) -> Union[int, float]:
        """
        Extract the pixel value from an EEORasterDataset at a given geographic coordinate.

        Parameters
        ----------
        ds : EEORasterDataset
            The raster dataset from which to extract a value.
        coordinates : Coordinate
            A tuple of (x, y) coordinates. Must contain exactly two values.
        band_idx : int, optional
            The index of the raster band to sample from in case of multiband raster. Defaults to 1.

        Returns
        -------
        ``int`` or ``float``
            The extracted pixel value from the specified band and coordinate.

        Raises
        ------
        ValueError
            If the provided coordinates do not contain exactly two elements.
        """
        ...

    def get_maximum_pixel(
            self,
            band_idx: int = 1,
            *,
            return_position_as_pixel_coordinate: bool = False,
    ) -> dict:
        """
        Get the maximum pixel value and its location from a raster band.

        :param ds: Input raster dataset.
        :param band_idx: Band index to evaluate (1-based).
        :param return_position_as_pixel_coordinate: If True, return (row, col) instead of spatial coordinates.

        :return: Dictionary containing the maximum value and its position.
        :rtype: dict[str, float | tuple]
        """
        ...

    def get_minimum_pixel(
            self,
            band_idx: int = 1,
            *,
            return_position_as_pixel_coordinate: bool = False,
    ) -> dict:
        """
        Get the minimum pixel value and its location from a raster band.

        :param ds: Input raster dataset.
        :param band_idx: Band index to evaluate (1-based).
        :param return_position_as_pixel_coordinate: If True, return (row, col) instead of spatial coordinates.

        :return: Dictionary containing the minimum value and its position.
        :rtype: dict[str, float | tuple]
        """
        ...

    def get_mean_pixel(
            self,
            band_idx: int = 1,
            *,
            return_position_as_pixel_coordinate: bool = False,
    ) -> dict:
        """
        Get the pixel whose value is closest to the mean of the raster band.

        :param ds: Input raster dataset.
        :param band_idx: Band index to evaluate (1-based).
        :param return_position_as_pixel_coordinate: If True, return (row, col) instead of spatial coordinates.

        :return: Dictionary containing the mean value and its position.
        """
        ...

    def get_percentile_pixel(
            self,
            percentile: float,
            band_idx: int = 1,
            *,
            return_position_as_pixel_coordinate: bool = False,
    ) -> dict:
        """
        Get the pixel corresponding to a given percentile of raster values.

        :param ds: Input raster dataset.
        :param percentile: Percentile in the range [0, 100].
        :param band_idx: Band index to evaluate (1-based).
        :param return_position_as_pixel_coordinate: If True, return (row, col) instead of spatial coordinates.
        :return: Dictionary containing the percentile value and its position.
        """
        ...
    # ===================================
    # OPERATIONS MODULE
    # ===================================
    def add(
            self,
            other: Union[EEORasterDataset, float, int],
            *,
            auto_align: bool = True,
            method: str = "bilinear"
    ) -> EEORasterDataset:
        """
        Pixel-wise addition of two rasters or a raster and a scalar.

        If `other` is an EEORasterDataset and its shape or spatial alignment
        differs from `ds`, the raster may be automatically resampled to match
        `ds` before performing the operation.


        :param ds: EEORasterDataset
            The base raster dataset.
        :param other: EEORasterDataset | float | int
            Raster or scalar value to add.
        :param auto_align: bool, default=True
            Whether to automatically resample `other` to match `ds` if their
            shapes or transforms differ.
        :param method: str, default="bilinear"
            Resampling method used when auto-alignment is required.

        :return: EEORasterDataset
            A new raster dataset containing the result.
        """
        ...

    def subtract(
            self,
            other: Union[EEORasterDataset, float, int],
            *,
            auto_align: bool = True,
            method: str = "bilinear"
    ) -> EEORasterDataset:
        """
        Pixel-wise subtraction of two rasters or a raster and a scalar.

        The result is computed as `ds - other`.

        Auto-alignment behavior is identical to `add()`.

        Returns
        -------
        EEORasterDataset
        """
        ...

    def multiply(
            self,
            other: Union[EEORasterDataset, float, int],
            *,
            auto_align: bool = True,
            method: str = "bilinear"
    ) -> EEORasterDataset:
        """
        Pixel-wise multiplication of two rasters or a raster and a scalar.

        Auto-alignment behavior is identical to `add()`.

        Returns
        -------
        EEORasterDataset
        """
        ...

    def divide(
            self,
            other: Union[EEORasterDataset, float, int],
            *,
            auto_align: bool = True,
            method: str = "bilinear",
            safe: bool = True
    ) -> EEORasterDataset:
        """
        Pixel-wise division of two rasters or a raster and a scalar.

        If `safe=True`, division by zero and invalid values are handled safely
        by suppressing warnings and replacing NaNs with zeros.

        Parameters
        ----------
        safe : bool, default=True
            Whether to enable safe division handling.

        Returns
        -------
        EEORasterDataset
        :param ds:
        """
        ...

    def power(
            self,
            exponent: Union[int, float]
    ) -> EEORasterDataset:
        """
        Pixel-wise exponentiation of raster values.

        Each pixel value is raised to the given scalar exponent.

        Returns
        -------
        EEORasterDataset
        """
        ...

    def sqrt(
            self
    ) -> EEORasterDataset:
        """
        Pixel-wise square root of raster values.

        Negative values are clipped to zero prior to computation.

        Returns
        -------
        EEORasterDataset
        """
        ...

    def log(
            self,
            base: Union[int, float] = ...
    ) -> EEORasterDataset:
        """
        Pixel-wise logarithm of raster values.

        Zero and negative values are handled safely by clamping to a small
        positive constant prior to computation.

        Parameters
        ----------
        base : int | float, default=e
            Logarithmic base.

        Returns
        -------
        EEORasterDataset
        """
        ...

    def absolute(
            self
    ) -> EEORasterDataset:
        """
        Pixel-wise absolute value of raster values.

        Returns
        -------
        EEORasterDataset
        """
        ...

    def mosaic(
            self,
            others: list[EEORasterDataset],
            *,
            resampling_method: str = "nearest",
            auto_reproject: bool = False,
            **kwargs,
    ) -> EEORasterDataset | None:
        """
        Mosaic one or more rasters into a single raster.

        Parameters
        ----------
        ds : EEORasterDataset
            Base raster.
        others : EEORasterDataset or Iterable[EEORasterDataset]
            One or more rasters to mosaic with `ds`.
        resampling_method : str, optional
            Resampling method used during merge.
        auto_reproject : bool, optional
            Reproject rasters to match `ds` CRS if needed.
        """
        ...

    def stack(
            self,
            others: Union[EEORasterDataset, Iterable[EEORasterDataset]],
    ) -> EEORasterDataset:
        """
            Stack rasters into a multi-band raster.

            All rasters must have identical CRS, transform, and shape.
        """
        ...

    # ===================================
    # PREPROCESSING MODULE
    # ===================================
    def clip_raster_with_vector(
            self,
            vector_file: Union[gpd.GeoDataFrame, str],
            *,
            crop: bool = True,
            pad: bool = False,
            all_touched: bool = False,
            invert: bool = False,
            nodata: Union[int, float, None] = None,
            show_preview: bool = False,
            plot_kwargs: dict | None = None,
    ) -> "EEORasterDataset":
        """
        Clip a raster using vector geometries.

        The raster can be cropped to the geometry bounds or masked while preserving
        the original extent. Supports advanced Rasterio masking options via keyword arguments.

        :param ds: The EEORasterDataset object.
        :param vector_file: (GeoDataFrame or str)
            Vector geometries for clipping. If a string is provided, it must be a
            valid path to a vector file readable by GeoPandas.
        :param crop: (bool, default=True)
            ``True``: Crop the output raster to the minimal bounding box of the geometries.
            ``False``: Retain the original raster extent; pixels outside the geometries are set to nodata.
        :param pad: (bool, default=False)
            If ``crop=True``, expands the output extent to fully include edge pixels.
        :param all_touched: (bool, default=False)
            ``True``: Include all pixels touched by geometries.
            ``False``: Include only pixels whose center lies within geometries.
        :param invert: (bool, default=False)
            ``True``: Mask pixels inside the geometries instead of outside.
        :param nodata: (int or float, optional)
            Value assigned to masked pixels. If ``None``, uses the raster’s existing nodata value.
        :param show_preview: (bool, default=False)
            If ``True``, displays a preview of the clipped raster.
        :param plot_kwargs: (dict, optional)
            Additional keyword arguments passed to rasterio.plot.show. Common options:
                - title (str)
                - cmap (str or Colormap)
                - vmin, vmax (float)
                - ax (matplotlib.axes.Axes)
        :param **kwargs: Additional keyword arguments forwarded to rasterio.mask.mask,
            excluding parameters already exposed explicitly (crop, pad, all_touched, invert, nodata).

        :return: Union[None, EEORasterDataset]
            Returns ``None`` if saved to disk, otherwise returns a new EEORasterDataset.
        """
        ...

    def clip_raster_with_bbox(
            self,
            bbox: tuple | list,
            plot_kwargs=None,
            show_preview: bool = False
    ) -> Union[None, "EEORasterDataset"]:
        """
        Clip a raster using a bounding box.

        The raster will be subsetted to the specified bounding box. The bounding
        box must be in the same CRS as the raster.

        :param ds: The EEORasterDataset object.
        :param bbox: (tuple or list)
            Bounding box coordinates as (minx, miny, maxx, maxy)..
        :param show_preview: (bool, default=False)
            If ``True``, displays a preview of the clipped raster.
        :param plot_kwargs: (dict, optional)
            Additional keyword arguments passed to rasterio.plot.show. Common options:
                - title (str)
                - cmap (str or Colormap)
                - vmin, vmax (float)
                - ax (matplotlib.axes.Axes)

        :return: Union[None, EEORasterDataset]
            Returns ``None`` if saved to disk, otherwise returns a new EEORasterDataset.
        """
        ...

    def standardize(
            self
    ) -> EEORasterDataset:
        """
        Z-score standardization of raster values.

        The transformation is defined as:
            (x - mean) / standard_deviation

        Returns
        -------
        EEORasterDataset
        """
        ...

    def normalize_min_max(
            self,
            *,
            new_min: Union[int, float] = ...,
            new_max: Union[int, float] = ...
    ) -> EEORasterDataset:
        """
        Min-max normalization of raster values.

        Values are linearly rescaled from the original data range
        to the interval [new_min, new_max].

        Returns
        -------
        EEORasterDataset
        """
        ...

    def normalize_percentile(
            self,
            *,
            lower_percentile: Union[int, float] = ...,
            upper_percentile: Union[int, float] = ...
    ) -> EEORasterDataset:
        """
        Percentile-based normalization of raster values.

        The lower and upper percentiles are computed using NaN-aware
        percentile estimation, and values are clipped to [0, 1].

        This method is robust to outliers and commonly used for
        visualization and normalization of skewed data.

        Parameters
        ----------
        lower_percentile : int | float, default=0.0
            Lower percentile threshold (0–100).
        upper_percentile : int | float, default=1.0
            Upper percentile threshold (0–100).

        Returns
        -------
        EEORasterDataset
        """
        ...

    def reproject_raster(
            self,
            *,
            target_crs: Union[int, str, CRS],
            resampling_method: Union[Resampling, ResamplingMethod] = "nearest",
    ) -> EEORasterDataset:
        """
        Reproject a raster to a new CRS.


        :param ds: EEORasterDataset
            Input raster dataset.

        :param target_crs: int | str | pyproj.CRS
            Target CRS (EPSG code, proj string, or a CRS object).

        :param resampling_method: rasterio.enums.Resampling | Literal
            The resampling method used during reprojection. May be either a
            `Resampling` enum value or one of the following string literals:
            ``"nearest", "bilinear", "cubic", "cubic_spline", "lanczos", "average",
            "mode", "max", "min", "med", "q1", "q3"``.
            Default is ``nearest``.



        :return: ``EEORasterDataset``
        """
        ...

    def resample(self,
                 *,
                 size: Optional[tuple[int, int]] = None,
                 scale_factor: Optional[float] = None,
                 resolution: Optional[tuple[float, float]] = None,
                 resampling_method: Union[Resampling, ResamplingMethod] = "nearest",
                 plot_kwargs=None,
                 show_preview: bool = False
                 ) -> EEORasterDataset:
        """
        Resample a raster to a different resolution using a scaling factor

        :param ds: The EEORasterDataset object.
        :param size: tuple[int, int]. Output pixel dimensions.
        :param scale_factor: float. Uniform scale factor (e.g. 0.5 halves the resolution).
        :param resolution: tuple[float, float]. Output spatial resolution (xres, yres) in CRS units.
        :param resampling_method: ``rasterio.enums.Resampling`` | Literal
            The resampling method used during reprojection. May be either a
            `Resampling` enum value or one of the following string literals:
            ``"nearest", "bilinear", "cubic", "cubic_spline", "lanczos", "average",
            "mode", "max", "min", "med", "q1", "q3"``.
            Default is ``nearest``.
        :param show_preview: (str, optional): Raster driver to use when saving. Defaults to "GTiff".
        :param plot_kwargs: Additional keyword arguments passed to rasterio.plot.show.
            Common options include:
                - title (str): Title of the plot.
                - cmap (str or Colormap): Colormap to use.
                - vmin, vmax (float): Value range for display.
                - ax (matplotlib.axes.Axes): Axes to plot on.
        :return: EEORasterDataset
        """
        ...

    # ===================================
    # VISUALIZATION MODULE
    # ===================================
    def plot_band_array(
            self,
            bands: int | Sequence[int] | None = None,
            *,
            cmap: str = "gray",
            figsize: tuple[int, int] = (8, 8),
            stretch: bool = False,
            pmin: float = 2,
            pmax: float = 98,
            title: str | None = None,
            save_path: str | None = None,
            **imshow_kwargs
    ) -> None:
        """
        Plot raster bands as NumPy arrays using row/column coordinates.

        Axes correspond to array indices, not spatial coordinates.

        :param ds: One or more raster datasets.
        :param bands: Band index or indices (1-based). If None, all bands are plotted.
        :param cmap: Matplotlib colormap.
        :param figsize: Size of the matplotlib figure
        :param stretch: Apply percentile contrast stretching.
        :param pmin: Lower percentile for stretch.
        :param pmax: Upper percentile for stretch.
        :param title: Optional figure title.
        :param save_path: File path if the figure should be saved to a file.
        :param imshow_kwargs: Additional arguments passed to matplotlib.pyplot.imshow.
        """
        ...

    def plot_raster(
            self,
            bands: int | Sequence[int] | None = None,
            *,
            cmap: str = "gray",
            figsize: tuple[int, int] = (8, 8),
            stretch: bool = False,
            pmin: float = 2,
            pmax: float = 98,
            title: str | None = None,
            save_path: str | None = None,
            **show_kwargs
    ) -> None:
        """
        Plot raster bands in spatial (CRS-aware) coordinates.

        :param ds: One or more raster datasets.
        :param bands: Band index or indices (1-based). If None, all bands are plotted.
        :param cmap: Colormap.
        :param figsize: Size of the matplotlib figure
        :param stretch: Apply percentile contrast stretching.
        :param pmin: Lower percentile.
        :param pmax: Upper percentile.
        :param title: Optional figure title.
        :param save_path: File path if the figure should be saved to a file.
        :param show_kwargs: Additional arguments passed to rasterio.plot.show.
        """
        ...

    def plot_histogram(
            self,
            bands: int | Sequence[int] | None = None,
            *,
            bins: int = 256,
            figsize: tuple[int, int] = (8, 8),
            log: bool = False,
            title: str | None = None,
            save_path: str | None = None,
            **hist_kwargs
    ) -> None:
        """
        Plot histograms of raster band values.

        :param ds: One or more raster datasets.
        :param bands: Band index or indices (1-based). If None, all bands are plotted.
        :param bins: Number of histogram bins.
        :param figsize: Size of the matplotlib figure
        :param log: Use logarithmic scale on the y-axis.
        :param title: Optional figure title.
        :param save_path: File path if the figure should be saved to a file.
        :param hist_kwargs: Additional arguments passed to matplotlib.pyplot.hist.
        """
        ...

    def plot_raster_with_histogram(
            self,
            bands: int | Sequence[int] | None = None,
            *,
            cmap: str = "gray",
            figsize: tuple[int, int] = (10, 5),
            bins: int = 256,
            pmin: float = 2,
            pmax: float = 98,
            stretch: bool = False,
            sharey: bool = False,
            title: str | None = None,
            save_path: str | None = None,
    ) -> None:
        """
        Plot raster bands alongside their histograms.

        Each band is displayed in spatial coordinates together with its
        corresponding value distribution.

        :param ds: Raster dataset.
        :param bands: Band index or indices (1-based). If None, all bands are plotted.
        :param cmap: Colormap.
        :param figsize: Size of the matplotlib figure
        :param bins: Number of histogram bins.
        :param stretch: Apply percentile contrast stretching.
        :param pmin: Lower percentile for the raster normalization. Used only if stretch is True.
        :param pmax: Upper percentile for the raster normalization. Used only if stretch is True.
        :param sharey: Share the y-axis between histograms.
        :param title: Optional figure title.
        :param save_path: File path if the figure should be saved to a file.
        """
        ...

    def plot_composite(
            self,
            bands: tuple[int, int, int],
            *,
            stretch: bool = False,
            figsize: tuple[int, int] = (8, 8),
            pmin: float = 2,
            pmax: float = 98,
            title: str | None = None,
            save_path: str | None = None,
    ) -> None:
        """
        Plot a three-band raster composite (e.g., RGB or false-color).

        :param ds: Raster dataset.
        :param bands: Tuple of three band indices (R, G, B).
        :param stretch: Apply percentile contrast stretching.
        :param figsize: Size of the matplotlib figure
        :param pmin: Lower percentile.
        :param pmax: Upper percentile.
        :param title: Optional figure title.
        :param save_path: File path if the figure should be saved to a file.
        """
        ...
