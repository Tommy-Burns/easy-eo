from __future__ import annotations

from typing import Sequence

from eeo.core.core import EEORasterDataset


def _as_list(obj):
    ...

def _normalize_bands(ds: EEORasterDataset, bands):
    ...

def _percentile_stretch(array, pmin=2, pmax=98):
    ...

def plot_band_array(
        ds: EEORasterDataset | list[EEORasterDataset],
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
        ds: EEORasterDataset | list[EEORasterDataset],
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
        ds: EEORasterDataset | list[EEORasterDataset],
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
        ds: EEORasterDataset,
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
        ds: EEORasterDataset,
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
