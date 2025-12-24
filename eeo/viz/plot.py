from __future__ import annotations

from typing import Sequence
import rasterio.plot as rioplot
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.pylabtools import figsize

from eeo.core.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_viz


# Visualization helper functions
def _as_list(obj):
    """
        Normalize an object to a list.

        If the input is already a list or tuple, it is returned unchanged.
        Otherwise, the input is wrapped in a single-element list.

        This helper allows public APIs to accept either a single object
        or multiple objects without additional branching logic.

        :param obj: Any object or a list/tuple of objects.
        :return: A list containing the input object(s).
        :rtype: list
    """
    return obj if isinstance(obj, (list, tuple)) else [obj]


def _normalize_bands(ds: EEORasterDataset, bands):
    """
        Normalize band selection into a list of band indices.

        This helper ensures that band inputs are consistently represented
        as a list of 1-based band indices, regardless of how the user
        specifies them.

        Accepted inputs:
        - None: selects all bands
        - int: selects a single band
        - iterable of int: selects multiple bands

        :param ds: Raster dataset used to determine band count.
        :type ds: EEORasterDataset
        :param bands: Band selection input.
        :type bands: int or sequence of int or None
        :return: List of 1-based band indices.
        :rtype: list[int]
    """
    if bands is None:
        return list(range(1, ds.get_count() + 1))
    if isinstance(bands, int):
        return [bands]
    return list(bands)


def _percentile_stretch(array, pmin=2, pmax=98):
    """
        Apply percentile-based contrast stretching to an array.

        This function rescales values in the input array to the range [0, 1]
        using lower and upper percentiles, improving visual contrast while
        suppressing extreme outliers.

        NaN values are ignored during percentile computation.

        :param arr: Input array (e.g., raster band).
        :param pmin: Lower percentile (default: 2).
        :param pmax: Upper percentile (default: 98).
        :return: Stretched array with values clipped to [0, 1].
        :rtype: numpy.ndarray
    """
    low, high = np.nanpercentile(array, (pmin, pmax))
    if high - low == 0:
        return np.zeros_like(array)
    return np.clip((array - low) / (high - low), 0, 1)


# Plot band as NumPy array
@eeo_raster_viz
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
    datasets = _as_list(ds)
    bands_list = _normalize_bands(datasets[0], bands)

    fig, axes = plt.subplots(
        len(bands_list),
        len(datasets),
        squeeze=False,
        figsize=figsize
    )

    for col, d in enumerate(datasets):
        for row, band in enumerate(bands_list):
            ax = axes[row, col]
            array = d.get_band(band)
            if stretch:
                array = _percentile_stretch(array, pmin, pmax)
            ax.imshow(array, cmap=cmap, **imshow_kwargs)
            ax.set_title(f"Band {band}")
            ax.axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# Plot raster in spatial coordinates
@eeo_raster_viz
def plot_raster(
        ds: EEORasterDataset | list[EEORasterDataset],
        bands: int | Sequence[int] | None = None,
        *,
        cmap: str = "gray",
        figsize: tuple[int, int] = (10, 5),
        stretch: bool = False,
        pmin: float = 2,
        pmax: float = 98,
        title: str | None = None,
        save_path: str | None = None,
        **show_kwargs
) -> None:
    datasets = _as_list(ds)
    bands_list = _normalize_bands(datasets[0], bands)

    fig, axes = plt.subplots(
        len(bands_list),
        len(datasets),
        squeeze=False,
        figsize=figsize
    )

    for col, d in enumerate(datasets):
        for row, band in enumerate(bands_list):
            ax = axes[row, col]
            array = d.get_band(band)
            if stretch:
                array = _percentile_stretch(array, pmin, pmax)
            rioplot.show(array, transform=d.get_transform(), ax=ax, cmap=cmap, **show_kwargs)
            ax.set_title(f"Band {band}")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# Plot histogram
@eeo_raster_viz
def plot_histogram(
        ds: EEORasterDataset | list[EEORasterDataset],
        bands: int | Sequence[int] | None = None,
        *,
        bins: int = 256,
        figsize: tuple[int, int] = (10, 5),
        log: bool = False,
        title: str | None = None,
        save_path: str | None = None,
        **hist_kwargs
) -> None:
    datasets = _as_list(ds)
    bands_list = _normalize_bands(datasets[0], bands)

    fig, axes = plt.subplots(
        len(bands_list),
        len(datasets),
        squeeze=False,
        figsize=figsize
    )

    for col, d in enumerate(datasets):
        for row, band in enumerate(bands_list):
            ax = axes[row, col]
            data = d.get_band(band).ravel()
            ax.hist(data, bins=bins, **hist_kwargs)
            if log:
                ax.set_yscale("log")
            ax.set_title(f"Band {band}")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# Plot raster and histogram side-by-side
@eeo_raster_viz
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
        save_path: str | None = None,
        title: str | None = None
) -> None:
    bands_list = _normalize_bands(ds, bands)

    fig, axes = plt.subplots(len(bands_list), 2, squeeze=False, sharey=sharey, figsize=figsize)

    for row, band in enumerate(bands_list):
        array = ds.get_band(band)
        if stretch:
            array = _percentile_stretch(array, pmin, pmax)

        rioplot.show(array, transform=ds.get_transform(), ax=axes[row, 0], cmap=cmap)
        axes[row, 1].hist(array.ravel(), bins=bins)

        axes[row, 0].set_title(f"Band {band}")
        axes[row, 1].set_title(f"Histogram of {band}")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# Plot composites
@eeo_raster_viz
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
    composite = np.stack([ds.get_band(b) for b in bands], axis=-1)

    if stretch:
        for i in range(3):
            composite[..., i] = _percentile_stretch(composite[..., i], pmin, pmax)

    plt.figure(figsize=figsize)
    plt.imshow(composite)
    plt.axis('off')

    if title:
        plt.title(title)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()
