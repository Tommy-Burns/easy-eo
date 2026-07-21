"""Terminal plotting functions for rasters."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import rasterio.plot as rioplot
from rasterio.transform import Affine

from eeo.common import is_rasterio_backed
from eeo.core.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_viz

# Reads for display are capped at the figure's pixel resolution times this
# oversampling factor, so moderate zooming stays sharp without ever pulling
# the full-resolution array.
_DISPLAY_OVERSAMPLE = 2.0


# Visualization helper functions
def _as_list(obj):
    """Wrap a single object in a list, passing lists and tuples through.

    Parameters
    ----------
    obj : object
        A single object, or a list/tuple of objects.

    Returns
    -------
    list or tuple
        ``obj`` unchanged when it is already a list or tuple, otherwise a
        one-element list ``[obj]``.
    """
    return obj if isinstance(obj, (list, tuple)) else [obj]


def _normalize_bands(ds: EEORasterDataset, bands):
    """Resolve a band selection to a list of 1-based band indices.

    Parameters
    ----------
    ds : EEORasterDataset
        Dataset whose band count is used when ``bands`` is None.
    bands : int or sequence of int or None
        None selects every band; an int selects a single band; a sequence
        selects the listed bands.

    Returns
    -------
    list of int
        1-based band indices.
    """
    if bands is None:
        return list(range(1, ds.get_count() + 1))
    if isinstance(bands, int):
        return [bands]
    return list(bands)


def _percentile_stretch(array, pmin=2, pmax=98):
    """Rescale an array to [0, 1] using percentile clipping.

    Values outside the ``pmin``-``pmax`` percentile range are clipped and the
    remaining range is scaled to [0, 1], improving display contrast while
    suppressing outliers. A constant array maps to all zeros.

    Parameters
    ----------
    array : numpy.ndarray
        Input array (e.g. a raster band).
    pmin : float, default 2
        Lower percentile.
    pmax : float, default 98
        Upper percentile.

    Returns
    -------
    numpy.ndarray
        Array clipped and scaled to [0, 1]. NaNs are ignored when computing
        the percentiles.
    """
    low, high = np.nanpercentile(array, (pmin, pmax))
    if high - low == 0:
        return np.zeros_like(array)
    return np.clip((array - low) / (high - low), 0, 1)


def _display_out_shape(shape: tuple[int, int], figsize: tuple[int, int]) -> tuple[int, int] | None:
    """Compute a decimated read shape capped at the figure's display budget.

    The budget is the figure size in pixels (``figsize`` times the Matplotlib
    ``figure.dpi``) times ``_DISPLAY_OVERSAMPLE``. Aspect ratio is preserved.

    Parameters
    ----------
    shape : tuple of int
        Native raster shape as ``(height, width)`` in pixels.
    figsize : tuple of int
        Figure size in inches, as passed to ``matplotlib.pyplot.subplots``.

    Returns
    -------
    tuple of int or None
        Decimated ``(height, width)`` for the read, or None when the raster
        already fits the display budget and should be read at full
        resolution.
    """
    dpi = float(plt.rcParams.get("figure.dpi", 100.0))
    max_height = figsize[1] * dpi * _DISPLAY_OVERSAMPLE
    max_width = figsize[0] * dpi * _DISPLAY_OVERSAMPLE

    height, width = shape
    scale = min(max_height / height, max_width / width)
    if scale >= 1.0:
        return None
    return max(1, round(height * scale)), max(1, round(width * scale))


def _read_band_for_display(
    ds: EEORasterDataset, band: int, figsize: tuple[int, int]
) -> tuple[np.ndarray, Affine]:
    """Read one band at display resolution, with a matching transform.

    Rasterio-backed datasets larger than the display budget are read
    decimated via ``out_shape`` (GDAL serves such reads from overviews when
    present), and the returned transform is rescaled so the decimated array
    still maps to the raster's true extent. Small rasters, and NumPy-backed
    datasets (whose pixels are already in memory), are returned in full.

    Parameters
    ----------
    ds : EEORasterDataset
        Dataset to read from.
    band : int
        1-based band index.
    figsize : tuple of int
        Figure size in inches, used to derive the display budget.

    Returns
    -------
    tuple of (numpy.ndarray, affine.Affine)
        The band as a 2D array, and the transform mapping that array to
        world coordinates.
    """
    transform = ds.get_transform()
    out_shape = _display_out_shape(ds.get_shape(), figsize)
    if out_shape is None or not is_rasterio_backed(ds):
        return ds.get_band(band), transform

    array = ds.read(band, out_shape=out_shape)
    height, width = ds.get_shape()
    out_height, out_width = out_shape
    return array, transform * Affine.scale(width / out_width, height / out_height)


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
    **imshow_kwargs,
) -> None:
    """Plot raster bands as arrays in row/column (pixel) space.

    Draws one subplot per band, with bands down the rows and datasets across
    the columns. Axes are array indices, not spatial coordinates; use
    ``plot_raster`` for CRS-aware axes.

    Parameters
    ----------
    ds : EEORasterDataset or list of EEORasterDataset
        One dataset, or several to display side by side.
    bands : int or sequence of int or None, default None
        1-based band(s) to plot; None plots every band.
    cmap : str, default "gray"
        Matplotlib colormap.
    figsize : tuple of int, default (8, 8)
        Figure size in inches.
    stretch : bool, default False
        If True, apply percentile contrast stretching to each band.
    pmin : float, default 2
        Lower percentile for the stretch (used only when ``stretch=True``).
    pmax : float, default 98
        Upper percentile for the stretch (used only when ``stretch=True``).
    title : str or None, default None
        Optional figure title.
    save_path : str or None, default None
        If given, save the figure to this path at 300 dpi.
    **imshow_kwargs
        Extra keyword arguments forwarded to ``matplotlib.pyplot.imshow``.

    Returns
    -------
    None
        Terminal operation; displays (and optionally saves) a figure.

    Notes
    -----
    Reads each band decimated to the figure's display resolution (rasterio
    ``out_shape``, served from overviews when present); rasters already
    within the display budget, and NumPy-backed datasets, are read in full.
    Displays the figure with ``matplotlib.pyplot.show`` and, when
    ``save_path`` is given, writes it to disk as a side effect.

    Examples
    --------
    >>> ds.plot_band_array(bands=[1, 2, 3], stretch=True)
    """
    datasets = _as_list(ds)
    bands_list = _normalize_bands(datasets[0], bands)

    fig, axes = plt.subplots(len(bands_list), len(datasets), squeeze=False, figsize=figsize)

    for col, d in enumerate(datasets):
        for row, band in enumerate(bands_list):
            ax = axes[row, col]
            array, _ = _read_band_for_display(d, band, figsize)
            if stretch:
                array = _percentile_stretch(array, pmin, pmax)
            ax.imshow(array, cmap=cmap, **imshow_kwargs)
            ax.set_title(f"Band {band}")
            ax.axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
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
    **show_kwargs,
) -> None:
    """Plot raster bands in spatial (CRS-aware) coordinates.

    Draws one subplot per band using the raster's transform, with bands down
    the rows and datasets across the columns. Use ``plot_band_array`` for
    plain array-index axes.

    Parameters
    ----------
    ds : EEORasterDataset or list of EEORasterDataset
        One dataset, or several to display side by side.
    bands : int or sequence of int or None, default None
        1-based band(s) to plot; None plots every band.
    cmap : str, default "gray"
        Matplotlib colormap.
    figsize : tuple of int, default (10, 5)
        Figure size in inches.
    stretch : bool, default False
        If True, apply percentile contrast stretching to each band.
    pmin : float, default 2
        Lower percentile for the stretch (used only when ``stretch=True``).
    pmax : float, default 98
        Upper percentile for the stretch (used only when ``stretch=True``).
    title : str or None, default None
        Optional figure title.
    save_path : str or None, default None
        If given, save the figure to this path at 300 dpi.
    **show_kwargs
        Extra keyword arguments forwarded to ``rasterio.plot.show``.

    Returns
    -------
    None
        Terminal operation; displays (and optionally saves) a figure.

    Notes
    -----
    Reads each band decimated to the figure's display resolution (rasterio
    ``out_shape``, served from overviews when present); rasters already
    within the display budget, and NumPy-backed datasets, are read in full.
    Displays the figure with ``matplotlib.pyplot.show`` and, when
    ``save_path`` is given, writes it to disk as a side effect.

    Examples
    --------
    >>> ds.plot_raster(bands=1, stretch=True)
    """
    datasets = _as_list(ds)
    bands_list = _normalize_bands(datasets[0], bands)

    fig, axes = plt.subplots(len(bands_list), len(datasets), squeeze=False, figsize=figsize)

    for col, d in enumerate(datasets):
        for row, band in enumerate(bands_list):
            ax = axes[row, col]
            array, transform = _read_band_for_display(d, band, figsize)
            if stretch:
                array = _percentile_stretch(array, pmin, pmax)
            rioplot.show(array, transform=transform, ax=ax, cmap=cmap, **show_kwargs)
            ax.set_title(f"Band {band}")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
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
    **hist_kwargs,
) -> None:
    """Plot per-band value histograms.

    Draws one histogram per band, with bands down the rows and datasets
    across the columns.

    Parameters
    ----------
    ds : EEORasterDataset or list of EEORasterDataset
        One dataset, or several to compare side by side.
    bands : int or sequence of int or None, default None
        1-based band(s) to plot; None plots every band.
    bins : int, default 256
        Number of histogram bins.
    figsize : tuple of int, default (10, 5)
        Figure size in inches.
    log : bool, default False
        If True, use a logarithmic y-axis.
    title : str or None, default None
        Optional figure title.
    save_path : str or None, default None
        If given, save the figure to this path at 300 dpi.
    **hist_kwargs
        Extra keyword arguments forwarded to ``matplotlib.pyplot.hist``.

    Returns
    -------
    None
        Terminal operation; displays (and optionally saves) a figure.

    Notes
    -----
    Reads each band at full resolution into memory; nodata pixels are counted
    as ordinary values. Displays the figure with ``matplotlib.pyplot.show``
    and, when ``save_path`` is given, writes it to disk as a side effect.

    Examples
    --------
    >>> ds.plot_histogram(log=True)
    """
    datasets = _as_list(ds)
    bands_list = _normalize_bands(datasets[0], bands)

    fig, axes = plt.subplots(len(bands_list), len(datasets), squeeze=False, figsize=figsize)

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
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
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
    title: str | None = None,
) -> None:
    """Plot each band alongside its value histogram.

    For every selected band, draws the raster (in spatial coordinates) and
    its histogram side by side on one row.

    Parameters
    ----------
    ds : EEORasterDataset
        Raster dataset to display.
    bands : int or sequence of int or None, default None
        1-based band(s) to plot; None plots every band.
    cmap : str, default "gray"
        Matplotlib colormap for the raster panel.
    figsize : tuple of int, default (10, 5)
        Figure size in inches.
    bins : int, default 256
        Number of histogram bins.
    pmin : float, default 2
        Lower percentile for the stretch (used only when ``stretch=True``).
    pmax : float, default 98
        Upper percentile for the stretch (used only when ``stretch=True``).
    stretch : bool, default False
        If True, apply percentile contrast stretching to the raster panel.
    sharey : bool, default False
        If True, share the y-axis across the histogram panels.
    save_path : str or None, default None
        If given, save the figure to this path at 300 dpi.
    title : str or None, default None
        Optional figure title.

    Returns
    -------
    None
        Terminal operation; displays (and optionally saves) a figure.

    Notes
    -----
    Reads each band decimated to the figure's display resolution (rasterio
    ``out_shape``, served from overviews when present); rasters already
    within the display budget, and NumPy-backed datasets, are read in full.
    For a decimated raster the histogram is computed from the decimated
    pixels, so bin counts are an approximation of the full-resolution
    histogram. Displays the figure with ``matplotlib.pyplot.show`` and, when
    ``save_path`` is given, writes it to disk as a side effect.

    Examples
    --------
    >>> ds.plot_raster_with_histogram(bands=[1, 2])
    """
    bands_list = _normalize_bands(ds, bands)

    fig, axes = plt.subplots(len(bands_list), 2, squeeze=False, sharey=sharey, figsize=figsize)

    for row, band in enumerate(bands_list):
        array, transform = _read_band_for_display(ds, band, figsize)
        if stretch:
            array = _percentile_stretch(array, pmin, pmax)

        rioplot.show(array, transform=transform, ax=axes[row, 0], cmap=cmap)
        axes[row, 1].hist(array.ravel(), bins=bins)

        axes[row, 0].set_title(f"Band {band}")
        axes[row, 1].set_title(f"Histogram of {band}")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
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
    """Plot a three-band RGB (or false-colour) composite.

    Stacks the three requested bands into an RGB image, in the order given
    (first band -> red, second -> green, third -> blue).

    Parameters
    ----------
    ds : EEORasterDataset
        Raster dataset to display.
    bands : tuple of int
        Exactly three 1-based band indices, mapped to R, G, B in order.
    stretch : bool, default False
        If True, apply percentile contrast stretching to each channel.
    figsize : tuple of int, default (8, 8)
        Figure size in inches.
    pmin : float, default 2
        Lower percentile for the stretch (used only when ``stretch=True``).
    pmax : float, default 98
        Upper percentile for the stretch (used only when ``stretch=True``).
    title : str or None, default None
        Optional figure title.
    save_path : str or None, default None
        If given, save the figure to this path at 300 dpi.

    Returns
    -------
    None
        Terminal operation; displays (and optionally saves) a figure.

    Notes
    -----
    Reads the three bands decimated to the figure's display resolution
    (rasterio ``out_shape``, served from overviews when present); rasters
    already within the display budget, and NumPy-backed datasets, are read
    in full. Percentile stretching writes the scaled values back into the
    composite's own dtype, so stretch a floating-dtype raster to avoid
    truncating them. Displays the figure with ``matplotlib.pyplot.show``
    and, when ``save_path`` is given, writes it to disk as a side effect.

    Examples
    --------
    >>> ds.plot_composite(bands=(3, 2, 1))
    """
    composite = np.stack([_read_band_for_display(ds, b, figsize)[0] for b in bands], axis=-1)

    if stretch:
        for i in range(3):
            composite[..., i] = _percentile_stretch(composite[..., i], pmin, pmax)

    plt.figure(figsize=figsize)
    plt.imshow(composite)
    plt.axis("off")

    if title:
        plt.title(title)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()
