"""Terminal plotting functions for rasters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import rasterio.plot as rioplot
from rasterio.transform import Affine

from eeo.common import get_nodata, is_rasterio_backed, resolve_band_index
from eeo.core.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_viz
from eeo.core.exceptions import ValidationError
from eeo.core.types import StrPath

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
        Dataset whose band count and names resolve the selection.
    bands : int or str or sequence of (int or str) or None
        None selects every band; a single int index or band name selects one
        band; a sequence selects the listed bands and may mix indices and
        names.

    Returns
    -------
    list of int
        1-based band indices.
    """
    if bands is None:
        return list(range(1, ds.get_count() + 1))
    if isinstance(bands, (int, str)):
        bands = [bands]
    return [resolve_band_index(ds, band) for band in bands]


def _band_label(ds: EEORasterDataset, band: int) -> str:
    """Return a subplot label for a band, appending its name when it has one."""
    name = ds.band_names[band - 1]
    return f"Band {band}" if name is None else f"Band {band} ({name})"


def _auto_grid(n_panels: int) -> tuple[int, int]:
    """Choose a near-square grid for ``n_panels``, never taller than it is wide.

    ``nrows = floor(sqrt(n))`` with the columns derived from it. Taking the
    floor keeps the grid landscape-shaped, which suits both the default figure
    sizes and the screens these are read on, and it wastes at most one row:
    2 and 3 panels stay a single row, 4 becomes 2x2, 6 becomes 2x3, 8 becomes
    2x4, 9 becomes 3x3.

    Parameters
    ----------
    n_panels : int
        Number of subplots to place.

    Returns
    -------
    tuple of int
        ``(nrows, ncols)``, large enough to hold ``n_panels``.
    """
    rows = max(1, int(n_panels**0.5))
    return rows, -(-n_panels // rows)


def _resolve_figsize(
    figsize: tuple[int, int] | None, base: tuple[int, int], rows: int, cols: int
) -> tuple[float, float]:
    """Resolve the figure size for a grid, honouring an explicit request.

    The per-function defaults were chosen for a single row of panels, and reuse
    them unchanged for a 2x2 block leaves square maps squeezed into wide, short
    cells. When the caller expresses no preference and the grid has more than
    one row, the width is kept and the height set to make the cells roughly
    square.

    Parameters
    ----------
    figsize : tuple of int or None
        The caller's request; None asks for the derived size.
    base : tuple of int
        The function's own default, treated as the size of a single row.
    rows, cols : int
        Grid shape.

    Returns
    -------
    tuple of float
        Figure size in inches: ``figsize`` when given, ``base`` for a
        single-row grid, else the width of ``base`` with a derived height.
    """
    if figsize is not None:
        return figsize
    if rows == 1:
        return base
    width = base[0]
    return width, width * rows / cols


def _grid_shape(n_panels: int, nrows: int | None, ncols: int | None) -> tuple[int, int]:
    """Resolve a requested grid to concrete ``(nrows, ncols)``.

    Parameters
    ----------
    n_panels : int
        Number of subplots to place.
    nrows, ncols : int or None
        Requested grid. Giving one derives the other; giving both is used as
        is, provided it has room for every panel.

    Returns
    -------
    tuple of int
        ``(nrows, ncols)``, large enough to hold ``n_panels``.

    Raises
    ------
    ValidationError
        If either value is not a positive integer, or if both are given and
        their product is smaller than ``n_panels``.
    """
    for label, value in (("nrows", nrows), ("ncols", ncols)):
        if value is not None and (not isinstance(value, int) or value < 1):
            raise ValidationError(f"{label} must be a positive integer; got {value!r}")

    if nrows is not None and ncols is not None:
        if nrows * ncols < n_panels:
            raise ValidationError(
                f"a {nrows}x{ncols} grid holds {nrows * ncols} panels, but {n_panels} "
                f"were requested; increase nrows or ncols"
            )
        return nrows, ncols
    if ncols is not None:
        return -(-n_panels // ncols), ncols
    assert nrows is not None  # both-None is handled by the caller
    return nrows, -(-n_panels // nrows)


def _panel_grid(
    datasets: Sequence[EEORasterDataset],
    bands_list: Sequence[int],
    nrows: int | None,
    ncols: int | None,
    figsize: tuple[int, int] | None,
    base_figsize: tuple[int, int],
    **subplots_kwargs,
):
    """Build the subplot grid and pair each axes with the panel it draws.

    Several datasets *and* several bands keep the semantic layout — bands down
    the rows, datasets across the columns — which is what aligns band *i* of
    one dataset beside band *i* of the next; reflowing that would destroy the
    comparison the figure exists to make. Anything else is one flat run of
    panels, laid out near-square by default (``_auto_grid``) so a four-band
    raster is a 2x2 block rather than a four-storey strip.

    A requested grid overrides the default, flowing the panels row-major and
    dataset-major within that (every band of the first dataset, then every band
    of the second), and switches off any leftover axes.

    Parameters
    ----------
    datasets : sequence of EEORasterDataset
        Datasets to draw.
    bands_list : sequence of int
        1-based band indices, applied to every dataset.
    nrows, ncols : int or None
        Requested grid; both None takes the default layout.
    figsize : tuple of int or None
        Requested figure size; None derives one (see ``_resolve_figsize``).
    base_figsize : tuple of int
        The calling function's own default, as the size for a single row.
    **subplots_kwargs
        Extra keyword arguments for ``matplotlib.pyplot.subplots``.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, list, tuple of float)
        The figure, a list of ``(axes, dataset, band)`` triples in draw order,
        and the figure size used — which callers need for the display budget.

    Raises
    ------
    ValidationError
        If the requested grid is invalid or too small (see ``_grid_shape``).
    """
    two_dimensional = len(datasets) > 1 and len(bands_list) > 1
    if nrows is None and ncols is None and two_dimensional:
        rows, cols = len(bands_list), len(datasets)
        size = _resolve_figsize(figsize, base_figsize, rows, cols)
        fig, axes = plt.subplots(rows, cols, squeeze=False, figsize=size, **subplots_kwargs)
        panels = [
            (axes[row, col], ds, band)
            for col, ds in enumerate(datasets)
            for row, band in enumerate(bands_list)
        ]
        return fig, panels, size

    pairs = [(ds, band) for ds in datasets for band in bands_list]
    if nrows is None and ncols is None:
        rows, cols = _auto_grid(len(pairs))
    else:
        rows, cols = _grid_shape(len(pairs), nrows, ncols)

    size = _resolve_figsize(figsize, base_figsize, rows, cols)
    fig, axes = plt.subplots(rows, cols, squeeze=False, figsize=size, **subplots_kwargs)

    flat = axes.ravel()
    for ax in flat[len(pairs) :]:
        ax.axis("off")
    # strict=False: the grid may hold more cells than panels, and the extras
    # were just switched off above.
    return fig, [(ax, ds, band) for ax, (ds, band) in zip(flat, pairs, strict=False)], size


def _mask_nodata_for_display(ds: EEORasterDataset, array):
    """Mask a band's declared nodata sentinel so display treats it as absent.

    The nodata contract (``CODE_STYLE.md``, "Nodata policy") requires every
    statistic to exclude nodata — a ``-9999`` fill must not shift a percentile
    stretch. Masking here is what makes the plots obey it: the stretch, the
    colorbar and the histograms all see valid pixels only, and Matplotlib
    renders the masked pixels transparent instead of colouring them.

    A float raster's nodata is already NaN under the contract, and NaN is
    ignored by the percentile helpers and rendered blank, so it needs no mask.

    Parameters
    ----------
    ds : EEORasterDataset
        Dataset the band was read from, consulted for its nodata value.
    array : numpy.ndarray
        Band values as read.

    Returns
    -------
    numpy.ndarray
        A masked array hiding the sentinel, or ``array`` unchanged when the
        dataset declares no nodata (every pixel is valid) or declares NaN.
    """
    nodata = get_nodata(ds)
    if nodata is None or np.isnan(nodata):
        return array
    return np.ma.masked_equal(array, nodata)


def _valid_values(array):
    """Return an array's non-nodata values, flattened.

    Parameters
    ----------
    array : numpy.ndarray
        Plain or masked array.

    Returns
    -------
    numpy.ndarray
        1D array of the unmasked values; the whole array when nothing is
        masked. NaNs survive — the percentile helpers ignore those separately.
    """
    if np.ma.isMaskedArray(array):
        return array.compressed()
    return np.asarray(array).ravel()


def _percentile_stretch(array, pmin=2, pmax=98):
    """Rescale an array to [0, 1] using percentile clipping.

    Values outside the ``pmin``-``pmax`` percentile range are clipped and the
    remaining range is scaled to [0, 1], improving display contrast while
    suppressing outliers. A constant array maps to all zeros.

    Used where the rescaled values are themselves needed — stacking an RGB
    composite. Single-band plots stretch with ``_stretch_limits`` instead,
    which normalizes at display time and so keeps the array in its own units.

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
        Array clipped and scaled to [0, 1]. Nodata pixels stay masked, and
        both they and NaNs are ignored when computing the percentiles.
    """
    values = _valid_values(array)
    if values.size == 0:  # every pixel is nodata
        return np.zeros_like(array)

    low, high = np.nanpercentile(values, (pmin, pmax))
    if high - low == 0:
        return np.zeros_like(array)
    return np.clip((array - low) / (high - low), 0, 1)


def _stretch_limits(array, pmin=2, pmax=98) -> tuple[float, float] | None:
    """Return percentile display limits for an array.

    The limits are the ``pmin``-``pmax`` percentiles of ``array``, to be handed
    to Matplotlib as ``vmin``/``vmax``. Normalizing at display time is
    equivalent to rescaling the data with ``_percentile_stretch`` — both map
    values through ``clip((x - low) / (high - low), 0, 1)`` — but leaves the
    array in its own units, so the rendered image is identical while anything
    reading the plotted values back reports real data values.

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
    tuple of float or None
        The ``(vmin, vmax)`` display limits, or None when the percentile range
        is empty (a constant array) or not finite (an all-nodata band, whose
        percentiles are NaN). Both cases fall back to Matplotlib's own
        autoscaling rather than handing it degenerate limits. Masked nodata
        pixels and NaNs are ignored when computing the percentiles.
    """
    values = _valid_values(array)
    if values.size == 0:  # every pixel is nodata
        return None

    low, high = np.nanpercentile(values, (pmin, pmax))
    if not (np.isfinite(low) and np.isfinite(high)) or high - low == 0:
        return None
    return float(low), float(high)


def _with_stretch_limits(
    draw_kwargs: dict[str, Any], array, pmin: float, pmax: float
) -> dict[str, Any]:
    """Copy draw keyword arguments, filling in percentile display limits.

    Returns a copy rather than mutating so that limits computed for one band
    cannot leak into the next subplot drawn from the same caller kwargs.
    Caller-supplied ``vmin``/``vmax`` win: they are left untouched.

    Parameters
    ----------
    draw_kwargs : dict
        Keyword arguments destined for ``imshow`` (directly or through
        ``rasterio.plot.show``).
    array : numpy.ndarray
        Array being drawn, whose percentiles set the limits.
    pmin : float
        Lower percentile.
    pmax : float
        Upper percentile.

    Returns
    -------
    dict
        A new dict with ``vmin``/``vmax`` set from the percentiles of
        ``array`` unless already present, or an unchanged copy for a constant
        array.
    """
    kwargs = dict(draw_kwargs)
    limits = _stretch_limits(array, pmin, pmax)
    if limits is not None:
        kwargs.setdefault("vmin", limits[0])
        kwargs.setdefault("vmax", limits[1])
    return kwargs


def _colorbar_extend(mappable) -> str:
    """Report which ends of a colorbar are clipping data, as an ``extend`` value.

    A stretched plot maps everything below ``vmin`` (or above ``vmax``) to the
    colormap's end colour, so those pixels are drawn as if they sat at the
    limit. The matching arrowhead on the colorbar is what tells a reader the
    scale is clipped rather than complete.

    Parameters
    ----------
    mappable : matplotlib.cm.ScalarMappable
        The drawn image, carrying both the plotted array and its limits.

    Returns
    -------
    str
        ``"both"``, ``"min"``, ``"max"``, or ``"neither"``, as
        ``matplotlib.figure.Figure.colorbar`` expects.
    """
    values = np.asarray(_valid_values(mappable.get_array()), dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "neither"

    vmin, vmax = mappable.get_clim()
    below = bool(finite.min() < vmin)
    above = bool(finite.max() > vmax)
    if below and above:
        return "both"
    if below:
        return "min"
    return "max" if above else "neither"


def _add_colorbar(fig, ax, mappable, ds: EEORasterDataset, band: int, label: str | None) -> None:
    """Attach a colorbar to one subplot, labelled in the band's own units.

    Drawn per subplot rather than per figure: bands in a grid carry unrelated
    ranges (and unrelated physical units), so a single shared scale would
    mislabel every panel but one.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure that owns the subplot.
    ax : matplotlib.axes.Axes
        Subplot the colorbar is attached to.
    mappable : matplotlib.cm.ScalarMappable or None
        The drawn image. None when the panel holds no image (e.g. a caller
        passed ``contour=True`` through to ``rasterio.plot.show``), in which
        case there is nothing to describe and no colorbar is drawn.
    ds : EEORasterDataset
        Dataset the band came from, consulted for the default label.
    band : int
        1-based band index.
    label : str or None
        Explicit label; None falls back to the band's name, and an unnamed
        band yields an unlabelled colorbar.

    Returns
    -------
    None
        Draws on ``fig`` as a side effect.
    """
    if mappable is None:
        return

    colorbar = fig.colorbar(mappable, ax=ax, shrink=0.7, extend=_colorbar_extend(mappable))
    text = label if label is not None else ds.band_names[band - 1]
    if text:
        colorbar.set_label(text)


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
        The band as a 2D array — masked where the dataset declares a nodata
        sentinel — and the transform mapping that array to world coordinates.
    """
    transform = ds.get_transform()
    out_shape = _display_out_shape(ds.get_shape(), figsize)
    if out_shape is None or not is_rasterio_backed(ds):
        return _mask_nodata_for_display(ds, ds.get_band(band)), transform

    array = ds.read(band, out_shape=out_shape)
    height, width = ds.get_shape()
    out_height, out_width = out_shape
    return (
        _mask_nodata_for_display(ds, array),
        transform * Affine.scale(width / out_width, height / out_height),
    )


# Plot band as NumPy array
@eeo_raster_viz
def plot_band_array(
    ds: EEORasterDataset | list[EEORasterDataset],
    bands: int | str | Sequence[int | str] | None = None,
    *,
    cmap: str = "gray",
    figsize: tuple[int, int] | None = None,
    nrows: int | None = None,
    ncols: int | None = None,
    stretch: bool = True,
    pmin: float = 2,
    pmax: float = 98,
    colorbar: bool = False,
    colorbar_label: str | None = None,
    title: str | None = None,
    save_path: StrPath | None = None,
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
    bands : int or str or sequence of (int or str) or None, default None
        Band(s) to plot, each a 1-based index or a band name; None plots
        every band. A sequence may mix indices and names.
    cmap : str, default "gray"
        Matplotlib colormap.
    figsize : tuple of int or None, default None
        Figure size in inches. None derives one: (8, 8) for a single
        row of panels, and for a taller grid the same width with the height
        set to keep the cells roughly square.
    nrows : int or None, default None
        Number of subplot rows. With ``ncols`` also None, the layout is one
        row per band and one column per dataset; giving either reflows the
        panels row-major into that grid instead (every band of the first
        dataset, then of the next), hiding any leftover cells.
    ncols : int or None, default None
        Number of subplot columns; see ``nrows``.
    stretch : bool, default True
        If True, apply percentile contrast stretching to each band. On by
        default because the ``pmin``-``pmax`` (2-98) stretch renders most
        rasters best; pass ``stretch=False`` to display raw values. The
        stretch sets each subplot's display limits and leaves the pixel
        values themselves unchanged.
    pmin : float, default 2
        Lower percentile for the stretch (used only when ``stretch=True``).
    pmax : float, default 98
        Upper percentile for the stretch (used only when ``stretch=True``).
    colorbar : bool, default False
        If True, draw a colorbar beside each subplot, scaled in the band's own
        values. Arrowheads mark the ends where the stretch clips data.
    colorbar_label : str or None, default None
        Label for the colorbars; None uses each band's name, leaving an
        unnamed band's colorbar unlabelled.
    title : str or None, default None
        Optional figure title.
    save_path : str or path-like or None, default None
        If given, save the figure to this path at 300 dpi.
    **imshow_kwargs
        Extra keyword arguments forwarded to ``matplotlib.pyplot.imshow``. An
        explicit ``vmin``/``vmax`` takes precedence over the stretch.

    Returns
    -------
    None
        Terminal operation; displays (and optionally saves) a figure.

    Raises
    ------
    IndexError
        If a band index is outside the range of available bands.
    ValidationError
        If ``bands`` names a band that is unknown or matches more than one
        band.

    Notes
    -----
    Reads each band decimated to the figure's display resolution (rasterio
    ``out_shape``, served from overviews when present); rasters already
    within the display budget, and NumPy-backed datasets, are read in full.
    A declared nodata value is excluded per the library's nodata contract: the
    sentinel is masked before the percentiles are taken, so it cannot shift the
    stretch or the colorbar, and those pixels render blank rather than as a
    colour. Displays the figure with ``matplotlib.pyplot.show`` and, when
    ``save_path`` is given, writes it to disk as a side effect.

    Examples
    --------
    >>> ds.plot_band_array(bands=[1, 2, 3])                # 2-98 stretch on by default
    >>> ds.plot_band_array(bands=[1, 2, 3], stretch=False)  # raw values
    >>> ds.plot_band_array(bands=1, colorbar=True)          # scale in band units
    """
    datasets = _as_list(ds)
    bands_list = _normalize_bands(datasets[0], bands)

    fig, panels, size = _panel_grid(datasets, bands_list, nrows, ncols, figsize, (8, 8))

    for ax, d, band in panels:
        array, _ = _read_band_for_display(d, band, size)
        draw_kwargs = imshow_kwargs
        if stretch:
            draw_kwargs = _with_stretch_limits(imshow_kwargs, array, pmin, pmax)
        image = ax.imshow(array, cmap=cmap, **draw_kwargs)
        ax.set_title(_band_label(d, band))
        ax.axis("off")
        if colorbar:
            _add_colorbar(fig, ax, image, d, band, colorbar_label)

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
    bands: int | str | Sequence[int | str] | None = None,
    *,
    cmap: str = "gray",
    figsize: tuple[int, int] | None = None,
    nrows: int | None = None,
    ncols: int | None = None,
    stretch: bool = True,
    pmin: float = 2,
    pmax: float = 98,
    colorbar: bool = False,
    colorbar_label: str | None = None,
    title: str | None = None,
    save_path: StrPath | None = None,
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
    bands : int or str or sequence of (int or str) or None, default None
        Band(s) to plot, each a 1-based index or a band name; None plots
        every band. A sequence may mix indices and names.
    cmap : str, default "gray"
        Matplotlib colormap.
    figsize : tuple of int or None, default None
        Figure size in inches. None derives one: (10, 5) for a single
        row of panels, and for a taller grid the same width with the height
        set to keep the cells roughly square.
    nrows : int or None, default None
        Number of subplot rows. With ``ncols`` also None, the layout is one
        row per band and one column per dataset; giving either reflows the
        panels row-major into that grid instead (every band of the first
        dataset, then of the next), hiding any leftover cells.
    ncols : int or None, default None
        Number of subplot columns; see ``nrows``.
    stretch : bool, default True
        If True, apply percentile contrast stretching to each band. On by
        default because the ``pmin``-``pmax`` (2-98) stretch renders most
        rasters best; pass ``stretch=False`` to display raw values. The
        stretch sets each subplot's display limits and leaves the pixel
        values themselves unchanged.
    pmin : float, default 2
        Lower percentile for the stretch (used only when ``stretch=True``).
    pmax : float, default 98
        Upper percentile for the stretch (used only when ``stretch=True``).
    colorbar : bool, default False
        If True, draw a colorbar beside each subplot, scaled in the band's own
        values. Arrowheads mark the ends where the stretch clips data.
    colorbar_label : str or None, default None
        Label for the colorbars; None uses each band's name, leaving an
        unnamed band's colorbar unlabelled.
    title : str or None, default None
        Optional figure title.
    save_path : str or path-like or None, default None
        If given, save the figure to this path at 300 dpi.
    **show_kwargs
        Extra keyword arguments forwarded to ``rasterio.plot.show``. An
        explicit ``vmin``/``vmax`` takes precedence over the stretch.

    Returns
    -------
    None
        Terminal operation; displays (and optionally saves) a figure.

    Raises
    ------
    IndexError
        If a band index is outside the range of available bands.
    ValidationError
        If ``bands`` names a band that is unknown or matches more than one
        band.

    Notes
    -----
    Reads each band decimated to the figure's display resolution (rasterio
    ``out_shape``, served from overviews when present); rasters already
    within the display budget, and NumPy-backed datasets, are read in full.
    A declared nodata value is excluded per the library's nodata contract: the
    sentinel is masked before the percentiles are taken, so it cannot shift the
    stretch or the colorbar, and those pixels render blank rather than as a
    colour. Displays the figure with ``matplotlib.pyplot.show`` and, when
    ``save_path`` is given, writes it to disk as a side effect.

    Examples
    --------
    >>> ds.plot_raster(bands=1)                 # 2-98 stretch on by default
    >>> ds.plot_raster(bands=1, stretch=False)  # raw values
    >>> ds.ndvi(red="red", nir="nir", name="NDVI").plot_raster(
    ...     cmap="RdYlGn", colorbar=True
    ... )  # colorbar labelled "NDVI", in index units
    """
    datasets = _as_list(ds)
    bands_list = _normalize_bands(datasets[0], bands)

    fig, panels, size = _panel_grid(datasets, bands_list, nrows, ncols, figsize, (10, 5))

    for ax, d, band in panels:
        array, transform = _read_band_for_display(d, band, size)
        draw_kwargs = show_kwargs
        if stretch:
            draw_kwargs = _with_stretch_limits(show_kwargs, array, pmin, pmax)
        rioplot.show(array, transform=transform, ax=ax, cmap=cmap, **draw_kwargs)
        ax.set_title(_band_label(d, band))
        if colorbar:
            # rasterio.plot.show returns the axes, not the image it drew.
            images = ax.get_images()
            _add_colorbar(fig, ax, images[-1] if images else None, d, band, colorbar_label)

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
    bands: int | str | Sequence[int | str] | None = None,
    *,
    bins: int = 256,
    figsize: tuple[int, int] | None = None,
    nrows: int | None = None,
    ncols: int | None = None,
    log: bool = False,
    title: str | None = None,
    save_path: StrPath | None = None,
    **hist_kwargs,
) -> None:
    """Plot per-band value histograms.

    Draws one histogram per band, with bands down the rows and datasets
    across the columns.

    Parameters
    ----------
    ds : EEORasterDataset or list of EEORasterDataset
        One dataset, or several to compare side by side.
    bands : int or str or sequence of (int or str) or None, default None
        Band(s) to plot, each a 1-based index or a band name; None plots
        every band. A sequence may mix indices and names.
    bins : int, default 256
        Number of histogram bins.
    figsize : tuple of int or None, default None
        Figure size in inches. None derives one: (10, 5) for a single
        row of panels, and for a taller grid the same width with the height
        set to keep the cells roughly square.
    nrows : int or None, default None
        Number of subplot rows. With ``ncols`` also None, the layout is one
        row per band and one column per dataset; giving either reflows the
        panels row-major into that grid instead (every band of the first
        dataset, then of the next), hiding any leftover cells.
    ncols : int or None, default None
        Number of subplot columns; see ``nrows``.
    log : bool, default False
        If True, use a logarithmic y-axis.
    title : str or None, default None
        Optional figure title.
    save_path : str or path-like or None, default None
        If given, save the figure to this path at 300 dpi.
    **hist_kwargs
        Extra keyword arguments forwarded to ``matplotlib.pyplot.hist``.

    Returns
    -------
    None
        Terminal operation; displays (and optionally saves) a figure.

    Raises
    ------
    IndexError
        If a band index is outside the range of available bands.
    ValidationError
        If ``bands`` names a band that is unknown or matches more than one
        band.

    Notes
    -----
    Reads each band at full resolution into memory. A declared nodata value is
    excluded per the library's nodata contract, so the counts describe valid
    pixels only. Displays the figure with ``matplotlib.pyplot.show`` and, when
    ``save_path`` is given, writes it to disk as a side effect.

    Examples
    --------
    >>> ds.plot_histogram(log=True)
    """
    datasets = _as_list(ds)
    bands_list = _normalize_bands(datasets[0], bands)

    fig, panels, _ = _panel_grid(datasets, bands_list, nrows, ncols, figsize, (10, 5))

    for ax, d, band in panels:
        data = _valid_values(_mask_nodata_for_display(d, d.get_band(band)))
        ax.hist(data, bins=bins, **hist_kwargs)
        if log:
            ax.set_yscale("log")
        ax.set_title(_band_label(d, band))

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
    bands: int | str | Sequence[int | str] | None = None,
    *,
    cmap: str = "gray",
    figsize: tuple[int, int] = (10, 5),
    bins: int = 256,
    pmin: float = 2,
    pmax: float = 98,
    stretch: bool = False,
    colorbar: bool = False,
    colorbar_label: str | None = None,
    save_path: StrPath | None = None,
    title: str | None = None,
) -> None:
    """Plot each band alongside its value histogram.

    For every selected band, draws the raster (in spatial coordinates) and
    its histogram side by side on one row.

    Parameters
    ----------
    ds : EEORasterDataset
        Raster dataset to display.
    bands : int or str or sequence of (int or str) or None, default None
        Band(s) to plot, each a 1-based index or a band name; None plots
        every band. A sequence may mix indices and names.
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
        The stretch sets that panel's display limits only: pixel values are
        unchanged, and the histogram always bins the band's raw values.
    colorbar : bool, default False
        If True, draw a colorbar beside each raster panel, scaled in the
        band's own values. Arrowheads mark the ends where the stretch clips
        data.
    colorbar_label : str or None, default None
        Label for the colorbars; None uses each band's name, leaving an
        unnamed band's colorbar unlabelled.
    save_path : str or path-like or None, default None
        If given, save the figure to this path at 300 dpi.
    title : str or None, default None
        Optional figure title.

    Returns
    -------
    None
        Terminal operation; displays (and optionally saves) a figure.

    Raises
    ------
    IndexError
        If a band index is outside the range of available bands.
    ValidationError
        If ``bands`` names a band that is unknown or matches more than one
        band.

    Notes
    -----
    Reads each band decimated to the figure's display resolution (rasterio
    ``out_shape``, served from overviews when present); rasters already
    within the display budget, and NumPy-backed datasets, are read in full.
    For a decimated raster the histogram is computed from the decimated
    pixels, so bin counts are an approximation of the full-resolution
    histogram. A declared nodata value is excluded per the library's nodata
    contract: the sentinel is masked, so it is absent from the histogram, the
    stretch, and the colorbar, and those pixels render blank.
    Displays the figure with ``matplotlib.pyplot.show`` and, when
    ``save_path`` is given, writes it to disk as a side effect.

    Examples
    --------
    >>> ds.plot_raster_with_histogram(bands=[1, 2])
    >>> ds.plot_raster_with_histogram(bands=1, colorbar=True)
    """
    bands_list = _normalize_bands(ds, bands)

    fig, axes = plt.subplots(len(bands_list), 2, squeeze=False, figsize=figsize)

    for row, band in enumerate(bands_list):
        array, transform = _read_band_for_display(ds, band, figsize)
        # Limits scale the image panel only; the histogram bins raw values, so
        # its x-axis stays in the band's own units whatever the stretch does.
        draw_kwargs = _with_stretch_limits({}, array, pmin, pmax) if stretch else {}

        rioplot.show(array, transform=transform, ax=axes[row, 0], cmap=cmap, **draw_kwargs)
        axes[row, 1].hist(_valid_values(array), bins=bins)

        if colorbar:
            # rasterio.plot.show returns the axes, not the image it drew.
            images = axes[row, 0].get_images()
            _add_colorbar(
                fig, axes[row, 0], images[-1] if images else None, ds, band, colorbar_label
            )

        axes[row, 0].set_title(_band_label(ds, band))
        axes[row, 1].set_title(f"Histogram of {_band_label(ds, band).lower()}")

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
    bands: Sequence[int | str],
    *,
    stretch: bool = True,
    figsize: tuple[int, int] = (8, 8),
    pmin: float = 2,
    pmax: float = 98,
    title: str | None = None,
    save_path: StrPath | None = None,
) -> None:
    """Plot a three-band RGB (or false-colour) composite.

    Stacks the three requested bands into an RGB image, in the order given
    (first band -> red, second -> green, third -> blue).

    Parameters
    ----------
    ds : EEORasterDataset
        Raster dataset to display.
    bands : sequence of (int or str)
        Exactly three bands, each a 1-based index or a band name, mapped to
        R, G, B in order.
    stretch : bool, default True
        If True, apply percentile contrast stretching to each channel. On by
        default because the ``pmin``-``pmax`` (2-98) stretch renders most
        composites best (and integer rasters such as Sentinel-2 reflectance
        display as black without it); pass ``stretch=False`` for raw values.
    figsize : tuple of int, default (8, 8)
        Figure size in inches.
    pmin : float, default 2
        Lower percentile for the stretch (used only when ``stretch=True``).
    pmax : float, default 98
        Upper percentile for the stretch (used only when ``stretch=True``).
    title : str or None, default None
        Optional figure title.
    save_path : str or path-like or None, default None
        If given, save the figure to this path at 300 dpi.

    Returns
    -------
    None
        Terminal operation; displays (and optionally saves) a figure.

    Raises
    ------
    IndexError
        If a band index is outside the range of available bands.
    ValidationError
        If ``bands`` does not resolve to exactly three bands, or names a band
        that is unknown or matches more than one band.

    Notes
    -----
    Reads the three bands decimated to the figure's display resolution
    (rasterio ``out_shape``, served from overviews when present); rasters
    already within the display budget, and NumPy-backed datasets, are read
    in full. When ``stretch=True`` the channels are rescaled to floating-point
    ``[0, 1]`` before display, so integer rasters (e.g. Sentinel-2 reflectance)
    render correctly. A declared nodata value is excluded from each channel's
    stretch per the library's nodata contract, and a pixel that is nodata in
    any channel is transparent in the composite — but only on the stretched
    (floating-point) path, since RGBA needs floats in ``[0, 1]``; an
    unstretched integer composite renders its raw nodata values. Displays the
    figure with ``matplotlib.pyplot.show`` and, when ``save_path`` is given,
    writes it to disk as a side effect.

    Examples
    --------
    >>> ds.plot_composite(bands=(3, 2, 1))
    """
    bands_list = _normalize_bands(ds, bands)
    if len(bands_list) != 3:
        raise ValidationError(f"a composite needs exactly 3 bands (R, G, B); got {len(bands_list)}")
    channels = [_read_band_for_display(ds, b, figsize)[0] for b in bands_list]

    # A pixel that is nodata in any channel is nodata in the composite, per the
    # nodata contract's contagion rule.
    invalid = np.zeros(channels[0].shape, dtype=bool)
    for channel in channels:
        invalid |= np.ma.getmaskarray(channel)

    if stretch:
        # Stretch into float [0, 1]; stacking the float results keeps the
        # composite floating so the scaled values are not truncated to an
        # integer band dtype (which would render the image black).
        channels = [_percentile_stretch(channel, pmin, pmax) for channel in channels]

    composite = np.stack([np.ma.filled(channel, 0) for channel in channels], axis=-1)

    if invalid.any() and np.issubdtype(composite.dtype, np.floating):
        # RGBA needs floats in [0, 1], which only the stretched path produces;
        # an unstretched integer composite keeps its raw nodata values.
        alpha = (~invalid).astype(composite.dtype)
        composite = np.concatenate([composite, alpha[..., None]], axis=-1)

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
