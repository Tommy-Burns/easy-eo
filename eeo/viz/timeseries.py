"""Terminal plotting functions for time series.

Two pictures answer most temporal questions. *What happened here?* is a line
through time at one place, which :func:`plot_trajectory` draws from the same
sample :meth:`eeo.EEOTimeSeries.extract_at` returns. *What did each date look
like?* is one small map per timestep, which :func:`plot_filmstrip` lays out as a
grid — the quick way to see which acquisitions are usable before compositing
them.

Both are terminal, like everything in :mod:`eeo.viz.plot`: they draw, optionally
save, and return nothing. Reads are decimated to the size each panel is actually
drawn at, so a filmstrip of full Sentinel-2 tiles costs a thumbnail per timestep
rather than a scene per timestep.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from eeo.common import resolve_band_index
from eeo.core.types import StrPath
from eeo.viz.plot import (
    _auto_grid,
    _band_label,
    _colorbar_extend,
    _grid_shape,
    _read_band_for_display,
    _resolve_figsize,
    _stretch_limits,
)

if TYPE_CHECKING:  # pragma: no cover - types only
    from eeo.timeseries.core import EEOTimeSeries

# One panel of a filmstrip, before the grid multiplies it up.
_PANEL_FIGSIZE = (4, 4)


def plot_trajectory(
    series: EEOTimeSeries,
    coordinates: Any,
    *,
    bands: Any = None,
    crs: Any = None,
    figsize: tuple[int, int] = (10, 4),
    title: str | None = None,
    save_path: StrPath | None = None,
    dpi: int = 300,
) -> None:
    """Plot what happened at one location, through time.

    One line per band, with a marker at every acquisition. A date where the pixel
    was nodata — under cloud, off the edge of the scene — draws no marker and
    breaks the line, so a gap looks like a gap rather than a dip to zero.

    Parameters
    ----------
    series : EEOTimeSeries
        Series to sample.
    coordinates : sequence of float
        ``(x, y)`` position, in the series' CRS unless ``crs`` says otherwise.
    bands : sequence of (int or str) or None, default None
        Which bands to draw, as 1-based indices or band names; None draws every
        band. One band on an index series is the usual case.
    crs : optional
        CRS the coordinates are given in — anything rasterio accepts, such as
        ``"EPSG:4326"`` — when that is not the series' own.
    figsize : tuple of int, default (10, 4)
        Figure size in inches. Wide by default: the x axis is a season.
    title : str or None, default None
        Figure title. None labels the plot with the sampled point.
    save_path : str or path-like or None, default None
        Write the figure here as well as showing it.
    dpi : int, default 300
        Resolution for ``save_path``.

    Returns
    -------
    None
        Terminal: displays the figure with ``matplotlib.pyplot.show`` and, with
        ``save_path``, writes it to disk as a side effect.

    Raises
    ------
    ValidationError
        For anything :meth:`eeo.EEOTimeSeries.extract_at` rejects — a point
        outside the series' extent, a band it does not have, coordinates that
        are not a pair.

    Notes
    -----
    Reads one pixel per timestep and band, so this costs the same as
    :meth:`eeo.EEOTimeSeries.extract_at` however large the scenes are.

    For anything beyond a look — a rolling mean, a resample, two locations on
    one axes — take the table from ``extract_at`` and plot it yourself; it is a
    pandas ``DataFrame``, and this function is a shortcut past the boilerplate,
    not a replacement for it.

    Examples
    --------
    >>> ndvi = ts.map(eeo.ndvi, red="B04", nir="B08", name="ndvi")  # doctest: +SKIP
    >>> ndvi.plot_trajectory((11.1, 46.6), crs="EPSG:4326")  # doctest: +SKIP
    """
    table = series.extract_at(coordinates, bands=bands, crs=crs)

    fig, ax = plt.subplots(figsize=figsize)
    for column in table.columns:
        # A marker per acquisition, so a single valid date between two gaps is
        # visible — a line alone would draw nothing there.
        ax.plot(table.index, table[column], marker="o", label=str(column))

    ax.set_xlabel("acquisition date")
    ax.set_ylabel(str(table.columns[0]) if len(table.columns) == 1 else "value")
    if len(table.columns) > 1:
        ax.legend()

    if title is None:
        x, y = table.attrs.get("x"), table.attrs.get("y")
        title = f"({x:g}, {y:g})" if x is not None and y is not None else None
    if title:
        ax.set_title(title)

    fig.autofmt_xdate()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def plot_filmstrip(
    series: EEOTimeSeries,
    *,
    band: int | str = 1,
    nrows: int | None = None,
    ncols: int | None = None,
    shared_scale: bool = True,
    pmin: float = 2,
    pmax: float = 98,
    cmap: Any = None,
    figsize: tuple[int, int] | None = None,
    title: str | None = None,
    save_path: StrPath | None = None,
    dpi: int = 300,
) -> None:
    """Plot one small map per timestep, laid out as a grid.

    The quick look at a series: which dates are clouded, when the field greened
    up, whether the flood had receded. Panels are titled with their acquisition
    date and ordered oldest first.

    Parameters
    ----------
    series : EEOTimeSeries
        Series to draw.
    band : int or str, default 1
        Which band each panel shows, as a 1-based index or a band name. One
        band, not several: a filmstrip compares dates, and mixing bands into the
        same grid would make it compare two things at once.
    nrows, ncols : int or None
        Grid to lay the panels out in. Both None chooses a near-square grid, so
        twelve timesteps are a 3x4 block rather than a twelve-storey strip.
    shared_scale : bool, default True
        Whether every panel uses the same colour limits, taken from the
        ``pmin``-``pmax`` percentiles of the whole series. True is what makes
        the panels comparable — with a scale each, a dark date and a bright one
        render identically and the figure misleads. False stretches each panel
        on its own, which is only for reading detail within a date.
    pmin : float, default 2
        Lower percentile for the colour limits.
    pmax : float, default 98
        Upper percentile for the colour limits.
    cmap : str or matplotlib.colors.Colormap or None, default None
        Colormap for the panels; None uses Matplotlib's default.
    figsize : tuple of int or None, default None
        Figure size in inches. None derives one from the grid.
    title : str or None, default None
        Figure title, drawn above the grid.
    save_path : str or path-like or None, default None
        Write the figure here as well as showing it.
    dpi : int, default 300
        Resolution for ``save_path``.

    Returns
    -------
    None
        Terminal: displays the figure with ``matplotlib.pyplot.show`` and, with
        ``save_path``, writes it to disk as a side effect.

    Raises
    ------
    ValidationError
        If ``band`` is not a band of the series, or if a requested grid cannot
        hold every timestep.

    Notes
    -----
    Each panel is read decimated to the size it is drawn at, not to the size of
    the whole figure, so the cost is a thumbnail per timestep however large the
    scenes are. Nodata pixels are masked and draw as gaps.

    A colorbar is drawn only when ``shared_scale`` is True, because one bar can
    only describe one scale — with a scale per panel there is nothing a single
    colorbar could honestly label.

    A long series makes a crowded figure. Slice it, or reduce it first:
    ``ts.resample_time("MS").composite().plot_filmstrip()`` is twelve panels for
    a year rather than forty.

    Examples
    --------
    >>> ts.plot_filmstrip(band="B04")  # doctest: +SKIP
    >>> ts.map(eeo.ndvi, red="B04", nir="B08").plot_filmstrip(cmap="RdYlGn")  # doctest: +SKIP
    """
    index = resolve_band_index(series.reference, band)
    datasets = list(series)

    if nrows is None and ncols is None:
        rows, cols = _auto_grid(len(datasets))
    else:
        rows, cols = _grid_shape(len(datasets), nrows, ncols)
    size = _resolve_figsize(figsize, _PANEL_FIGSIZE, rows, cols)
    # Constrained rather than tight layout, unlike the single-raster plots: a
    # colorbar spanning every axes is what tight_layout cannot place.
    fig, axes = plt.subplots(rows, cols, squeeze=False, figsize=size, layout="constrained")
    flat = axes.ravel()
    for ax in flat[len(datasets) :]:
        ax.axis("off")

    # The display budget is one panel's worth, not the whole figure's: a grid of
    # n panels drawn at figure resolution would read n times the pixels it can
    # show.
    panel_figsize = (size[0] / cols, size[1] / rows)
    panels = [_read_band_for_display(ds, index, panel_figsize)[0] for ds in datasets]

    limits: tuple[float, float] | None = None
    if shared_scale:
        # Percentiles of every panel together, which is what makes one colour
        # mean one value across the whole figure.
        stacked = np.ma.concatenate([np.ma.ravel(panel) for panel in panels])
        limits = _stretch_limits(stacked, pmin, pmax)

    image = None
    for ax, array, stamp in zip(flat, panels, series.timestamps, strict=False):
        draw: dict[str, Any] = {"cmap": cmap}
        if limits is not None:
            draw["vmin"], draw["vmax"] = limits
        elif not shared_scale:
            own = _stretch_limits(array, pmin, pmax)
            if own is not None:
                draw["vmin"], draw["vmax"] = own
        image = ax.imshow(array, **draw)
        ax.set_title(f"{stamp:%Y-%m-%d}", fontsize=9)
        ax.axis("off")

    if shared_scale and image is not None:
        fig.colorbar(
            image,
            ax=flat.tolist(),
            extend=_colorbar_extend(image),
            label=_band_label(series.reference, index),
        )

    if title:
        fig.suptitle(title)

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()
    plt.close(fig)
