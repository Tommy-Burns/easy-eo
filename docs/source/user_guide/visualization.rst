Visualization Functions
=======================

Easy-EO provides a small but expressive set of visualization utilities for
exploring raster datasets. Visualization functions are **termination
operations**: they consume an ``EEORasterDataset`` (or a list of datasets),
produce plots, and do not return a new dataset.

All visualization functions are designed to integrate naturally at the
end of a chainable workflow.

Overview
--------

Visualization in Easy-EO supports:
    - Plotting raster data as raw NumPy arrays (row/column space)
    - CRS-aware spatial plotting
    - Histogram inspection of raster values
    - Side-by-side raster and histogram views
    - Three-band composites (RGB or false-color)

Most plotting functions apply **percentile contrast stretching** by default
(a 2-98 percentile stretch), because that renders most Earth-observation
rasters best without any tuning.

Every ``bands`` argument accepts a 1-based band index, a band name, or a list
mixing the two (``bands=["red", 2, "blue"]``), and each subplot is titled with
the band's name when it has one. See :doc:`band_names`.

Percentile Contrast Stretching
------------------------------

Every visualization function accepts a ``stretch`` parameter. When
``stretch=True``, raster values are normalized using a percentile-based
contrast stretch:

.. math::

   x_{norm} = \frac{x - p_{min}}{p_{max} - p_{min}}

where ``pmin`` and ``pmax`` (defaulting to ``2`` and ``98``) are computed using
NaN-aware percentile estimation.

Defaults:
    - ``plot_band_array``, ``plot_raster`` and ``plot_composite`` default to
      ``stretch=True`` — so rasters display well out of the box. Pass
      ``stretch=False`` to show raw values.
    - ``plot_raster_with_histogram`` defaults to ``stretch=False``, so its
      histogram shows the **raw** value distribution (the point of pairing a
      raster with its histogram is to inspect real values before choosing a
      stretch). Pass ``stretch=True`` to stretch its raster panel.

Important behavior notes:
    - When ``stretch=False``, values are passed directly to Matplotlib and may be
      auto-scaled according to Matplotlib’s default behavior.
    - For the single-band plots (``plot_band_array``, ``plot_raster``,
      ``plot_raster_with_histogram``), ``stretch=True`` sets the subplot's
      **display limits** — ``vmin`` and ``vmax`` — to :math:`p_{min}` and
      :math:`p_{max}`. Matplotlib applies the normalization above when it
      renders, so the array itself keeps its own values and units. Passing an
      explicit ``vmin``/``vmax`` overrides the stretch.
    - ``plot_composite`` instead rescales each channel into ``[0, 1]``, because
      the three channels are stacked into a single RGB image. Its output is
      float-like even when the source bands are ``uint16`` — which is why
      integer rasters such as Sentinel-2 reflectance render correctly rather
      than as black.
    - A band with an empty percentile range (a constant band, or an all-nodata
      one) has no meaningful limits, so Matplotlib's own autoscaling applies.
    - Nodata pixels are excluded from the percentiles and are not drawn, per
      the :doc:`nodata contract <nodata_and_dtype>`: the stretch describes
      valid data only, and gaps render blank instead of as a colour.
    - Percentile stretching is intended for **visualization only** and does not
      modify the underlying dataset.

This approach is robust to outliers and commonly used for EO raster inspection.

Subplot Layout
--------------

``plot_band_array``, ``plot_raster`` and ``plot_histogram`` draw one subplot per
band, arranged **near-square** by default: four bands come out as a 2x2 block,
not a four-storey strip. Two and three panels stay a single row, six become
2x3, nine become 3x3.

Pass ``nrows`` or ``ncols`` to choose the grid yourself:

.. code-block:: python

   # four bands in one row instead of the default 2x2
   scene.plot_raster(ncols=4, figsize=(16, 4))

   # the same for several separate datasets
   plot_raster([ndvi, ndwi, evi, savi], bands=1, nrows=2)

Giving one of the two derives the other, so ``ncols=2`` with five panels gives
three rows and leaves the last cell blank. Giving both is honoured as long as
the grid has room; a grid too small to hold every panel raises
``ValidationError`` rather than silently dropping bands.

Notes:
    - Panels fill row-major, dataset-major within that: every band of the first
      dataset, then every band of the second.
    - Several datasets **and** several bands keep the **semantic** layout — rows
      are bands, columns are datasets — so band *i* of one dataset sits beside
      band *i* of the next. That alignment is the point of a multi-dataset
      figure, so it is never reflowed automatically; ``nrows``/``ncols`` still
      override it.
    - ``figsize`` follows the grid. Left unset, it is the function's usual size
      for a single row of panels, and for a taller grid the same width with the
      height set to keep the cells roughly square. Passing a ``figsize``
      disables that entirely.
    - ``plot_raster_with_histogram`` has no ``nrows``/``ncols``: its two columns
      are structural (raster beside histogram).

Colorbars
---------

``plot_band_array``, ``plot_raster`` and ``plot_raster_with_histogram`` accept
``colorbar=True``, which draws a scale beside each subplot in the band's own
values — index units for a spectral index, metres for a DEM, reflectance for a
raw band:

.. code-block:: python

   ndvi = scene.ndvi(red="red", nir="nir", name="NDVI")
   ndvi.plot_raster(cmap="RdYlGn", colorbar=True)

The label comes from the band's name, so an index named at creation labels its
own colorbar. Pass ``colorbar_label`` to override it — useful for units the
band name does not carry:

.. code-block:: python

   dem.plot_raster(cmap="Spectral_r", colorbar=True, colorbar_label="elevation (m)")

Notes:
    - The bar spans the **display limits**, not the full data range. With
      ``stretch=True`` those are the 2-98 percentiles, so arrowheads appear on
      the ends where values are clipped.
    - Each subplot gets its own bar. Bands in a grid carry unrelated ranges and
      often unrelated units, so a single shared scale would mislabel every
      panel but one.
    - A declared nodata value is excluded, so a ``-9999`` fill cannot stretch
      the scale to reach it. See :doc:`nodata_and_dtype`.
    - ``plot_composite`` takes no colorbar: an RGB composite maps three bands to
      colour channels, so there is no single scalar scale to label.

Saving Figures
--------------

Every plotting function writes the figure to disk when given ``save_path``, at
the resolution ``dpi`` asks for:

.. code-block:: python

   scene.plot_raster(cmap="RdYlGn", save_path="ndvi.png")             # 300 dpi
   scene.plot_raster(cmap="RdYlGn", save_path="ndvi.png", dpi=120)    # preferably for the web

The default, 300, is a print resolution and multiplies with ``figsize``: a 15x5
figure lands near 4500x1500 pixels and well over 10 MB. That is the wrong size
for a figure committed to a repository, embedded in a README, or shown on a web
page — pass ``dpi=100`` to ``dpi=150`` for those.

Raising ``dpi`` much above 200 buys less than it appears to for the image
panels. Bands are read decimated to the figure's on-screen display budget
(``figsize`` times the Matplotlib ``figure.dpi``, oversampled twice over), so
past that point a larger canvas holds the same pixels. Histogram panels, being
drawn rather than read, do keep sharpening.

Plot Band Arrays (Array Coordinates)
------------------------------------

.. function:: plot_band_array(ds, bands=None, *, cmap="gray", figsize=None, nrows=None, ncols=None, stretch=True, pmin=2, pmax=98, colorbar=False, colorbar_label=None, title=None, save_path=None, dpi=300, **imshow_kwargs)

   Plot raster bands as NumPy arrays using row/column coordinates.

   Axes correspond to array indices, not spatial (CRS) coordinates.

   :param ds: One or more raster datasets.
   :type ds: EEORasterDataset or list[EEORasterDataset]
   :param bands: Band index or indices (1-based). If ``None``, all bands are plotted.
   :type bands: int | Sequence[int] | None
   :param cmap: Matplotlib colormap.
   :param nrows: Subplot rows; unset with ``ncols``, the grid is near-square.
   :param ncols: Subplot columns; giving either reflows the panels into that grid.
   :param stretch: Apply percentile contrast stretching. Defaults to ``True``; pass ``False`` for raw values.
   :param pmin: Lower percentile used when ``stretch=True``.
   :param pmax: Upper percentile used when ``stretch=True``.
   :param colorbar: Draw a colorbar beside each subplot, in band units.
   :param colorbar_label: Colorbar label; ``None`` uses the band's name.
   :param title: Optional figure title.
   :param save_path: File path if the figure should be saved to disk.
   :param dpi: Resolution of the saved file, in dots per inch (default ``300``).
       Lower it for a figure committed to a repository or shown on the web.
   :param imshow_kwargs: Additional keyword arguments passed to
       ``matplotlib.pyplot.imshow``.

Plot Raster (Spatial Coordinates)
---------------------------------

.. function:: plot_raster(ds, bands=None, *, cmap="gray", figsize=None, nrows=None, ncols=None, stretch=True, pmin=2, pmax=98, colorbar=False, colorbar_label=None, title=None, save_path=None, dpi=300, **show_kwargs)

   Plot raster bands in spatial (CRS-aware) coordinates.

   Internally uses ``rasterio.plot.show`` and preserves the dataset’s affine
   transform.

   :param ds: One or more raster datasets.
   :type ds: EEORasterDataset or list[EEORasterDataset]
   :param bands: Band index or indices (1-based). If ``None``, all bands are plotted.
   :param cmap: Matplotlib colormap.
   :param figsize: Size of the matplotlib figure; ``None`` derives one from the grid.
   :param nrows: Subplot rows; unset with ``ncols``, the grid is near-square.
   :param ncols: Subplot columns; giving either reflows the panels into that grid.
   :param stretch: Apply percentile contrast stretching. Defaults to ``True``; pass ``False`` for raw values.
   :param pmin: Lower percentile used when ``stretch=True``.
   :param pmax: Upper percentile used when ``stretch=True``.
   :param colorbar: Draw a colorbar beside each subplot, in band units.
   :param colorbar_label: Colorbar label; ``None`` uses the band's name.
   :param title: Optional figure title.
   :param save_path: File path if the figure should be saved to disk.
   :param dpi: Resolution of the saved file, in dots per inch (default ``300``).
       Lower it for a figure committed to a repository or shown on the web.
   :param show_kwargs: Additional keyword arguments passed to
       ``rasterio.plot.show``.

Plot Histogram
--------------

.. function:: plot_histogram(ds, bands=None, *, bins=256, figsize=None, nrows=None, ncols=None, log=False, title=None, save_path=None, dpi=300, **hist_kwargs)

   Plot histograms of raster band values.

   Histogram values are computed from flattened band arrays. Non-finite values
   are ignored.

   :param ds: One or more raster datasets.
   :type ds: EEORasterDataset or list[EEORasterDataset]
   :param bands: Band index or indices (1-based). If ``None``, all bands are plotted.
   :param bins: Number of histogram bins.
   :param figsize: Size of the matplotlib figure; ``None`` derives one from the grid.
   :param nrows: Subplot rows; unset with ``ncols``, the grid is near-square.
   :param ncols: Subplot columns; giving either reflows the panels into that grid.
   :param log: Use a logarithmic scale on the y-axis.
   :param title: Optional figure title.
   :param save_path: File path if the figure should be saved to disk.
   :param dpi: Resolution of the saved file, in dots per inch (default ``300``).
       Lower it for a figure committed to a repository or shown on the web.
   :param hist_kwargs: Additional keyword arguments passed to
       ``matplotlib.pyplot.hist``.

Plot Raster with Histogram
--------------------------

.. function:: plot_raster_with_histogram(ds, bands=None, *, cmap="gray", figsize: tuple[int, int] = (10, 5), bins=256, pmin=2, pmax=98, stretch=False, colorbar=False, colorbar_label=None, title=None, save_path=None, dpi=300)

   Plot raster bands alongside their corresponding histograms.

   Each band is shown in spatial coordinates together with its value
   distribution.

   :param ds: Raster dataset.
   :type ds: EEORasterDataset
   :param bands: Band index or indices (1-based). If ``None``, all bands are plotted.
   :param cmap: Matplotlib colormap.
   :param figsize: Size of the matplotlib figure; ``None`` derives one from the grid.
   :param bins: Number of histogram bins.
   :param stretch: Apply percentile contrast stretching to the raster display. Defaults to ``False`` so the histogram shows the raw value distribution.
   :param pmin: Lower percentile used when ``stretch=True``.
   :param pmax: Upper percentile used when ``stretch=True``.
   :param colorbar: Draw a colorbar beside each raster panel, in band units.
   :param colorbar_label: Colorbar label; ``None`` uses the band's name.
   :param title: Optional figure title.
   :param save_path: File path if the figure should be saved to disk.
   :param dpi: Resolution of the saved file, in dots per inch (default ``300``).
       Lower it for a figure committed to a repository or shown on the web.

Plot Composite (RGB / False-Color)
----------------------------------

.. function:: plot_composite(ds, bands, *, stretch=True, figsize=(8, 8), pmin=2, pmax=98, title=None, save_path=None, dpi=300)

   Plot a three-band raster composite (e.g., RGB or false-color).

   Bands are stacked in the order provided and displayed using Matplotlib.

   :param ds: Raster dataset.
   :type ds: EEORasterDataset
   :param bands: Tuple of three band indices ``(R, G, B)``.
   :param stretch: Apply percentile contrast stretching independently to each band. Defaults to ``True``; pass ``False`` for raw values.
   :param figsize: Size of the matplotlib figure.
   :param pmin: Lower percentile used when ``stretch=True``.
   :param pmax: Upper percentile used when ``stretch=True``.
   :param title: Optional figure title.
   :param save_path: File path if the figure should be saved to disk.
   :param dpi: Resolution of the saved file, in dots per inch (default ``300``).
       Lower it for a figure committed to a repository or shown on the web.

   .. note::

      ``stretch`` is ``True`` by default, so a 2-98 percentile stretch is applied
      to each band — this is what makes integer rasters (e.g. Sentinel-2
      reflectance) display correctly rather than as black. When ``stretch=False``,
      composite values are passed directly to Matplotlib and may be auto-scaled
      depending on their data range.
