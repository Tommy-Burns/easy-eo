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

Most plotting functions support optional **percentile contrast stretching**
for improved visual interpretation.

Percentile Contrast Stretching
------------------------------

Several visualization functions accept a ``stretch`` parameter. When
``stretch=True``, raster values are normalized using a percentile-based
contrast stretch:

.. math::

   x_{norm} = \frac{x - p_{min}}{p_{max} - p_{min}}

where ``pmin`` and ``pmax`` are computed using NaN-aware percentile estimation.

Important behavior notes:
    - When ``stretch=False``, values are passed directly to Matplotlib and may be
      auto-scaled according to Matplotlib’s default behavior.
    - When ``stretch=True``, values are explicitly normalized into the range
      ``[0, 1]``.
    - The output becomes float-like, even if the original raster bands are
      ``uint16`` or integer types.
    - Percentile stretching is intended for **visualization only** and does not
      modify the underlying dataset.

This approach is robust to outliers and commonly used for EO raster inspection.

Plot Band Arrays (Array Coordinates)
------------------------------------

.. function:: plot_band_array(ds, bands=None, *, cmap="gray", figsize: tuple[int, int] = (8, 8), stretch=False, pmin=2, pmax=98, title=None, save_path=None, **imshow_kwargs)

   Plot raster bands as NumPy arrays using row/column coordinates.

   Axes correspond to array indices, not spatial (CRS) coordinates.

   :param ds: One or more raster datasets.
   :type ds: EEORasterDataset or list[EEORasterDataset]
   :param bands: Band index or indices (1-based). If ``None``, all bands are plotted.
   :type bands: int | Sequence[int] | None
   :param cmap: Matplotlib colormap.
   :param stretch: Apply percentile contrast stretching.
   :param pmin: Lower percentile used when ``stretch=True``.
   :param pmax: Upper percentile used when ``stretch=True``.
   :param title: Optional figure title.
   :param save_path: File path if the figure should be saved to disk.
   :param imshow_kwargs: Additional keyword arguments passed to
       ``matplotlib.pyplot.imshow``.

Plot Raster (Spatial Coordinates)
---------------------------------

.. function:: plot_raster(ds, bands=None, *, cmap="gray", figsize: tuple[int, int] = (10, 5), stretch=False, pmin=2, pmax=98, title=None, save_path=None, **show_kwargs)

   Plot raster bands in spatial (CRS-aware) coordinates.

   Internally uses ``rasterio.plot.show`` and preserves the dataset’s affine
   transform.

   :param ds: One or more raster datasets.
   :type ds: EEORasterDataset or list[EEORasterDataset]
   :param bands: Band index or indices (1-based). If ``None``, all bands are plotted.
   :param cmap: Matplotlib colormap.
   :param figsize: Size of the matplotlib figure
   :param stretch: Apply percentile contrast stretching.
   :param pmin: Lower percentile used when ``stretch=True``.
   :param pmax: Upper percentile used when ``stretch=True``.
   :param title: Optional figure title.
   :param save_path: File path if the figure should be saved to disk.
   :param show_kwargs: Additional keyword arguments passed to
       ``rasterio.plot.show``.

Plot Histogram
--------------

.. function:: plot_histogram(ds, bands=None, *, bins=256, figsize: tuple[int, int] = (10, 5), log=False, title=None, save_path=None, **hist_kwargs)

   Plot histograms of raster band values.

   Histogram values are computed from flattened band arrays. Non-finite values
   are ignored.

   :param ds: One or more raster datasets.
   :type ds: EEORasterDataset or list[EEORasterDataset]
   :param bands: Band index or indices (1-based). If ``None``, all bands are plotted.
   :param bins: Number of histogram bins.
   :param figsize: Size of the matplotlib figure
   :param log: Use a logarithmic scale on the y-axis.
   :param title: Optional figure title.
   :param save_path: File path if the figure should be saved to disk.
   :param hist_kwargs: Additional keyword arguments passed to
       ``matplotlib.pyplot.hist``.

Plot Raster with Histogram
--------------------------

.. function:: plot_raster_with_histogram(ds, bands=None, *, cmap="gray", figsize: tuple[int, int] = (10, 5), bins=256, pmin=2, pmax=98, stretch=False, sharey=False, title=None, save_path=None)

   Plot raster bands alongside their corresponding histograms.

   Each band is shown in spatial coordinates together with its value
   distribution.

   :param ds: Raster dataset.
   :type ds: EEORasterDataset
   :param bands: Band index or indices (1-based). If ``None``, all bands are plotted.
   :param cmap: Matplotlib colormap.
   :param figsize: Size of the matplotlib figure
   :param bins: Number of histogram bins.
   :param stretch: Apply percentile contrast stretching to the raster display.
   :param pmin: Lower percentile used when ``stretch=True``.
   :param pmax: Upper percentile used when ``stretch=True``.
   :param sharey: Share the y-axis between histogram plots.
   :param title: Optional figure title.
   :param save_path: File path if the figure should be saved to disk.

Plot Composite (RGB / False-Color)
----------------------------------

.. function:: plot_composite(ds, bands, *, stretch=False, figsize=(8, 8), pmin=2, pmax=98, title=None, save_path=None)

   Plot a three-band raster composite (e.g., RGB or false-color).

   Bands are stacked in the order provided and displayed using Matplotlib.

   :param ds: Raster dataset.
   :type ds: EEORasterDataset
   :param bands: Tuple of three band indices ``(R, G, B)``.
   :param stretch: Apply percentile contrast stretching independently to each band.
   :param figsize: Size of the matplotlib figure.
   :param pmin: Lower percentile used when ``stretch=True``.
   :param pmax: Upper percentile used when ``stretch=True``.
   :param title: Optional figure title.
   :param save_path: File path if the figure should be saved to disk.

   .. note::

      When ``stretch=False``, composite values are passed directly to Matplotlib
      and may be auto-scaled depending on their data range.
