.. Landing page. The section title is visually hidden (the hero carries the
   heading), and the hidden toctree at the bottom defines the navbar tabs:
   each entry becomes one tab, and that page's own toctree fills the sidebar.

:html_theme.sidebar_secondary.remove: true

Easy-EO
=======

.. container:: eeo-hero

   .. grid:: 1 1 2 2
      :gutter: 4
      :class-container: eeo-hero-grid

      .. grid-item::
         :class: eeo-hero-copy

         .. rst-class:: eeo-hero-eyebrow

         Earth observation · Python

         .. rst-class:: eeo-hero-title

         Chainable raster processing for Earth observation

         .. rst-class:: eeo-hero-lede

         Clip, resample, compute indices, and plot in one readable chain.
         Easy-EO wraps Rasterio, NumPy, and Matplotlib, so it reads every
         format ``rasterio`` supports.

         .. container:: eeo-hero-actions

            .. button-ref:: getting_started
               :ref-type: doc
               :class: eeo-btn eeo-btn-primary

               Get started

            .. button-ref:: tutorials
               :ref-type: doc
               :class: eeo-btn eeo-btn-ghost

               Browse tutorials

         .. rst-class:: eeo-hero-install

         ``pip install easy-eo``

      .. grid-item::
         :class: eeo-hero-code

         .. code-block:: python

            from eeo import load_raster

            nir = load_raster(path="nir.tif")
            red = load_raster(path="red.tif")

            ndvi = (
                nir.clip_raster_with_bbox(bbox=(0, 0, 1000, 1000))
                .resample(scale_factor=2)
                .normalized_difference(other=red)
            )
            ndvi.plot_raster_with_histogram(stretch=True)

Why Easy-EO?
------------

Working directly with libraries like Rasterio can be powerful, but they often require verbose
boilerplate code for simple operations such as raster reprojection, resampling, arithmetic
between rasters, clipping, mosaicking, or plotting. Easy-EO abstracts these routines into
**high-level, chainable methods**, allowing users to:

- Load satellite imagery of any area of interest **straight into your code** from a STAC
  catalog, without visiting a download portal or fetching full scenes — only your area
  is read (see :doc:`user_guide/loading_satellite_data`).
- Perform multiple operations in a single, readable chain.
- Persist intermediate results **in memory** without writing to disk unnecessarily.
- Automatically align rasters with differing shapes or coordinate reference systems.
- Return a consistent **EEORasterDataset** object from each operation, enabling further chaining.
- Handle many acquisitions of the same place as **one time series**: build cloud-free
  composites, group scenes by week, month, or year, and plot how a pixel changes over time
  (see :doc:`user_guide/time_series`).
- Use **terminal visualization methods** for plotting bands, composites, and histograms,
  which do not return EEORasterDataset but instead display results.

Key Features
------------

.. grid:: 1 2 3 3
   :gutter: 3
   :class-container: eeo-feature-grid

   .. grid-item-card:: :octicon:`plus-circle` Raster algebra
      :link: user_guide/ops
      :link-type: doc
      :class-card: eeo-card

      Pixel-wise addition, subtraction, multiplication, division, and power.
      Operator overloading allows ``+``, ``-``, ``*``, ``/``, and ``**``.

   .. grid-item-card:: :octicon:`graph` Spectral indices
      :link: user_guide/spectral_indices
      :link-type: doc
      :class-card: eeo-card

      Normalized-difference indices such as NDVI, or your own, returned as
      arrays or as datasets you can keep chaining.

   .. grid-item-card:: :octicon:`screen-full` Spatial operations
      :link: user_guide/preprocessing
      :link-type: doc
      :class-card: eeo-card

      Clip with bounding boxes or vector geometries, mosaic rasters, or stack
      them as new bands.

   .. grid-item-card:: :octicon:`sliders` Standardization
      :link: user_guide/preprocessing
      :link-type: doc
      :class-card: eeo-card

      Z-score, min–max, or percentile-based normalization.

   .. grid-item-card:: :octicon:`image` Visualization
      :link: user_guide/visualization
      :link-type: doc
      :class-card: eeo-card

      Bands, composites, histograms, or a raster beside its histogram, with
      percentile contrast stretching.

   .. grid-item-card:: :octicon:`globe` Satellite data from STAC
      :link: user_guide/loading_satellite_data
      :link-type: doc
      :class-card: eeo-card

      Search any STAC catalog and read only your area of interest, without
      downloading a full scene.

   .. grid-item-card:: :octicon:`cloud` Cloud masking
      :link: user_guide/masking_clouds
      :link-type: doc
      :class-card: eeo-card

      Remove cloud, shadow, and haze before they quietly change your answer.

   .. grid-item-card:: :octicon:`history` Time series
      :link: user_guide/time_series
      :link-type: doc
      :class-card: eeo-card

      Ordered, timestamped collections of scenes of the same place.

   .. grid-item-card:: :octicon:`arrow-switch` xarray interop
      :link: user_guide/xarray_interop
      :link-type: doc
      :class-card: eeo-card

      Convert to and from xarray for labelled arrays, dask, and the wider
      ecosystem.

Chainable Workflow
------------------

All methods in Easy-EO are designed to be chainable, except for visualization operations
which are terminal. For example:

.. code-block:: python

    from eeo import load_raster

    ds_nir = load_raster(path="path/to/nir.tif")
    ds_red = load_raster(path="path/to/red.tif")

    # Chainable example: clip -> resample -> compute NDVI -> multiply
    result = (
        ds_nir.clip_raster_with_bbox(bbox=(0, 0, 1000, 1000))
        .resample(scale_factor=2)
        .normalized_difference(other=ds_red)
        .multiply(other=100)
    )

Visualization is always done at the end of the chain:

.. code-block:: python

    # Terminal operation: display raster and histogram
    result.plot_raster_with_histogram(bands=[1,2], stretch=True)

Explore the Docs
----------------

.. grid:: 1 2 4 4
   :gutter: 3
   :class-container: eeo-explore-grid

   .. grid-item-card:: Getting Started
      :link: getting_started
      :link-type: doc
      :class-card: eeo-card eeo-card-explore

      Install Easy-EO and learn the core concepts.

   .. grid-item-card:: User Guide
      :link: user_guide/index
      :link-type: doc
      :class-card: eeo-card eeo-card-explore

      Each topic in depth, from loading data to time series.

   .. grid-item-card:: Tutorials
      :link: tutorials
      :link-type: doc
      :class-card: eeo-card eeo-card-explore

      Sixteen runnable notebooks, each openable in Colab.

   .. grid-item-card:: API Reference
      :link: modules/index
      :link-type: doc
      :class-card: eeo-card eeo-card-explore

      Every public class, function, and method.

.. toctree::
   :hidden:
   :maxdepth: 2

   Getting Started <getting_started>
   User Guide <user_guide/index>
   Tutorials <tutorials>
   API Reference <modules/index>
   Citation <citation>
