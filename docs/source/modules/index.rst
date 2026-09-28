API Reference
=============

The public API of Easy-EO, one page per subpackage. Most of it is reachable
as chainable methods on :class:`~eeo.core.core.EEORasterDataset`.

.. grid:: 1 2 3 3
   :gutter: 3
   :class-container: eeo-feature-grid

   .. grid-item-card:: ``eeo.core``
      :link: core
      :link-type: doc
      :class-card: eeo-card eeo-card-api

      Core dataset class, loaders, decorators, and backend adapters.

   .. grid-item-card:: ``eeo.core.adapters``
      :link: adapters
      :link-type: doc
      :class-card: eeo-card eeo-card-api

      Backend adapters for NumPy-, rasterio- and xarray-backed rasters.

   .. grid-item-card:: ``eeo.analysis``
      :link: analysis
      :link-type: doc
      :class-card: eeo-card eeo-card-api

      Spectral indices and pixel statistics.

   .. grid-item-card:: ``eeo.ops``
      :link: ops
      :link-type: doc
      :class-card: eeo-card eeo-card-api

      Chainable raster algebra and band/tile merging.

   .. grid-item-card:: ``eeo.preprocessing``
      :link: preprocessing
      :link-type: doc
      :class-card: eeo-card eeo-card-api

      Clip, resample, reproject, normalize, and masking.

   .. grid-item-card:: ``eeo.timeseries``
      :link: timeseries
      :link-type: doc
      :class-card: eeo-card eeo-card-api

      Ordered, timestamped collections of rasters.

   .. grid-item-card:: ``eeo.viz``
      :link: viz
      :link-type: doc
      :class-card: eeo-card eeo-card-api

      Terminal visualization helpers for rasters.

   .. grid-item-card:: ``eeo.io``
      :link: io
      :link-type: doc
      :class-card: eeo-card eeo-card-api

      Data access and exchange beyond the local filesystem.

   .. grid-item-card:: ``eeo.datasets``
      :link: datasets
      :link-type: doc
      :class-card: eeo-card eeo-card-api

      Curated sample datasets with checksum-verified caching.

.. toctree::
   :hidden:
   :maxdepth: 1

   core
   adapters
   analysis
   ops
   preprocessing
   timeseries
   viz
   io
   datasets
