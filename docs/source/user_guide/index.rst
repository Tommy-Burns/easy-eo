User Guide
==========

Each page covers one part of Easy-EO in depth. If you are new, start with
:doc:`core_dataset`, then follow the topic you need.

.. grid:: 1 2 2 2
   :gutter: 3
   :class-container: eeo-feature-grid

   .. grid-item-card:: :octicon:`stack` Fundamentals
      :class-card: eeo-card

      - :doc:`core_dataset`
      - :doc:`band_names`
      - :doc:`nodata_and_dtype`
      - :doc:`../backends`

   .. grid-item-card:: :octicon:`download` Loading data
      :class-card: eeo-card

      - :doc:`sample_data`
      - :doc:`loading_satellite_data`
      - :doc:`loading_downloaded_scenes`
      - :doc:`masking_clouds`

   .. grid-item-card:: :octicon:`beaker` Processing & analysis
      :class-card: eeo-card

      - :doc:`ops`
      - :doc:`spectral_indices`
      - :doc:`preprocessing`
      - :doc:`statistical_locations`
      - :doc:`time_series`

   .. grid-item-card:: :octicon:`image` Visualization & scale
      :class-card: eeo-card

      - :doc:`visualization`
      - :doc:`xarray_interop`
      - :doc:`large_rasters`

.. toctree::
   :hidden:
   :caption: Fundamentals

   core_dataset
   band_names
   nodata_and_dtype
   ../backends

.. toctree::
   :hidden:
   :caption: Loading data

   sample_data
   loading_satellite_data
   loading_downloaded_scenes
   masking_clouds

.. toctree::
   :hidden:
   :caption: Processing & analysis

   ops
   spectral_indices
   preprocessing
   statistical_locations
   time_series

.. toctree::
   :hidden:
   :caption: Visualization & scale

   visualization
   xarray_interop
   large_rasters
