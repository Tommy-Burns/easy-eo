Tutorials
=========

Sixteen runnable notebooks, grouped from first install through to complete
analyses. Every notebook is standalone — start wherever the topic you need is —
and none of them depends on files in the repository or on a working directory:
they get their data from :func:`~eeo.datasets.load_sample_dataset` or from a
STAC catalog.

**Run them in Colab.** Click any |colab| badge below and the notebook opens in a
hosted environment with nothing to install locally — its first cell detects Colab
and installs Easy-EO with the extras that notebook needs. Colab discards its
filesystem at the end of a session, so the sample data is downloaded again each
time you start one.

**Run them locally.** Clone the repository, or download a single notebook from
its source link:

.. code-block:: bash

   pip install "easy-eo[stac,xarray]" jupyterlab
   jupyter lab

Notebooks marked |net| query a live STAC catalog: they need network access and
the ``stac`` extra, and are stored without outputs. The rest run offline once
the sample data is cached, and are stored with their outputs.

-----

Getting started
---------------

`Installation and setup <https://github.com/Tommy-Burns/easy-eo/blob/main/examples/00_getting_started/01_installation_and_setup.ipynb>`_
   Check the geospatial stack, see which optional extras are available, and warm the sample cache.

   .. image:: https://colab.research.google.com/assets/colab-badge.svg
      :target: https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/00_getting_started/01_installation_and_setup.ipynb
      :alt: Open Installation and setup in Colab

`Quickstart: NDVI <https://github.com/Tommy-Burns/easy-eo/blob/main/examples/00_getting_started/02_quickstart_ndvi.ipynb>`_
   Open a scene, compute a vegetation index, plot it, save it — the shortest useful workflow.

   .. image:: https://colab.research.google.com/assets/colab-badge.svg
      :target: https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/00_getting_started/02_quickstart_ndvi.ipynb
      :alt: Open Quickstart: NDVI in Colab

Fundamentals
------------

`Reading and inspecting <https://github.com/Tommy-Burns/easy-eo/blob/main/examples/01_fundamentals/01_reading_and_inspecting.ipynb>`_
   What ``load_raster`` gives you, deferred reads, ``describe``, metadata, band names, ``load_array``.

   .. image:: https://colab.research.google.com/assets/colab-badge.svg
      :target: https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/01_fundamentals/01_reading_and_inspecting.ipynb
      :alt: Open Reading and inspecting in Colab

`Clipping and mosaicking <https://github.com/Tommy-Burns/easy-eo/blob/main/examples/01_fundamentals/02_clip_and_mosaic.ipynb>`_
   Clip to a vector or a bbox with ``crop``/``invert``/``all_touched``, then merge tiles side by side.

   .. image:: https://colab.research.google.com/assets/colab-badge.svg
      :target: https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/01_fundamentals/02_clip_and_mosaic.ipynb
      :alt: Open Clipping and mosaicking in Colab

`Reprojecting and resampling <https://github.com/Tommy-Burns/easy-eo/blob/main/examples/01_fundamentals/03_reproject_and_resample.ipynb>`_
   Change CRS, change pixel size, and choose a resampling method that suits the data.

   .. image:: https://colab.research.google.com/assets/colab-badge.svg
      :target: https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/01_fundamentals/03_reproject_and_resample.ipynb
      :alt: Open Reprojecting and resampling in Colab

`Band algebra and indices <https://github.com/Tommy-Burns/easy-eo/blob/main/examples/01_fundamentals/04_band_algebra_and_indices.ipynb>`_
   Arithmetic on rasters, integer wraparound, safe division, the six spectral indices, normalization.

   .. image:: https://colab.research.google.com/assets/colab-badge.svg
      :target: https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/01_fundamentals/04_band_algebra_and_indices.ipynb
      :alt: Open Band algebra and indices in Colab

`Stacking bands <https://github.com/Tommy-Burns/easy-eo/blob/main/examples/01_fundamentals/05_stacking_bands.ipynb>`_
   Build a multi-band scene from per-band files, and why stacking is not mosaicking.

   .. image:: https://colab.research.google.com/assets/colab-badge.svg
      :target: https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/01_fundamentals/05_stacking_bands.ipynb
      :alt: Open Stacking bands in Colab

`Statistics and extraction <https://github.com/Tommy-Burns/easy-eo/blob/main/examples/01_fundamentals/06_statistics_and_extraction.ipynb>`_
   Min/max/mean/percentile pixels, sampling values at coordinates, area statistics.

   .. image:: https://colab.research.google.com/assets/colab-badge.svg
      :target: https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/01_fundamentals/06_statistics_and_extraction.ipynb
      :alt: Open Statistics and extraction in Colab

`Visualization <https://github.com/Tommy-Burns/easy-eo/blob/main/examples/01_fundamentals/07_visualization.ipynb>`_
   Every plot function, contrast stretching, composites, and dropping down to Matplotlib.

   .. image:: https://colab.research.google.com/assets/colab-badge.svg
      :target: https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/01_fundamentals/07_visualization.ipynb
      :alt: Open Visualization in Colab

Data access
-----------

`The sample data helper <https://github.com/Tommy-Burns/easy-eo/blob/main/examples/02_data_access/01_sample_data.ipynb>`_
   The bundled sample, lazy handles, caching, COGs, attribution.

   .. image:: https://colab.research.google.com/assets/colab-badge.svg
      :target: https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/02_data_access/01_sample_data.ipynb
      :alt: Open The sample data helper in Colab

`Searching and loading from STAC <https://github.com/Tommy-Burns/easy-eo/blob/main/examples/02_data_access/02_stac_search_and_load.ipynb>`_ |net|
   Search a live catalog, load assets over HTTP, crop to an AOI, work with mixed resolutions.

   .. image:: https://colab.research.google.com/assets/colab-badge.svg
      :target: https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/02_data_access/02_stac_search_and_load.ipynb
      :alt: Open Searching and loading from STAC in Colab

`xarray interop <https://github.com/Tommy-Burns/easy-eo/blob/main/examples/02_data_access/03_xarray_interop.ipynb>`_
   ``to_xarray()`` / ``from_xarray()``, and when to reach for xarray instead.

   .. image:: https://colab.research.google.com/assets/colab-badge.svg
      :target: https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/02_data_access/03_xarray_interop.ipynb
      :alt: Open xarray interop in Colab

Real-world analyses
-------------------

`Flood mapping with NDWI <https://github.com/Tommy-Burns/easy-eo/blob/main/examples/03_real_world/01_flood_mapping_ndwi.ipynb>`_ |net|
   Pakistan, 2022: pre- and post-monsoon NDWI, permanent water removed, inundated cropland quantified.

   .. image:: https://colab.research.google.com/assets/colab-badge.svg
      :target: https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/03_real_world/01_flood_mapping_ndwi.ipynb
      :alt: Open Flood mapping with NDWI in Colab

`Vegetation health and drought <https://github.com/Tommy-Burns/easy-eo/blob/main/examples/03_real_world/02_vegetation_health_and_drought.ipynb>`_ |net|
   California's Central Valley: NDVI, NDMI, SAVI and EVI, and stress that greenness alone misses.

   .. image:: https://colab.research.google.com/assets/colab-badge.svg
      :target: https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/03_real_world/02_vegetation_health_and_drought.ipynb
      :alt: Open Vegetation health and drought in Colab

`Urban footprint and land cover <https://github.com/Tommy-Burns/easy-eo/blob/main/examples/03_real_world/03_urban_footprint_land_cover.ipynb>`_ |net|
   Cairo: why NDBI alone calls desert built-up, and a four-class land-cover map that does not.

   .. image:: https://colab.research.google.com/assets/colab-badge.svg
      :target: https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/03_real_world/03_urban_footprint_land_cover.ipynb
      :alt: Open Urban footprint and land cover in Colab

`Terrain analysis with a DEM <https://github.com/Tommy-Burns/easy-eo/blob/main/examples/03_real_world/04_terrain_analysis_with_dem.ipynb>`_
   Slope, aspect and hillshade from elevation, hypsometry, and terrain against land cover.

   .. image:: https://colab.research.google.com/assets/colab-badge.svg
      :target: https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/03_real_world/04_terrain_analysis_with_dem.ipynb
      :alt: Open Terrain analysis with a DEM in Colab

-----

.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :alt: Open in Colab

.. |net| replace:: (needs network)

.. seealso::

   :doc:`getting_started` for the same ground in prose,
   :doc:`user_guide/sample_data` for the data the offline notebooks use, and
   :doc:`user_guide/loading_satellite_data` for the STAC workflow the
   real-world notebooks are built on.
