Sample Data
===========

Easy-EO ships a tiny, curated sample so every tutorial and quickstart runs in
minutes without hunting for data. The helpers in :mod:`eeo.datasets` download a
hosted Sentinel-2 / Copernicus-DEM subset on first use, cache it locally, and
verify it against a checksum baked into the package — so a fetch is instant
after the first call and never returns corrupt data. Downloading uses only the
Python standard library, so it adds **no extra dependency**.

.. seealso::

   :doc:`band_names` for how the loaded bands are addressed by name, and
   :doc:`spectral_indices` for computing NDVI and friends on the sample.

-----

Loading the sample
------------------

``load()`` returns a ready-to-use dataset for rasters — band names and the
acquisition timestamp are already set — so you can go straight to analysis:

.. code-block:: python

   import eeo

   scene = eeo.datasets.load("sentinel2_small")
   scene.band_names          # ['blue', 'green', 'red', 'nir']
   scene.timestamp           # 2023-09-07 10:00:31+00:00

   ndvi = scene.ndvi(red="red", nir="nir")
   ndvi.plot_raster()

Vector datasets have no raster representation, so ``load()`` returns their
cached path — read them with GeoPandas:

.. code-block:: python

   import geopandas as gpd

   boundary = gpd.read_file(eeo.datasets.load("sentinel2_small_boundary"))
   clipped = scene.clip_raster_with_vector(boundary)

Getting file paths with ``fetch()``
-----------------------------------

When you want the files rather than an opened dataset — to pass to another
library, or to work with the Cloud-Optimized GeoTIFFs directly — use
``fetch()``. It returns a single :class:`~pathlib.Path` for a one-file dataset,
or a list of paths (in band order) for a multi-file one:

.. code-block:: python

   scene_path = eeo.datasets.fetch("sentinel2_small")         # Path (one 4-band file)
   boundary_path = eeo.datasets.fetch("sentinel2_small_boundary")  # Path
   band_paths = eeo.datasets.fetch("sentinel2_small_bands")   # [Path, Path, Path, Path]

Available datasets
------------------

List everything with ``eeo.datasets.available()``; describe one with
``eeo.datasets.info(name)``.

.. list-table::
   :header-rows: 1
   :widths: 26 10 64

   * - Name
     - Kind
     - Contents
   * - ``sentinel2_small``
     - raster
     - Sentinel-2 L2A blue/green/red/nir as one 4-band file, 1024×1024 @ 10 m,
       EPSG:32633.
   * - ``sentinel2_small_cog``
     - raster
     - Cloud-Optimized GeoTIFF variant of the 4-band stack (HTTP range-read).
   * - ``sentinel2_small_bands``
     - raster
     - The same four bands as *separate* single-band files (stacked in memory
       on load); handy for workflows that treat bands as individual rasters.
   * - ``dem_small``
     - raster
     - Copernicus GLO-30 DEM warped onto the same grid (float32 metres).
   * - ``dem_small_cog``
     - raster
     - Cloud-Optimized GeoTIFF variant of the DEM.
   * - ``sentinel2_small_boundary``
     - vector
     - Region-of-interest polygon (GeoPackage, EPSG:4326) inside the footprint.

``sentinel2_small`` is a single pre-stacked 4-band GeoTIFF, so ``load`` opens it
lazily and ``fetch`` returns one path; ``sentinel2_small_bands`` is the same
imagery split across four single-band files, read into an in-memory stack on
load and returned by ``fetch`` as a list of paths. All names share one
1024×1024 grid (the DEM is warped to match), so the imagery, elevation, and
boundary overlay pixel-for-pixel.

Where files are cached
----------------------

Files are cached under ``~/.cache/easy-eo`` by default. The location is
resolved as:

1. ``$EEO_DATA_DIR`` if set,
2. ``$XDG_CACHE_HOME/easy-eo`` if ``XDG_CACHE_HOME`` is set,
3. ``~/.cache/easy-eo`` otherwise.

A cached file whose checksum still matches is reused untouched; a missing or
corrupted file is transparently re-downloaded.

Licensing and attribution
--------------------------

The sample is derived from open Copernicus data. If you redistribute figures or
data made from it, carry the attribution — ``eeo.datasets.info(name)`` prints
the exact text:

- **Sentinel-2:** *Contains modified Copernicus Sentinel-2 L2A data 2023 (tile
  T33UUP, acquired 2023-09-07), processed by ESA; accessed via Microsoft
  Planetary Computer.*
- **Copernicus DEM GLO-30:** *© DLR e.V. 2010–2014 and © Airbus Defence and
  Space GmbH 2014–2018, provided under COPERNICUS by the European Union and
  ESA; accessed via Microsoft Planetary Computer.*
