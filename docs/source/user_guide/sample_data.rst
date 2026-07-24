Sample Data
===========

Easy-EO ships a tiny, curated sample so every tutorial and quickstart runs in
minutes without hunting for data. :func:`~eeo.datasets.load_sample_dataset`
returns a namespace whose attributes are the individual sample files; each is
downloaded on first use, cached locally, and verified against a checksum baked
into the package — so a fetch is instant after the first call and never returns
corrupt data. Downloading uses only the Python standard library, so it adds
**no extra dependency**.

.. seealso::

   :doc:`band_names` for how the loaded bands are addressed by name, and
   :doc:`spectral_indices` for computing NDVI and friends on the sample.

-----

Loading a sample
----------------

Call :func:`~eeo.datasets.load_sample_dataset` once, then open any file by
dotted name with :func:`~eeo.load_raster` — no string keys, and your editor
autocompletes the available names:

.. code-block:: python

   from eeo.datasets import load_sample_dataset
   from eeo import load_raster

   sd = load_sample_dataset()                     # instant — no download

   scene = load_raster(sd.sentinel2_cog_stacked)  # downloads that one file
   dem = load_raster(sd.copernicus_dem)
   blue = load_raster(sd.sentinel2_blue)

   ndvi = scene.ndvi(red="red", nir="nir")
   ndvi.plot_raster()

Each attribute is *lazy*: holding ``sd.copernicus_dem`` touches no network — the
file is downloaded and checksum-verified only when it is actually opened. Nothing
is fetched that you do not use. To warm the whole cache up front (before going
offline), pass ``load_sample_dataset(prefetch=True)``.

The ``boundary`` sample is a vector, so read it with GeoPandas rather than
``load_raster`` (the handle is a path, so pass it straight in):

.. code-block:: python

   import geopandas as gpd

   roi = gpd.read_file(sd.boundary)
   clipped = scene.clip_raster_with_vector(roi)

Need the raw cached path (to hand to another library, or to inspect the
Cloud-Optimized GeoTIFFs directly)? Use ``.path``:

.. code-block:: python

   scene_path = sd.sentinel2_cog_stacked.path   # pathlib.Path (downloads if needed)

Available samples
-----------------

Iterate ``sd`` to see every handle; each carries a description and the required
attribution (``sd.<name>.info()`` / ``sd.<name>.attribution``).

.. list-table::
   :header-rows: 1
   :widths: 26 10 64

   * - Attribute
     - Kind
     - Contents
   * - ``sentinel2_stacked``
     - raster
     - Sentinel-2 L2A blue/green/red/nir as one 4-band file, 1024×1024 @ 10 m,
       EPSG:32633.
   * - ``sentinel2_cog_stacked``
     - raster
     - Cloud-Optimized GeoTIFF variant of the 4-band stack (HTTP range-read).
   * - ``sentinel2_blue`` / ``sentinel2_green`` / ``sentinel2_red`` / ``sentinel2_nir``
     - raster
     - The four Sentinel-2 bands as separate single-band files.
   * - ``copernicus_dem``
     - raster
     - Copernicus GLO-30 DEM warped onto the same grid (float32 metres).
   * - ``copernicus_dem_cog``
     - raster
     - Cloud-Optimized GeoTIFF variant of the DEM.
   * - ``boundary``
     - vector
     - Region-of-interest polygon (GeoPackage, EPSG:4326) inside the footprint.

All rasters share one 1024×1024 grid (the DEM is warped to match), so the
imagery, elevation, and boundary overlay pixel-for-pixel.

Where files are cached
----------------------

Files are cached under ``~/.cache/easy-eo`` by default. The location is
resolved as:

1. ``$EEO_DATA_DIR`` if set,
2. ``$XDG_CACHE_HOME/easy-eo`` if ``XDG_CACHE_HOME`` is set,
3. ``~/.cache/easy-eo`` otherwise.

A cached file whose checksum still matches is reused untouched; a missing or
corrupted file is transparently re-downloaded. :func:`eeo.datasets.cache_dir`
returns the resolved directory.

Licensing and attribution
--------------------------

The sample is derived from open Copernicus data. If you redistribute figures or
data made from it, carry the attribution — ``sd.<name>.info()`` and
``sd.<name>.attribution`` print the exact text:

- **Sentinel-2:** *Contains modified Copernicus Sentinel-2 L2A data 2023 (tile
  T33UUP, acquired 2023-09-07), processed by ESA; accessed via Microsoft
  Planetary Computer.*
- **Copernicus DEM GLO-30:** *© DLR e.V. 2010–2014 and © Airbus Defence and
  Space GmbH 2014–2018, provided under COPERNICUS by the European Union and
  ESA; accessed via Microsoft Planetary Computer.*
