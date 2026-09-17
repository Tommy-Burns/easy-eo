Backends & Adapters
===================

Easy-EO uses an **adapter-based backend architecture** to decouple raster
processing logic from data storage formats.

This design allows the same high-level API to operate on:

- Raster files on disk (Rasterio-backed)
- In-memory NumPy arrays
- Raster files opened lazily as dask-chunked xarray arrays

The backend is transparent by default (meaning users normally do not need to know
or care whether the data is backed by Rasterio or NumPy, as all public methods
behave the same), but advanced users can access or convert it explicitly when needed.

.. note::

   The xarray *backend* — a dataset whose pixels stay lazy and chunked — is a
   different thing from xarray *interop*. :doc:`user_guide/xarray_interop`
   converts a dataset to a :class:`xarray.DataArray` and back at the boundary;
   both sides are read into memory, and the backend is unchanged.

------

Conceptual Overview
-------------------

At the core of Easy-EO is the :class:`EEORasterDataset`, which delegates
all I/O and metadata access to an internal **adapter**.

::

   EEORasterDataset
          |
          v
    BaseRasterAdapter (abstract)
       /       |       \
      v        v        v
 Rasterio    NumPy    Xarray
  Adapter   Adapter   Adapter

Each adapter exposes a **uniform interface** for:

- Metadata access (CRS, transform, bounds)
- Reading raster values
- Writing or persisting data
- Accessing the underlying backend object

-----

Available Adapters
------------------

RasterioAdapter
^^^^^^^^^^^^^^^

The ``RasterioAdapter`` wraps a ``rasterio.DatasetReader`` and provides
full support for spatial operations.

It may used when:

- Loading rasters from disk
- Performing spatial resampling
- Writing georeferenced outputs

This adapter supports:

- CRS-aware operations
- Spatial transforms
- RasterIO resampling and reprojection

NumPyRasterioAdapter
^^^^^^^^^^^^^^^^^^^^

The ``NumPyRasterioAdapter`` wraps an in-memory NumPy array together with
explicit spatial metadata.

It may be used when:

- Creating datasets from arrays
- Performing numerical or analytical operations
- Prototyping without disk I/O

This adapter supports:

- Fast array-based operations
- Explicit CRS and transform handling
- Seamless promotion to Rasterio when required

XarrayAdapter
^^^^^^^^^^^^^

The ``XarrayAdapter`` wraps a :class:`xarray.DataArray` opened by
``rioxarray.open_rasterio`` with dask chunks. Opening a file reads only its
metadata; pixels are computed when something reads them, and then only the
bands and window asked for. It needs the ``lazy`` extra
(``pip install "easy-eo[lazy]"``) and is selected by passing ``chunks`` to the
loader — nothing else in your code changes:

.. code-block:: python

   ds = eeo.load_raster("scene.tif", chunks="auto")        # dask decides
   ds = eeo.load_raster("scene.tif", chunks={"y": 2048, "x": 2048})

``"auto"`` lets dask pick chunk sizes aligned with the file's internal blocks,
capped at dask's configured chunk size (128 MiB by default) — so a single
Landsat or Sentinel-2 band usually fits in one chunk. Pass explicit sizes to cut
it finer.

This adapter supports:

- The same metadata as the rasterio backend: CRS, transform, bounds, nodata,
  dtype, driver and band names
- Reading bands and windows (``read(indexes, window=...)``), computing only
  the chunks the selection touches
- Saving one chunk at a time, so writing a scene to disk never holds the whole
  of it in memory

Operations work on a lazy dataset exactly as they do on any other, and they do
not read it to do so: promoting a lazy dataset to the rasterio backend reopens
its file rather than reading its pixels, after which the block-wise engine
streams from the file a window at a time. So ``ds.to_rasterio()``, which every
operation calls, is free for a lazy dataset opened from a file.

Reading over HTTP
^^^^^^^^^^^^^^^^^

The loader takes a URL wherever it takes a path, so a cloud-optimized GeoTIFF
can be read where it sits:

.. code-block:: python

   url = "https://example.com/scenes/B04.tif"
   ds = eeo.load_raster(url, chunks=1024)
   patch = ds.read(1, window=Window(2048, 2048, 512, 512))

Nothing is downloaded whole. GDAL fetches byte ranges, so opening the raster
costs its header and reading a window costs the tiles that window covers. The
same works without ``chunks`` on the rasterio backend, and for ``s3://``,
``gs://`` and ``az://`` paths, as well as for a file inside a local archive
(``/vsizip/products.zip/B04.tif``).

.. note::

   A remote read is only as selective as the file allows. A *cloud-optimized*
   GeoTIFF is internally tiled, so a window touches a few tiles; a plain
   striped GeoTIFF makes GDAL fetch whole rows, and a JPEG 2000 may fetch far
   more than you asked for.

.. note::

   This means an operation on a lazy dataset does not build a dask graph and
   does not return a lazy result — it returns an ordinary rasterio-backed
   raster, computed there and then. What the lazy backend gives you today is
   an unread, chunked view of a file, and reads bounded to the bands and
   windows you ask for.

-----

Explicit Backend Conversion
---------------------------

Advanced users may explicitly convert between backends.

Convert to Rasterio
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   ds_rio = ds.to_rasterio()

This creates an in-memory Rasterio-backed dataset and returns a new
``EEORasterDataset``.

Convert to NumPy
^^^^^^^^^^^^^^^^

.. code-block:: python

   array = ds.to_array()

This returns the raster data as a NumPy array with shape:

- ``(bands, height, width)`` for multiband rasters
- ``(height, width)`` for single-band rasters

-------


Design Philosophy
-----------------

This adapter-based design provides:

- Separation of concerns
- Backend extensibility
- Performance-aware operations
- A clean, stable public API

Most users will never need to think about backends — but when they do,
the system remains explicit and predictable.
