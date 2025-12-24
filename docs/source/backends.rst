Backends & Adapters
===================

Easy-EO uses an **adapter-based backend architecture** to decouple raster
processing logic from data storage formats.

This design allows the same high-level API to operate on:

- Raster files on disk (Rasterio-backed)
- In-memory NumPy arrays
- Future backends (e.g. xarray, cloud-native rasters)

The backend is transparent by default (meaning users normally do not need to know
or care whether the data is backed by Rasterio or NumPy, as all public methods
behave the same), but advanced users can access or convert it explicitly when needed.

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
        /     \
       v       v
 Rasterio   NumPy
  Adapter   Adapter

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
