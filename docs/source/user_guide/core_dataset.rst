The EEORasterDataset Abstraction
================================

At the heart of Easy-EO is the :class:`eeo.core.EEORasterDataset` class.
It represents a raster dataset as a **chainable, in-memory object**
that provides a high-level interface for geospatial raster processing,
analysis, and visualization.

Motivation
----------

Geospatial libraries such as Rasterio expose powerful but **low-level**
interfaces. Common workflows often require:

- Explicit file handling
- Manual reprojection and alignment
- Careful tracking of metadata
- Repeated reads and writes to disk

Easy-EO introduces a dataset-centric abstraction that:

- Keeps raster data in memory by default
- Returns new datasets from processing operations
- Enables expressive, readable, chainable workflows
- Defers disk I/O until explicitly requested

-----

EEORasterDataset
----------------

The :class:`~eeo.core.EEORasterDataset` class is the fundamental object
used throughout Easy-EO. All raster operations operate on or return
instances of this class.

Internally, an ``EEORasterDataset`` wraps a raster backend (typically
Rasterio), but exposes only a curated, stable, high-level API.

-----

Creating a Dataset
------------------

Datasets are typically created using the :func:`load_raster` helper:

.. code-block:: python

    from eeo import load_raster

    ds = load_raster("image.tif")

Datasets may also be created from NumPy arrays:

.. code-block:: python

    from eeo import load_array
    import numpy as np

    array = np.random.rand(512, 512)
    ds = load_array(array, crs=4326)


.. note::
    Some operations (e.g., resampling, reprojection) require a Rasterio backend.
    These operations must be performed on Rasterio-supported files loaded with
    :func:`load_raster`, not on arrays loaded with :func:`load_array`.

-----

Dataset Metadata
----------------

Common raster metadata is exposed through explicit accessor methods:

- ``get_crs()`` – Coordinate reference system
- ``get_transform()`` – Affine transform
- ``get_shape()`` – Raster shape (height, width)
- ``get_bounds()`` – Spatial bounding box
- ``get_width()`` – Raster width
- ``get_height()`` – Raster height
- ``get_count()`` – Number of bands
- ``get_metadata()`` – Raster metadata dictionary

These methods intentionally mirror Rasterio concepts while keeping the
public API stable and explicit.

-----

Reading Raster Data
-------------------

Raster values can be accessed as NumPy arrays using:

``read()``
    Read the full raster. Intended for single-band datasets.

``get_band(idx)``
    Read a single band (1-based indexing).

Example:

.. code-block:: python

    band1 = ds.get_band(1)

For multi-band datasets, ``get_band()`` is preferred to avoid loading
unnecessary data into memory.

-----

Chainable Operations
--------------------

Most processing functions in Easy-EO are **chainable**.

Operations such as clipping, algebra, normalization, and resampling
return new ``EEORasterDataset`` instances, enabling fluent workflows:

.. code-block:: python

    result = ds.clip_raster_with_bbox((0, 0, 1000, 1000))
               .normalize_percentile(lower_percentile=2, upper_percentile=98)
               .standardize()

Internally, these operations are implemented as standalone functions and
bound dynamically to ``EEORasterDataset`` using decorators.

-----

Terminal Operations
-------------------

Some operations are **terminal**, meaning they do not return a dataset.

These include:

- Visualization functions
- Explicit saving to disk

Terminal operations are intended to appear at the end of a chain:

.. code-block:: python

    ds.normalize_min_max().plot_raster()

-----

Saving and Persistence
----------------------

To persist a dataset to disk, use:

``save_raster(path, driver="GTiff")``

Example:

.. code-block:: python

    ds.normalize_min_max().save_raster("output.tif")

Until this method is called, datasets typically remain in memory.

-----

Resource Management
-------------------

Because ``EEORasterDataset`` may wrap open file handles or in-memory
datasets, resources should be released explicitly when no longer needed:

.. code-block:: python

    ds.close()

A ``__del__`` fallback exists,
but explicit cleanup is recommended in long-running workflows.

-----

Accessing the Underlying Data
-----------------------------

Easy-EO does not restrict access to underlying raster representations.

Advanced users may:

- Retrieve underlying dataset as NumPy arrays using ``.read()`` or ``.get_band()``
- Retrieve underlying dataset as Rasterio datasets using ``.ds``

This allows advanced users perform specific analyses which are not yet implemented in `easy-eo`.

.. warning::
   Directly modifying the underlying dataset may invalidate assumptions
   made by Easy-EO. If metadata or geometry must be changed, prefer using
   provided preprocessing operations such as reprojection or resampling.

-----

Summary
-------

- ``EEORasterDataset`` is the central abstraction in Easy-EO
- Processing functions return datasets to enable chaining
- Visualization and saving are terminal operations
- Raster data remains in memory until explicitly persisted
- Low-level access remains available for advanced users

