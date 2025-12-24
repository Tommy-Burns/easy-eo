Getting Started
===============

Easy-EO is a Python package for chainable raster processing, algebra, and visualization.
It provides high-level abstractions over Rasterio, NumPy, and Matplotlib, enabling users
to perform common earth-observation analysis and visualization tasks efficiently.

This guide shows how to install `easy-eo` load rasters, perform operations, compute indices, and visualize results.

Installation
============

Easy-EO requires **Python 3.10 or higher** and relies on several core geospatial and
scientific libraries, including:

- **Rasterio** for raster I/O and geospatial transformations.
- **GeoPandas** for vector data support.
- **NumPy** for numerical and array operations.
- **Matplotlib** for visualization of rasters and histograms.

Installing Easy-EO (this current version) automatically installs these dependencies with compatible versions:

- `rasterio>=1.4,<1.5`
- `geopandas>=1.1,<1.2`
- `numpy>=1.26,<2.2`
- `matplotlib>=3.10,<3.11`

----

It is recommended to install `easy-eo` in a `conda` environment:

.. code-block:: bash

    conda create -n env_name python=3.10
    conda activate env_name
    pip install easy-eo

This will install **Easy-EO** along with all required dependencies. Make sure your
Python environment is version **3.10 or above**.

----

Verify installation by importing the package and checking versions:

.. code-block:: python

    import eeo
    import rasterio
    import geopandas as gpd
    import numpy as np
    import matplotlib

    print("Easy-EO version:", eeo.__version__)
    print("Rasterio version:", rasterio.__version__)
    print("GeoPandas version:", gpd.__version__)
    print("NumPy version:", np.__version__)
    print("Matplotlib version:", matplotlib.__version__)
    print("Easy-EO installed!")


Core Concepts
=============

Loading a raster
^^^^^^^^^^^^^^^^

.. code-block:: python

    from eeo import load_raster

    # Load a raster from disk
    ds = load_raster("path/to/image.tif")

This function performs validation and returns an :class:`~eeo.core.EEORasterDataset`.
Advanced users may access the underlying rasterio dataset via `ds.ds`

**Properties of the raster can be inspected as:**

.. code-block:: python

    print(ds.get_crs())       # Coordinate reference system
    print(ds.get_shape())     # (height, width)
    print(ds.get_transform()) # Affine transform

Accessing bands
^^^^^^^^^^^^^^^

.. code-block:: python

    # Read a single band
    band1 = ds.get_band(1)

    # Read multiple bands
    bands = [ds.get_band(i) for i in range(1, ds.get_count() + 1)]


Algebra and Arithmetic
----------------------

Easy-EO supports **pixel-wise operations** with chainable syntax.

.. code-block:: python

    from eeo.ops.algebra import add, subtract, multiply, divide

    ds2 = load_raster("path/to/other.tif")

    # Add two rasters (auto-aligns if necessary)
    result = ds + ds2

    # Multiply raster by scalar
    result2 = ds * 2

    # Chain operations
    result3 = (ds - ds2).divide(100)

.. note::
    Supports operator overloading: ``+``, ``-``, ``*``, ``/``, ``**``.
    `auto_align=True` ensures datasets with different shapes can be processed safely.

Indices
-------

Easy-EO allows computing **normalized or custom indices**:

.. code-block:: python

    from eeo.analysis import normalized_difference

    # NDVI-like computation
    ndvi = normalized_difference(ds_nir, ds_red)

    # Return as EEORasterDataset
    ndvi_ds = ds_nir.normalized_difference(ds_red, return_as_ndarray=False)

.. note::
    Supports chaining with other operations and can return either **NumPy arrays** or **EEORasterDataset**.

Clipping, Mosaicking, and Stacking
----------------------------------

.. code-block:: python

    # Clip raster to bounding box
    clipped = ds.clip_raster_with_bbox((0, 0, 1000, 1000))

    # Clip using vector (using a geopandas GeodataFrame)
    import geopandas as gpd
    shapefile = gpd.read_file("vector.shp")
    clipped2 = ds.clip_raster_with_vector(shapefile, crop=True)


    # Clip using vector (using the path to a geopandas supported vector file)
    shapefile_path = r"/path/to/vector_file"
    clipped3 = ds.clip_raster_with_vector(shapefile_path, crop=True)

    # Mosaic multiple rasters
    mosaic_ds = ds.mosaic([ds2, ds3], auto_reproject=True)

    # Stack multiple rasters as bands
    stacked = ds.stack([ds2, ds3])

.. note::
    - Auto-reprojects if CRS mismatch
    - Returns a multi-band raster
    - Supports single or multiple rasters as input

Visualization
-------------

Visualization functions are **terminal operations** and should be used last in a chain.

.. code-block:: python

    from eeo.viz import (
        plot_raster,
        plot_histogram,
        plot_composite,
        plot_raster_with_histogram
    )

    # Plot a single band
    ds.plot_raster(bands=1, cmap="gray", stretch=True)

    # Plot histogram for multiple bands
    ds.plot_histogram(bands=[1,2,3], bins=256, sharey=True)

    # Plot raster and its histogram
    ds.plot_raster_with_histogram(bands=[1,2], stretch=True, sharey=True)

    # Plot composite (e.g., RGB)
    ds.plot_composite(bands=(4,3,2), stretch=True)

.. note::
    - Percentile stretching (`stretch=True`) improves contrast
    - `sharey=True` aligns histogram axes across multiple bands
    - Multi-band plotting works with single or multiple rasters
    - Composite plotting supports RGB/false-color conventions

Normalization and Standardization
---------------------------------

.. code-block:: python

    # Z-score standardization
    standardized = ds.standardize()

    # Min-max normalization
    normalized = ds.normalize_min_max(new_min=0, new_max=1)

    # Percentile-based normalization
    percentile_norm = ds.normalize_percentile(lower_percentile=2, upper_percentile=98)

.. note::
    Useful before visualization or analysis. Can chain with other operations.

Saving and Persistence
----------------------

To save a dataset to disk, use:

``save_raster(path, driver="GTiff")``

Example:

.. code-block:: python

    ds.normalize_min_max().save_raster("output.tif")

Until this method is called, datasets typically live in memory, allowing
fast experimentation without unnecessary disk I/O.

-----

Resource Management
-------------------

Because ``EEORasterDataset`` wraps a Rasterio dataset, it holds file
handles and GDAL resources.

To explicitly release resources, call: ``close()``

Example:

.. code-block:: python

    ds.close()

Notes:

    - Datasets created from in-memory files (e.g. clipping, mosaicking)
      become unusable after closing
    - A ``__del__`` method exists as a safety fallback, but explicit
      ``close()`` calls are strongly recommended

-----

Tips
----

.. code-block:: python

    # Use chainable operations for clarity
    ds.clip_raster_with_bbox((0,0,1000,1000))
       .normalize_min_max()
       .plot_raster()

.. note::
    - Terminal operations like `plot_raster` **do not return EEORasterDataset**
    - Operator overloading provides concise arithmetic (`+`, `-`, `*`, `/`)
