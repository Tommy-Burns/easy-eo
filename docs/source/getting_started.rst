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

Installing Easy-EO automatically installs these dependencies within the tested
version ranges:

- ``rasterio>=1.4,<2``
- ``geopandas>=1.1,<2``
- ``numpy>=1.26,<3``
- ``matplotlib>=3.8``

Install from either package manager:

.. code-block:: bash

    pip install easy-eo

.. code-block:: bash

    conda install -c conda-forge easy-eo

Whichever you choose, stay with it — see :ref:`extras-package-manager` below.

----

Optional extras
^^^^^^^^^^^^^^^

Heavier, feature-specific dependencies live behind optional extras so the base
install stays small. Install an extra by name:

.. code-block:: bash

    pip install "easy-eo[stac]"

Several extras can be installed at once: ``pip install "easy-eo[stac,xarray]"``.

.. list-table::
    :header-rows: 1
    :widths: 12 30 58

    * - Extra
      - Installs
      - Enables
    * - ``stac``
      - ``pystac-client``, ``planetary-computer``
      - Searching STAC catalogs and loading assets straight into an
        :class:`~eeo.core.EEORasterDataset` — see
        :doc:`user_guide/loading_satellite_data`
    * - ``xarray``
      - ``xarray``, ``rioxarray``
      - Converting between an :class:`~eeo.core.EEORasterDataset` and a
        georeferenced :class:`xarray.DataArray`, to hand data to the wider
        xarray ecosystem and back — see :doc:`user_guide/xarray_interop`

.. _extras-package-manager:

Extras with conda
-----------------

conda has no concept of extras. ``[project.optional-dependencies]`` becomes
wheel metadata that pip understands, whereas a conda package carries one flat
dependency list with nothing to opt into — and brackets already mean something
else in conda's own grammar (key-value constraints such as
``easy-eo[channel=conda-forge]``), so ``conda install "easy-eo[stac]"`` does not
merely fail to find the extra, it fails to parse.

Install the same packages by name instead:

.. list-table::
    :header-rows: 1
    :widths: 12 44 44

    * - Extra
      - pip
      - conda
    * - ``stac``
      - ``pip install "easy-eo[stac]"``
      - ``conda install -c conda-forge easy-eo pystac-client planetary-computer``
    * - ``xarray``
      - ``pip install "easy-eo[xarray]"``
      - ``conda install -c conda-forge easy-eo xarray rioxarray``

.. warning::

    Install Easy-EO and its extras with the **same** package manager. A conda
    package's files are tracked by conda's solver; pip-installed files are not.
    So pip-installing an extra into a conda-managed environment works at first
    and breaks later: a subsequent ``conda install`` or ``conda update`` can
    overwrite those files, or resolve a second copy of a transitive dependency
    that conda already manages (``pystac-client`` alone brings ``pystac``,
    ``requests`` and ``python-dateutil``). Every extra dependency listed above
    is on conda-forge, so mixing is never necessary.

Using a feature without its extra installed raises
:class:`~eeo.MissingDependencyError` — an ``ImportError`` whose message names
the exact install command for the package manager Easy-EO was installed with.
On a pip install:

.. code-block:: text

    MissingDependencyError: STAC search requires the optional 'pystac_client'
    package, which is not installed. Install the 'stac' extra with:
    pip install 'easy-eo[stac]'

and on a conda install, where that bracket syntax could not be run at all:

.. code-block:: text

    MissingDependencyError: STAC search requires the optional 'pystac_client'
    package, which is not installed. Easy-EO was installed by conda, which has
    no equivalent of pip's extras, so install the 'stac' extra's packages by
    name: conda install -c conda-forge pystac-client planetary-computer. Do not
    pip install them into a conda-managed environment: conda does not track
    pip-installed files, so a later conda install or update can overwrite them.

Which one you see is decided by conda's own record of the ``easy-eo`` package,
not by the kind of environment: Easy-EO pip-installed into a conda environment
still gets the pip command, which is the correct advice for it.

``eeo.show_versions()`` reports which extras are present, which is worth
pasting into any bug report.

----

Installing into a fresh environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Easy-EO and its geospatial dependencies are worth isolating from other
projects. With conda, create the environment and install from conda-forge:

.. code-block:: bash

    conda create -n eeo python=3.11
    conda activate eeo
    conda install -c conda-forge easy-eo

With pip, a virtual environment does the same job:

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate     # Windows: .venv\Scripts\activate
    pip install easy-eo

Either way the required dependencies come with it. Make sure the environment's
Python is **3.10 or above**.

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

    # The same thing as a bound method
    ndvi = ds_nir.normalized_difference(ds_red)

.. note::
    Returns a new **EEORasterDataset**, so it chains with any other operation.
    Call ``.to_array()`` on the result when you need the raw **NumPy array**.

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

    # Plot a single band (2-98 percentile stretch is on by default)
    ds.plot_raster(bands=1, cmap="gray")

    # Plot histogram for multiple bands
    ds.plot_histogram(bands=[1,2,3], bins=256)

    # Plot raster and its histogram (histogram shows raw values; opt in to stretch)
    ds.plot_raster_with_histogram(bands=[1,2], stretch=True)

    # Plot composite (e.g., RGB) — stretched by default
    ds.plot_composite(bands=(4,3,2))

.. note::
    - Percentile stretching is on by default for ``plot_raster``,
      ``plot_band_array`` and ``plot_composite`` (pass ``stretch=False`` for raw
      values); ``plot_raster_with_histogram`` defaults to raw, and its histogram
      bins true values whether or not the raster panel is stretched
    - Every histogram gets its own y-axis, so a quiet band is not flattened by
      a busy one
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
    (
        ds.clip_raster_with_bbox((0, 0, 1000, 1000))
        .normalize_min_max()
        .plot_raster()
    )

.. note::
    - Terminal operations like `plot_raster` **do not return EEORasterDataset**
    - Operator overloading provides concise arithmetic (`+`, `-`, `*`, `/`)
