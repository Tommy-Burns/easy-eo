.. Easy-EO documentation master file, created by
   sphinx-quickstart on Tue Dec 16 23:28:11 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Easy-EO documentation
=====================
Easy-EO is a Python package for **chainable raster processing, algebra, and visualization**.
It provides high-level abstractions over libraries such as `Rasterio <https://rasterio.readthedocs.io/en/stable/>`_,
`NumPy <https://numpy.org/>`_, and `Matplotlib <https://matplotlib.org/>`_, enabling users to perform common earth-observation analyses and
visualization tasks efficiently, without dealing with the underlying complexity.
It supports all ``rasterio`` supported datasets since it is built on ``rasterio``.

Why Easy-EO?
------------

Working directly with libraries like Rasterio can be powerful, but they often require verbose
boilerplate code for simple operations such as raster reprojection, resampling, arithmetic
between rasters, clipping, mosaicking, or plotting. Easy-EO abstracts these routines into
**high-level, chainable methods**, allowing users to:

   - Perform multiple operations in a single, readable chain.
   - Persist intermediate results **in memory** without writing to disk unnecessarily.
   - Automatically align rasters with differing shapes or coordinate reference systems.
   - Return a consistent **EEORasterDataset** object from each operation, enabling further chaining.
   - Use **terminal visualization methods** for plotting bands, composites, and histograms,
     which do not return EEORasterDataset but instead display results.

Chainable Workflow
------------------

All methods in Easy-EO are designed to be chainable, except for visualization operations
which are terminal. For example:

.. code-block:: python

    from eeo import load_raster

    ds_nir = load_raster("path/to/nir.tif")
    ds_red = load_raster("path/to/red.tif")

    # Chainable example: clip -> resample -> compute NDVI -> multiply
    result = ds_nir.clip_raster_with_bbox((0,0,1000,1000))
                   .resample(scale_factor=2)
                   .normalized_difference(ds_red)
                   .multiply(100)

Visualization is always done at the end of the chain:

.. code-block:: python

    # Terminal operation: display raster and histogram
    result.plot_raster_with_histogram(bands=[1,2], stretch=True)

Key Features
------------

- **Raster algebra:** Supports pixel-wise addition, subtraction, multiplication, division, and power operations. Operator overloading allows `+`, `-`, `*`, `/`, and `**` for concise syntax.
- **Raster indices:** Compute normalized difference indices (e.g., NDVI) or custom indices, returning either NumPy arrays or EEORasterDataset for further chaining.
- **Spatial operations:** Clip rasters using bounding boxes or vector geometries, mosaic multiple rasters, or stack rasters as new bands.
- **Standardization & normalization:** Apply z-score, min-max, or percentile-based normalization.
- **Visualization:** Plot individual bands, composites, histograms, or raster with histogram. Supports multi-band rasters and percentile-based contrast stretching.

Getting Started
---------------

See :doc:`getting_started` for step-by-step instructions on loading rasters, performing
arithmetic, computing indices, and visualizing results.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   user_guide/core_dataset
   user_guide/ops
   user_guide/preprocessing
   user_guide/visualization
   user_guide/statistical_locations
   backends

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   modules/core
   modules/adapters
   modules/analysis
   modules/ops
   modules/preprocessing
   modules/viz

