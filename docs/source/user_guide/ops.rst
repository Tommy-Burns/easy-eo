Selected Raster Operations
===========================

Easy-EO provides a collection of **high-level raster operations** designed for
pixel-wise analysis, transformation, and compositing of earth-observation data.

All operations in this section operate on :class:`~eeo.core.core.EEORasterDataset`
objects and return new datasets unless otherwise noted. Most functions are
**chainable**, meaning they can be combined into expressive processing pipelines.

.. note::

   With the exception of visualization functions, all operations described here
   return either an ``EEORasterDataset`` or a NumPy array and **do not write to disk**
   unless explicitly requested.

-----

Normalized Difference
---------------------

Normalized difference indices are widely used in remote sensing (e.g. NDVI, NDWI).
This library exposes algebraic primitives instead of predefined indices.
Most vegetation and water indices can be expressed directly using
normalized_difference or raster arithmetic.

.. function:: normalized_difference(ds, other, *, auto_align=True, method="bilinear", return_as_ndarray=False)

   Compute a normalized difference index using the formula:

   ``(ds - other) / (ds + other)``

   This operation is typically used to highlight relative differences between
   two spectral bands.

   **Parameters**

   - **ds** (:class:`EEORasterDataset`)
     First raster (e.g. NIR band).
   - **other** (:class:`EEORasterDataset`)
     Second raster (e.g. Red band).
   - **auto_align** (bool, default=True)
     Automatically resample ``other`` to match the spatial resolution, transform,
     and extent of ``ds`` if needed.
   - **method** (str, default="bilinear")
     Resampling method used during alignment.
   - **return_as_ndarray** (bool, default=False)
     If ``True``, returns a NumPy array instead of an ``EEORasterDataset``.

   **Returns**

   - ``numpy.ndarray`` or ``EEORasterDataset``

   **Example**

   .. code-block:: python

      ndvi = ds_nir.normalized_difference(ds_red)
      ndvi_ds = ds_nir.normalized_difference(ds_red, return_as_ndarray=False) # Return as EEORasterDataset

      # Alternatively
      ndvi = normalized_difference(ds_nir, ds_red)
      ndvi_ds = normalized_difference(ds_nir, ds_red, return_as_ndarray=True) # Return as Numpy ndarray

-----

Pixel Value Extraction
----------------------

.. function:: extract_value_at_coordinate(ds, coordinates, band_idx=1)

   Extract a single pixel value at a given geographic coordinate.

   **Parameters**

   - **ds** (:class:`EEORasterDataset`)
     Raster dataset to sample.
   - **coordinates** (tuple)
     Coordinate pair ``(x, y)`` in the raster's CRS.
   - **band_idx** (int, default=1)
     Band index to sample in multiband rasters.

   **Returns**

   - ``int`` or ``float``

   **Example**

   .. code-block:: python

      value = extract_value_at_coordinate(ds, (500000, 4100000))

-----

Arithmetic Operations
---------------------

Easy-EO supports **pixel-wise arithmetic** between rasters and scalars.
These operations are also exposed via Python operators (``+``, ``-``, ``*``, ``/``, ``**``).

All arithmetic operations:
    - Work per pixel
    - Preserve raster metadata
    - Optionally auto-align rasters before computation

Addition
^^^^^^^^

.. function:: add(ds, other, *, auto_align=True, method="bilinear")

   Pixel-wise addition of two rasters or a raster and a scalar.

   **Example**

   .. code-block:: python

      result = ds + ds2
      result = ds.add(10)

Subtraction
^^^^^^^^^^^

.. function:: subtract(ds, other, *, auto_align=True, method="bilinear")

   Pixel-wise subtraction computed as ``ds - other``.

Multiplication
^^^^^^^^^^^^^^

.. function:: multiply(ds, other, *, auto_align=True, method="bilinear")

   Pixel-wise multiplication of raster values.

Division
^^^^^^^^

.. function:: divide(ds, other, *, auto_align=True, method="bilinear", safe=True)

   Pixel-wise division of raster values.

   If ``safe=True``, division by zero and invalid values are handled gracefully
   by suppressing warnings and replacing invalid results with zeros.

Power
^^^^^

.. function:: power(ds, exponent)

   Raise each pixel value to a scalar exponent.

   **Example**

   .. code-block:: python

      squared = ds ** 2

-----

Mathematical Transformations
----------------------------

These operations apply mathematical transformations independently to each pixel.

Square Root
^^^^^^^^^^^

.. function:: sqrt(ds)

   Compute the square root of raster values.

   Negative values are clipped to zero before computation.

Logarithm
^^^^^^^^^

.. function:: log(ds, base=e)

   Compute the logarithm of raster values.

   Zero and negative values are safely clamped to a small positive constant
   before applying the logarithm.

Absolute Value
^^^^^^^^^^^^^^

.. function:: absolute(ds)

   Compute the absolute value of raster pixels.

-----

Mosaicking
----------

.. function:: mosaic(ds, others, *, resampling_method="nearest", auto_reproject=False, **kwargs)

   Merge multiple rasters into a single mosaic.

   **Parameters**

   - **ds** (:class:`EEORasterDataset`)
     Base raster.
   - **others** (list of :class:`EEORasterDataset`)
     Additional rasters to merge.
   - **resampling_method** (str)
     Resampling strategy used during merging.
   - **auto_reproject** (bool)
     Automatically reproject rasters to match ``ds`` CRS if required.

   **Returns**

   - ``EEORasterDataset``

   **Example**

   .. code-block:: python

      mosaic_ds = ds.mosaic([ds2, ds3], auto_reproject=True)

-----

Stacking
--------

.. function:: stack(ds, others)

   Stack multiple rasters into a **multi-band raster**.

   All rasters must share identical:
       - CRS
       - Transform
       - Shape

   **Example**

   .. code-block:: python

      stacked = ds.stack([ds_red, ds_green, ds_blue])

-----

Chaining Behavior
-----------------

All operations in this section:
    - Return ``EEORasterDataset`` unless explicitly documented otherwise
    - Can be chained together
    - Remain in-memory until explicitly saved

**Example workflow**

.. code-block:: python

   result = ds.clip_raster_with_bbox((0, 0, 1000, 1000))
              .normalized_difference(ds2)
              .normalize_min_max()
              .save_raster("output.tif")


Visualization functions are **terminal operations** and should always appear
at the end of a chain.
