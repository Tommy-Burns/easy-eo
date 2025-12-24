Statistical Pixel Location Operations
=====================================
Easy-EO provides a set of nodata-safe statistical operations that not only
compute summary statistics but also return the *spatial location* of those
values within a raster.

These functions may be useful for exploratory geospatial analysis

All functions operate on raster bands and return both the statistic
and its spatial or pixel location.

----

General Behavior
----------------
All statistical location functions:
    - Mask nodata values automatically
    - Operate on a single band
    - Are chainable via ``EEORasterDataset``
    - Return structured dictionaries

Returned structure:

.. code-block:: python

    {
        "value": float,
        "position": (row, col)  # or (x, y)
    }

-------------------------------------

Pixel Coordinates vs Spatial Coordinates
----------------------------------------
By default, positions are returned in **spatial coordinates** (CRS-aware).
You may instead request **pixel coordinates**:

.. code-block:: python

    max_pixel = ds.get_maximum_pixel(return_position_as_pixel_coordinate=True)

-------------------------------------


Available Functions
-------------------

Maximum Pixel
~~~~~~~~~~~~~

Returns the maximum value and its location.

.. code-block:: python

    result = ds.get_maximum_pixel()
    result["value"]
    result["position"]

-------------------------------------

Minimum Pixel
~~~~~~~~~~~~~

Returns the minimum value and its location.

.. code-block:: python

    ds.get_minimum_pixel()

-------------------------------------

Mean Pixel
~~~~~~~~~~

Returns the pixel whose value is closest to the mean.

.. note::

   The mean itself may not exist as an exact pixel value.
   Easy-EO returns the nearest pixel to the mean by absolute difference.

-------------------------------------

Percentile Pixel
~~~~~~~~~~~~~~~~

Returns the pixel corresponding to a given percentile.

.. code-block:: python

    ds.get_percentile_pixel(95)

-------------------------------------

Chaining Example
----------------

.. code-block:: python

       data = ds.clip_raster_with_bbox(bbox)
                .normalize_percentile()
                .get_maximum_pixel()

-------------------------------------

Performance Notes
-----------------

- All computations are NumPy-based
