Preprocessing Functions
=======================

Preprocessing operations prepare raster datasets for analysis, modeling,
or visualization. These operations modify raster geometry, values, or spatial
reference while preserving metadata and supporting Easy-EO’s chainable workflow.

All preprocessing functions operate on
:class:`~eeo.core.core.EEORasterDataset` objects and return new datasets unless
explicitly instructed to save results to disk.

-----

Clipping and Masking
--------------------

Clipping restricts raster data spatially using either vector geometries or
explicit bounding boxes. Easy-EO supports both *cropping* and *masking* workflows.

Vector-based Clipping
^^^^^^^^^^^^^^^^^^^^^

.. function:: clip_raster_with_vector(ds, vector_file, *, crop=True, pad=False, all_touched=False, invert=False, nodata=None, show_preview=False, plot_kwargs=None)

   Clip or mask a raster using vector geometries.

   This method can either:
       - **Crop** the raster to the geometry bounds, or
       - **Mask** pixels outside (or inside) the geometries while preserving the
         original raster extent.

   **Parameters**

   - **ds** (:class:`EEORasterDataset`)
     Input raster dataset.
   - **vector_file** (GeoDataFrame or str)
     Vector geometries used for clipping. If a string is provided, it must be a
     valid path readable by GeoPandas.
   - **crop** (bool, default=True)
     If ``True``, the output raster is cropped to the minimal bounding box of
     the geometries.
     If ``False``, the raster extent is preserved and pixels outside the
     geometries are set to nodata.
   - **pad** (bool, default=False)
     If ``crop=True``, expands the output extent to fully include edge pixels.
   - **all_touched** (bool, default=False)
     If ``True``, all pixels touched by geometries are included.
     If ``False``, only pixels whose center lies within geometries are included.
   - **invert** (bool, default=False)
     If ``True``, masks pixels *inside* the geometries instead of outside.
   - **nodata** (int or float, optional)
     Value assigned to masked pixels. If ``None``, uses the dataset’s nodata value.
   - **show_preview** (bool, default=False)
     If ``True``, displays a preview of the result.
   - **plot_kwargs** (dict, optional)
     Additional keyword arguments passed to ``rasterio.plot.show``.

   **Returns**

   - ``EEORasterDataset``

   **Example**

   .. code-block:: python

      clipped = ds.clip_raster_with_vector(
          "boundary.shp",
          crop=True,
          all_touched=True
      )

-----

Bounding Box Clipping
^^^^^^^^^^^^^^^^^^^^^

.. function:: clip_raster_with_bbox(ds, bbox, plot_kwargs=None, show_preview=False)

   Clip a raster using a bounding box.

   This method subsets the raster to the provided bounding box. The bounding
   box must be defined in the same CRS as the raster.

   **Parameters**

   - **ds** (:class:`EEORasterDataset`)
     Input raster dataset.
   - **bbox** (tuple or list)
     Bounding box coordinates as ``(minx, miny, maxx, maxy)``.
   - **show_preview** (bool, default=False)
     Display a preview of the clipped raster.
   - **plot_kwargs** (dict, optional)
     Additional keyword arguments passed to ``rasterio.plot.show``.

   **Returns**

   - ``EEORasterDataset``

   **Example**

   .. code-block:: python

      clipped = ds.clip_raster_with_bbox((500000, 4100000, 510000, 4110000))

-----

Value Normalization and Standardization
---------------------------------------

These operations modify raster **pixel values**, not spatial geometry.

.. note::

   Although percentile normalization is often used for visualization, these
   methods produce real numeric transformations suitable for analysis and
   machine learning workflows.

Z-score Standardization
^^^^^^^^^^^^^^^^^^^^^^^

.. function:: standardize(ds)

   Apply Z-score standardization to raster values:

   ``(x - mean) / standard_deviation``

   This transformation centers the data around zero and scales it to unit
   variance.

   **Returns**

   - ``EEORasterDataset``

   **Use cases**

   - Machine learning
   - Statistical analysis
   - Feature normalization

-----

Min–Max Normalization
^^^^^^^^^^^^^^^^^^^^^

.. function:: normalize_min_max(ds, *, new_min=0, new_max=1)

   Linearly rescale raster values to a new range.

   **Parameters**

   - **new_min** (int or float)
     Lower bound of the target range.
   - **new_max** (int or float)
     Upper bound of the target range.

   **Returns**

   - ``EEORasterDataset``

-----

Percentile Normalization
^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: normalize_percentile(ds, *, lower_percentile=2, upper_percentile=98)

   Normalize raster values using percentile thresholds.

   Values outside the percentile range are clipped, and the remaining values
   are scaled to the interval ``[0, 1]``.

   This method is robust to outliers and commonly used for skewed data.

   **Parameters**

   - **lower_percentile** (float)
     Lower percentile threshold (0–100).
   - **upper_percentile** (float)
     Upper percentile threshold (0–100).

   **Returns**

   - ``EEORasterDataset``

-----

Spatial Reference Operations
----------------------------

Reprojection
^^^^^^^^^^^^

.. function:: reproject_raster(ds, *, target_crs, resampling_method="nearest")

   Reproject a raster to a new coordinate reference system (CRS).

   **Parameters**

   - **ds** (:class:`EEORasterDataset`)
     Input raster.
   - **target_crs** (int | str | pyproj.CRS)
     Target CRS (EPSG code, PROJ string, or CRS object).
   - **resampling_method**
     Resampling strategy used during reprojection.

   **Returns**

   - ``EEORasterDataset``

-----

Resampling
^^^^^^^^^^

.. function:: resample(ds, *, size=None, scale_factor=None, resolution=None, resampling_method="nearest", plot_kwargs=None, show_preview=False)

   Resample a raster to a new resolution or spatial shape.

   Only **one** of ``size``, ``scale_factor``, or ``resolution`` must be provided.

   **Backend handling**

   Resampling is a spatial operation that requires a Rasterio backend.
   If the input dataset is backed by a NumPy array, it is **automatically
   promoted** to an in-memory Rasterio dataset before resampling.

   This promotion is transparent to the user and preserves spatial
   metadata such as CRS, transform, data type, and nodata values.

   **Parameters**

   - **size** (tuple[int, int], optional)
     Output raster dimensions as ``(height, width)``.

   - **scale_factor** (float, optional)
     Uniform scaling factor applied to both spatial dimensions.

   - **resolution** (tuple[float, float], optional)
     Target spatial resolution in CRS units as ``(xres, yres)``.

   - **resampling_method** (str)
     Resampling algorithm (e.g. ``nearest``, ``bilinear``, ``cubic``).

   - **show_preview** (bool)
     If ``True``, displays a visual preview of the resampled raster.

   **Returns**

   - ``EEORasterDataset``
     A new dataset with updated resolution and transform.


   **Notes**

   - NumPy-backed datasets are promoted internally using an in-memory
     Rasterio dataset.
   - Advanced users can explicitly control backend conversion using
     :meth:`EEORasterDataset.to_rasterio` and :meth:`EEORasterDataset.to_array`.

   **Example**

   .. code-block:: python

      ds = load_array(array, transform=transform, crs=4326)

      # Resampling works transparently
      ds_resampled = ds.resample(scale_factor=2.0)

   **Processing flow**

   ::

      NumPy-backed dataset
              ↓ (automatic promotion)
      Rasterio-backed dataset
              ↓ resample()
      New EEORasterDataset

-----

Chaining Example
----------------

All preprocessing operations are chainable and remain in memory until explicitly saved.

.. code-block:: python

   output = ds.clip_raster_with_vector("boundary.shp")
              .reproject_raster(target_crs=4326)
              .save_raster("processed.tif")

