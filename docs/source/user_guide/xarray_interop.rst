Working with the xarray Ecosystem
=================================

Easy-EO covers the everyday raster workflow: open a scene, clip it, compute an
index, plot it, write it out. `xarray <https://docs.xarray.dev/>`_ covers
something different — labelled N-dimensional arrays, parallel computation
through `dask <https://www.dask.org/>`_, and a large ecosystem of packages built
on top of that model. Neither replaces the other, so Easy-EO converts between
the two in both directions:

- :meth:`~eeo.core.core.EEORasterDataset.to_xarray` hands a dataset to xarray as
  a georeferenced :class:`xarray.DataArray`.
- :func:`eeo.from_xarray` takes a DataArray back and returns a dataset you can
  keep chaining.

Both need the optional ``xarray`` extra, which installs xarray itself and
`rioxarray <https://corteva.github.io/rioxarray/>`_ (the package that carries
CRS, geotransform, and nodata on a DataArray through its ``.rio`` accessor):

.. code-block:: bash

   pip install "easy-eo[xarray]"

Without the extra, either call raises :class:`~eeo.MissingDependencyError` with
the install command in the message.

.. seealso::

   :doc:`band_names` for how band names are addressed,
   :doc:`nodata_and_dtype` for what the nodata value means to Easy-EO, and
   :doc:`loading_satellite_data` for getting a scene to convert in the first
   place.

-----

Out and back
------------

The whole feature in six lines — search or load a scene, cross into xarray for
something xarray is good at, and come back:

.. code-block:: python

   import eeo

   ds = eeo.load_raster("scene.tif")            # 4-band scene
   da = ds.to_xarray()                          # georeferenced DataArray

   smoothed = da.rolling(x=5, y=5, center=True).mean()   # xarray's job
   result = eeo.from_xarray(smoothed).ndvi(red="red", nir="nir")   # back to Easy-EO

Nothing needs re-georeferencing on either crossing: the CRS, geotransform, and
nodata value travel with the array.

-----

What travels
------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Easy-EO
     - On the DataArray
     - Notes
   * - values and dtype
     - the array itself
     - Passed through untouched — no rescaling, no casting.
   * - CRS
     - ``da.rio.crs`` (a ``spatial_ref`` coordinate)
     - A dataset with no CRS produces a DataArray with none.
   * - geotransform
     - ``da.rio.transform()``
     - Also readable as the pixel-centre ``y``/``x`` coordinates.
   * - nodata
     - ``da.rio.nodata`` (the ``_FillValue`` attribute)
     - The pixels keep whatever marks them — ``NaN`` or the sentinel.
   * - :attr:`~eeo.core.core.EEORasterDataset.band_names`
     - the ``long_name`` attribute
     - rioxarray's own convention, so ``da.rio.to_raster()`` writes the names
       back as GDAL band descriptions.
   * - :attr:`~eeo.core.core.EEORasterDataset.timestamp`
     - a scalar ``time`` coordinate
     - Stored in UTC without a timezone, because xarray datetimes carry none.
   * - :attr:`~eeo.core.core.EEORasterDataset.attrs`
     - ``da.attrs``
     - Copied both ways, minus the keys above that Easy-EO fills itself
       (``long_name``, ``_FillValue``, ``grid_mapping``).

-----

The DataArray Easy-EO produces
------------------------------

:meth:`~eeo.core.core.EEORasterDataset.to_xarray` lays the array out exactly as
:func:`rioxarray.open_rasterio` would, so anything that already reads rioxarray
output reads this too:

.. code-block:: python

   >>> da = eeo.load_raster("scene.tif").to_xarray()
   >>> da.dims
   ('band', 'y', 'x')
   >>> da.band.values
   array([1, 2, 3, 4])
   >>> da.rio.crs.to_epsg()
   32633

Three details worth knowing:

- **The band dimension is always there**, even for a single-band raster, whose
  DataArray has shape ``(1, height, width)``. A raster has bands; a
  one-band raster has one.
- **Bands are numbered from 1**, matching Easy-EO's 1-based band indexing and
  rioxarray's ``band`` coordinate. Names live in ``long_name``, not in the
  coordinate.
- **A rotated or sheared grid gets 2-D coordinates.** One-dimensional ``y`` and
  ``x`` axes cannot describe a rotated grid, so — again following
  :func:`rioxarray.open_rasterio` — such a raster carries 2-D ``yc``/``xc``
  coordinates instead, and the exact affine stays available from
  ``da.rio.transform()``.

-----

Reading a DataArray in
----------------------

:func:`eeo.from_xarray` is deliberately permissive about layout. It accepts
``(band, y, x)`` in any dimension order and ``(y, x)`` for a single band, and it
collapses spare length-1 dimensions (the ones an ``expand_dims`` leaves behind):

.. code-block:: python

   eeo.from_xarray(da)                          # (band, y, x)
   eeo.from_xarray(da.transpose("y", "x", "band"))
   eeo.from_xarray(da.sel(band=1))              # (y, x) -> one band
   eeo.from_xarray(da.expand_dims("time"))      # length-1 dimension collapsed

The spatial dimensions are whichever ones rioxarray identifies — ``y`` and ``x``
by name, or whatever you declare. Data from another source often names them
differently:

.. code-block:: python

   ds = eeo.from_xarray(da.rio.set_spatial_dims(x_dim="lon", y_dim="lat"))

The coordinates decide the geotransform
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A DataArray carries its geotransform in two places at once: the stored affine,
and the coordinate axes. Ordinary xarray operations move the pixels and update
the coordinates, and the stored affine can be left behind. Easy-EO therefore
takes the **coordinates** as the authority, so a slice lands where its
coordinates say it does:

.. code-block:: python

   window = da.isel(y=slice(2000, 3000), x=slice(1500, 2500))
   ds = eeo.from_xarray(window)     # placed at the window's own origin

The stored affine is used only where the coordinates cannot speak — a rotated
grid, a single-pixel axis, or a DataArray with no coordinates — and whenever it
already agrees with them, so an untouched DataArray round-trips exactly.

One consequence is worth stating plainly: **the pixels are taken as they are
laid out.** If you sorted the ``y`` axis ascending (``da.sortby("y")``), the
result is a south-up raster with a positive north-south term in its transform,
correctly georeferenced but stored bottom row first. Easy-EO does not silently
reorder your pixels. Sort ``y`` descending first if you want the conventional
north-up layout.

What is rejected, and why
^^^^^^^^^^^^^^^^^^^^^^^^^

Each of these raises :class:`~eeo.ValidationError` with the fix in the message,
rather than guessing:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Input
     - Why, and what to do
   * - an :class:`xarray.Dataset`
     - A dataset holds several variables; a raster is one array. Select a
       variable (``dataset["reflectance"]``) or merge them with
       ``dataset.to_dataarray()``.
   * - a ``time`` dimension longer than one step
     - That is a time series, not a band stack, and loading timesteps as bands
       would mislabel the data. Select a step (``da.isel(time=0)``) or reduce it
       (``da.mean("time")``).
   * - two or more band-like dimensions
     - A raster has one. Select or reduce the extras first.
   * - an unevenly spaced coordinate axis
     - No single geotransform can describe it, so it is not a raster grid.
       ``da.isel(x=[0, 2, 5])`` produces one; reindex or interpolate onto a
       regular grid instead.
   * - unidentifiable spatial dimensions
     - Rename them to ``y``/``x``, or declare them with
       ``da.rio.set_spatial_dims()``.

-----

What a round trip guarantees
----------------------------

A dataset that goes out and comes back is the same raster: **CRS, geotransform,
nodata, dtype, band count, pixel values, band names, timestamp, and attrs are
all preserved**, and a second trip changes nothing a first one did not.

Three normalisations are applied on the way, by design rather than by accident:

- A 2-D DataArray comes back with a band dimension of length 1, since an
  Easy-EO raster always has bands.
- A labelled ``band`` coordinate (``band=["red", "nir"]``) is renumbered to
  ``1..n``. The labels are not lost — they travel in ``long_name`` and arrive as
  :attr:`~eeo.core.core.EEORasterDataset.band_names`.
- A timezone-aware :attr:`~eeo.core.core.EEORasterDataset.timestamp` returns as
  the same instant expressed in naive UTC, because xarray datetimes carry no
  timezone.

-----

Memory
------

:meth:`~eeo.core.core.EEORasterDataset.to_xarray` **reads the whole raster into
memory.** It is a conversion, not a lazy view, so clip or resample a full scene
before converting it:

.. code-block:: python

   da = ds.clip_raster_with_bbox(bbox).to_xarray()   # not the whole tile

The returned array never shares memory with the dataset, so writing into it is
safe. In the other direction, :func:`eeo.from_xarray` adds **no copy** on top of
the DataArray's own values — converting a large scene does not double its
memory — which means the dataset wraps the buffer the DataArray handed over.
Treat the conversion as handing that buffer to Easy-EO, and pass ``da.copy()``
if the two must stay fully separate. A dask-backed DataArray is computed during
the conversion, since an Easy-EO dataset is an in-memory one.

-----

When to use rioxarray instead
-----------------------------

Converting is cheap, but it is not always the right move. Stay in
xarray/rioxarray when:

- **The data is larger than memory.** rioxarray's dask chunking streams it;
  an Easy-EO dataset is in memory today, so the conversion is bounded by RAM.
- **Time is a real dimension of the problem.** Multi-date stacks, temporal
  reducers, and per-pixel trajectories are what xarray's dimension model is for.
  Easy-EO deliberately refuses to read a time dimension as bands.
- **The data has more than three dimensions**, or dimensions that are not bands
  and pixels at all.

Come back to Easy-EO when you want the chainable raster workflow — indices,
clipping, mosaicking, normalisation, plotting — on one scene at a time, without
managing dimensions by hand.
