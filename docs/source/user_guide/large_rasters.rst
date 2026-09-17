Working with Large Rasters
==========================

A Sentinel-2 tile is 10980 x 10980 pixels per band; a Landsat scene is
8081 x 7991. Held whole, in the float32 a computation needs, one band of either
is about half a gigabyte — and a laptop that reads two of them, computes an
index, and keeps the result is already juggling two gigabytes to answer one
question.


.. seealso::

   :doc:`../backends` for the adapter model underneath this page,
   :doc:`nodata_and_dtype` for what happens to nodata along the way, and
   :doc:`ops` for the operations themselves.

-----

The short version
-----------------

.. list-table::
    :header-rows: 1
    :widths: 34 66

    * - If you want to
      - Do this
    * - Process a scene larger than memory
      - Nothing special. Algebra, the indices and normalization already run a
        block at a time.
    * - Keep a whole *chain* bounded
      - Write each big intermediate to disk, or run one pass with
        :func:`~eeo.core.blockwise.apply_blockwise` and ``save_path=``.
    * - Avoid reading a file you only want metadata from
      - ``eeo.load_raster(path)`` — opening never reads pixels.
    * - Read a scene from a URL
      - ``eeo.load_raster("https://.../scene.tif")`` — see :doc:`../backends`.
    * - Chunk a raster and read only parts of it
      - ``eeo.load_raster(path, chunks="auto")``, with the ``lazy`` extra.
    * - Know what a call costs before you make it
      - The table in `What holds a whole raster`_.

-----

Streaming is the default
------------------------

Every pixel-wise operation runs through a block-wise engine: it reads one
window of each input, computes that window, writes it into the output, and
moves on. Peak memory follows the block, not the scene, and no setting turns it
on — a raster small enough to fit in one block simply gets one block.

This covers all the algebra (:meth:`~eeo.core.core.EEORasterDataset.add` and
friends), every spectral index, and
:meth:`~eeo.core.core.EEORasterDataset.normalize_min_max`.

Operations that need a statistic over the whole raster before they can touch a
pixel — :meth:`~eeo.core.core.EEORasterDataset.standardize`,
:meth:`~eeo.core.core.EEORasterDataset.normalize_percentile`, and the
``get_*_pixel`` family — make two bounded passes instead: measure, then apply.
Neither pass holds the raster.

.. note::

   Percentiles of an *integer* raster are computed exactly, by counting values
   rather than sorting them — which covers every raw Sentinel-2 and Landsat
   band, since those are the rasters big enough for it to matter. A
   floating-point raster is read to compute a percentile, because an exact
   percentile of float data cannot be had in bounded memory.

-----

Where a chain stops being bounded
---------------------------------

Each operation *returns* its result, and a returned raster is held in memory.
One operation on a scene larger than memory is fine; a chain of them is not,
because the first intermediate has nowhere to live:

.. code-block:: python

   # Bounded: one pass, result written straight to disk.
   ds.ndvi("red", nir="nir").save_raster("ndvi.tif")

   # Not bounded on a very large scene: three full-size intermediates.
   ds.add(5).multiply(2).normalize_min_max()

For a scene that genuinely does not fit, do the whole computation in one pass
and stream it to a file. :func:`~eeo.core.blockwise.apply_blockwise` is public
for exactly this: hand it a plain NumPy function and the bands it applies to.

.. code-block:: python

   import numpy as np
   from eeo.core.blockwise import BlockSource, apply_blockwise

   def savi(nir, red, soil=0.5):
       nir, red = nir.astype("float32"), red.astype("float32")
       return (nir - red) * (1 + soil) / (nir + red + soil)

   result = apply_blockwise(
       ds,
       savi,
       sources=[BlockSource.from_dataset(ds, band="nir"),
                BlockSource.from_dataset(ds, band="red")],
       fractional=True,
       save_path="savi.tif",
   )

The nodata and dtype rules apply per block exactly as they do to a built-in
operation, and ``save_path=`` means the output never exists in memory either.

**Measured**, on an 8000 x 8000 two-band scene (268 MB on disk), each run in
its own process:

.. list-table::
    :header-rows: 1
    :widths: 58 42

    * - Computing NDVI of the whole scene
      - Peak memory
    * - ``apply_blockwise(..., save_path=...)``
      - 490 MiB
    * - ``ds.ndvi(...).save_raster(...)`` (result held in memory)
      - 1181 MiB
    * - Whole-array NumPy on both bands
      - 1293 MiB

About 256 MiB of each figure is GDAL's own block cache, not the computation.

-----

Opening a raster lazily
-----------------------

The ``lazy`` extra adds a second backend: the raster is opened as a
dask-chunked :class:`xarray.DataArray` instead of through rasterio.

.. code-block:: bash

   pip install "easy-eo[lazy]"

.. code-block:: python

   ds = eeo.load_raster("scene.tif", chunks="auto")        # dask picks sizes
   ds = eeo.load_raster("scene.tif", chunks={"y": 2048, "x": 2048})

Nothing else in your code changes — every operation, statistic and plot works
the same way on either backend, and returns the same answer.

What it gives you:

- **Metadata for free.** Opening the file and reading its CRS, transform,
  dtype, nodata or band names touches no pixels at all.
- **Reads bounded to what you ask for.** ``ds.read(1, window=...)`` computes
  only the chunks that window covers.
- **Remote rasters.** A COG over HTTP is read in place, by range request; see
  :doc:`../backends`.

What it does *not* give you, despite the name:

.. note::

   An operation on a lazy dataset does not build a dask graph and does not
   return a lazy result. It computes there and then and hands back an ordinary
   raster. Easy-EO promotes the dataset to the rasterio backend first — which
   for a dataset opened from a file costs nothing, because it reopens the file
   rather than reading it — and then streams block by block as usual.

So the lazy backend is not what makes large scenes work; block-wise execution
is, and that is always on. Reach for ``chunks=`` when you want an unread view
of a file, selective window reads, or a raster that lives behind a URL.

.. note::

   ``chunks="auto"`` caps chunks at dask's configured chunk size, 128 MiB by
   default — which a single Landsat or Sentinel-2 band fits inside, so it can
   come back as one chunk. Pass explicit sizes if you want it cut finer.

-----

Statistics, descriptions and plots
----------------------------------

:meth:`~eeo.core.core.EEORasterDataset.describe` reads nothing at all unless
you ask it for statistics, and then how much it reads is up to you:

.. code-block:: python

   ds.describe()                    # structure only — no pixels read
   ds.describe(stats="approx")      # decimated read, capped at 1024 px a side
   ds.describe(stats="exact")       # reads every pixel

Plot functions read at display resolution: a 10980-pixel band drawn into a
600-pixel figure is read decimated, served from the file's overviews when it
has them.

.. warning::

   :meth:`~eeo.core.core.EEORasterDataset.plot_histogram` is the exception. A
   histogram is a statistic over every pixel, so it reads the band in full —
   about 950 MiB on a full Landsat band, against roughly 300 MiB for the other
   plots. Clip or resample first if that matters.

-----

What holds a whole raster
-------------------------

Everything below is bounded by the *scene*, not by a block. None of it is a
defect — each one's job is to produce or consume a whole array.

.. list-table::
    :header-rows: 1
    :widths: 38 62

    * - Call
      - Holds
    * - :meth:`~eeo.core.core.EEORasterDataset.to_array`,
        :meth:`~eeo.core.core.EEORasterDataset.get_band`
      - The array you asked for — that is the point of them
    * - :meth:`~eeo.core.core.EEORasterDataset.stack`,
        :meth:`~eeo.core.core.EEORasterDataset.mosaic`
      - Every input, plus the combined output
    * - :meth:`~eeo.core.core.EEORasterDataset.to_xarray`
      - A copy of the raster (it is a boundary crossing, not a view)
    * - :meth:`~eeo.core.core.EEORasterDataset.describe` with ``stats="exact"``
      - One band at a time, promoted to float64 for the statistics
    * - :meth:`~eeo.core.core.EEORasterDataset.plot_histogram`
      - One band at a time, as above
    * - :meth:`~eeo.core.core.EEORasterDataset.mask_clouds`,
        :meth:`~eeo.core.core.EEORasterDataset.clear_fraction`
      - The scene and its quality band
    * - Any operation's **result**
      - The output raster, unless you streamed it to a file
