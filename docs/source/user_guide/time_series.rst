Time Series Analysis
====================

One satellite image tells you what a place looked like on one morning. Most
questions are not about one morning: *when did this field green up?* *what does
this valley normally look like in August?* *where was the flood a week later?*

Those need the same place seen many times. A satellite has seen it many times —
Sentinel-2 passes every few days — so the data exists; the work is in handling
fifty scenes as one thing instead of fifty.

That is what a time series is for.

.. seealso::

   :doc:`loading_satellite_data` for finding scenes in a catalog,
   :doc:`masking_clouds` for what masking does to a single scene, and the
   `time-series tutorial notebook
   <https://github.com/Tommy-Burns/easy-eo/blob/main/examples/04_timeseries/01_cloud_free_composite_and_trends.ipynb>`_
   for a complete worked example.

-----

The short version
-----------------

Search a catalog, hand the results over, and ask for a cloud-free image:

.. code-block:: python

   import eeo

   results = eeo.stac_search(
       collection="sentinel-2-l2a",
       bbox=(5.60, 52.05, 5.65, 52.085),
       datetime="2023-04-01/2023-09-30",
   )

   ts = eeo.time_series(source=results, assets=["B04", "B08", "SCL"])
   clear = ts.composite()

   ndvi = clear.ndvi(red="B04", nir="B08")

``clear`` is one ordinary raster, built from whichever acquisition saw each
pixel through the clouds. Everything else on this page is variations on those
four lines.

-----

Building a series
-----------------

:func:`eeo.time_series` reads whatever holds your scenes:

.. code-block:: python

   ts = eeo.time_series(source=results, assets=["B04", "B08", "SCL"])  # a catalog search
   ts = eeo.time_series(source="scenes/")                              # a folder of GeoTIFFs
   ts = eeo.time_series(source=[scene_may, scene_june, scene_july])    # scenes you loaded

Every timestep needs to know when it was taken. Catalog results and the
Sentinel-2 and Landsat loaders record that for you. A folder has to be read from
the filenames, which works as long as they carry a date — ``20230412``,
``2023-04-12``, or either with a time after it, which is how Sentinel-2 and
Landsat files arrive. Where they do not, say where the date is instead — the
function is handed each file's path, so it can read the date from anywhere. Here
the files are named ``B04.tif`` and it is the folder around each one that carries
the date:

.. code-block:: python

   import datetime as dt

   ts = eeo.time_series(
       source="scenes/",             # scenes/2023-04-12/B04.tif
       pattern="*/B04.tif",
       timestamp=lambda path: dt.datetime.fromisoformat(path.parent.name),
   )

And if you built the scenes by hand, pass the dates yourself:

.. code-block:: python

   ts = eeo.time_series(source=scenes, timestamps=[date_one, date_two, date_three])

The series sorts itself oldest-first, and behaves like a list: ``len(ts)``,
``ts[0]``, ``ts[2:5]``, and ``for scene in ts``. Each timestep is a normal
dataset with every operation available on it.

Include the quality band (``"SCL"`` for Sentinel-2, ``"qa_pixel"`` for Landsat)
if you want cloud handled for you later.

.. note::

   Loading from a catalog reads only your area of interest, not whole tiles, and
   keeps each scene in a temporary cache so the series holds files rather than
   pixels. Pass ``cache="scenes/"`` to keep those files: catalog download links
   expire, but files on your disk do not.

-----

One operation, every timestep
-----------------------------

:meth:`~eeo.EEOTimeSeries.map` applies any Easy-EO operation to every timestep
and gives you a new series back. It is the same function you would call on one
scene:

.. code-block:: python

   ndvi_series = ts.map(eeo.ndvi, red="B04", nir="B08", name="ndvi")
   clipped = ts.map(eeo.clip_raster_with_vector, vector_file=boundary)
   masked = ts.map(eeo.mask_clouds)

Your own functions work too — anything that takes a dataset and returns one:

.. code-block:: python

   brightened = ts.map(lambda scene: scene.multiply(other=2))

The original series is untouched; ``map`` builds a new one.

-----

Collapsing a series into one raster
-----------------------------------

Four methods turn a series back into a single dataset:

.. code-block:: python

   typical = ts.median()      # the usual value at each pixel
   average = ts.mean()
   peak = ndvi_series.max()   # how green each pixel ever got
   lowest = ts.min()

Reach for ``median()`` over ``mean()`` on real imagery: a cloud missed by the
mask is an odd value among the others, which a median ignores and an average
does not.

Cloudy timesteps are not a problem — they are simply absent. If a pixel was
under cloud on two dates out of five, its median comes from the three dates that
saw it. Only a pixel that was never seen at all comes back empty.

And ``composite()`` is the one to reach for on satellite imagery, because it
handles the cloud first:

.. code-block:: python

   clear = ts.composite()

   clear.band_names
   # ['B04', 'B08']

That is :meth:`~eeo.EEOTimeSeries.map` with :func:`eeo.mask_clouds` followed by
``median()``, in one call — with the quality band left out of the result, since
an average of scene-class numbers would not mean anything.

One raster for the whole season is not always the question. To collapse *within*
each month instead of across everything, group first:

.. code-block:: python

   monthly = ts.resample_time(freq="MS").composite()

   len(monthly)              # one timestep per month
   monthly.timestamps[0]     # the month it covers

That is a series again, so everything on this page still works on it — including
reducing it a second time, which is how you get the greenest month of a season.
Periods are written the way pandas writes them: ``"D"`` a day, ``"7D"`` seven
days, ``"W"`` a week, ``"MS"`` a month, ``"YS"`` a year. A month with no
acquisitions is simply absent from the result.

Comparing two series
--------------------

Two sets of scenes of the same place, years apart. What changed?

.. code-block:: python

   before = eeo.time_series(source=results_2020, assets=["B04", "B08"])
   after = eeo.time_series(source=results_2023, assets=["B04", "B08"])

   change = after.map_with(before, eeo.subtract)
   change.median()             # the usual change

The scenes are paired off in order — first with first, second with second — so
``change`` holds one raster per pair, and is a series like any other.

The series you call the method on goes in first, so that reads *after minus
before*. The two have to hold the same number of scenes. Their dates are not
expected to line up, since being from different years is the whole point.

Do not confuse it with handing ``map`` a raster, which uses the same one at every
timestep:

.. code-block:: python

   after.map(eeo.subtract, other=one_raster)   # every scene less that one raster
   after.map_with(before, eeo.subtract)        # every scene less its own partner

-----

Following one place through time
--------------------------------

The other direction: instead of collapsing time, collapse space.
:meth:`~eeo.EEOTimeSeries.extract_at` samples one location at every timestep:

.. code-block:: python

   trend = ndvi_series.extract_at(coordinates=(5.625, 52.0675), crs="EPSG:4326")

   trend.plot()                  # a chart of the season
   trend["ndvi"].idxmax()        # the date it peaked
   trend["ndvi"].mean()

What comes back is a pandas ``DataFrame``, indexed by acquisition date with one
column per band — so anything you already do with pandas works here. Dates when
the pixel was under cloud show as ``NaN``: a gap reads as a gap rather than as a
sudden dip.

Give the coordinates in longitude and latitude with ``crs="EPSG:4326"``, or in
the imagery's own units without it.

Two plots
---------

A line through time at one place, and a contact sheet of every date:

.. code-block:: python

   ndvi.plot_trajectory(coordinates=(5.625, 52.0675), crs="EPSG:4326")
   ndvi.plot_filmstrip(cmap="RdYlGn")

The trajectory breaks where the pixel was clouded, so a gap looks like a gap. The
filmstrip gives every timestep the same colour scale — that is what lets you
compare the dates, and it is why a cloudy one looks obviously wrong. It is the
fastest way to see what you actually downloaded.

-----

Things worth knowing
--------------------

**Do not filter the clouds out of your search.** It is tempting to ask only for
clear scenes, but a cloudy scene still sees the ground somewhere, and that
somewhere may be the only look you get at it. Let the cloudy ones in and let
``composite()`` sort out which pixels to use.

**Catalogs list some scenes twice.** A scene gets published again whenever the
mission reprocesses its archive, and both copies match a search — which counts
that day twice in a composite, and pulls it toward whichever dates happen to be
duplicated. Drop the extras before you read anything:

.. code-block:: python

   ts = eeo.time_series(source=results.deduplicate(), assets=["B04", "B08", "SCL"])

The best-processed copy of each acquisition survives. Two tiles of one overpass
are not duplicates and are both kept — over a wide area they cover different
ground. A series built without this says so when it notices, and
``ts.deduplicate()`` fixes one you have already built.

**Timesteps must line up.** Every scene in a series has to be on the same grid,
which is normally automatic for one area from one catalog. A wide area can draw
scenes that were tiled differently; the series will say so, and
``auto_align=True`` (or ``auto_reproject=True`` across a change of projection)
puts them on the first scene's grid for you.

**Large areas.** A composite of many full scenes does not have to fit in memory:
``save_path=`` writes the result straight to disk, and ``mask_dir=`` does the
same for the masked scenes along the way.

.. code-block:: python

   ts.composite(mask_dir="masked/", save_path="composite.tif")

**Sentinel-2 and January 2022.** The mission changed how it stores its numbers
on 25 January 2022. Easy-EO gives you the stored numbers, so a series that spans
that date mixes two conventions and will warn you. Keep a series on one side of
it.

**If you already know xarray**, ``ts.to_xarray()`` hands the whole series over as
one array with a ``time`` dimension, and everything you do there works from
that point on. It reads every scene into memory, so reduce or slice the series
first if it covers full tiles.

-----

Where to look next
------------------

- The `tutorial notebook
  <https://github.com/Tommy-Burns/easy-eo/blob/main/examples/04_timeseries/01_cloud_free_composite_and_trends.ipynb>`_
  works a full season end to end, with figures.
- :doc:`../modules/timeseries` documents every argument these calls take.
- :doc:`masking_clouds` covers choosing what counts as cloud.
- :doc:`large_rasters` covers memory in general.
