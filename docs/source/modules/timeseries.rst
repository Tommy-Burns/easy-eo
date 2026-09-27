Time Series Module
==================

.. seealso::

   :doc:`../user_guide/time_series` introduces these calls with a worked
   workflow; this page is the argument-by-argument reference.

An ordered, timestamped collection of rasters covering one area:
:class:`eeo.EEOTimeSeries`. Timesteps are kept deliberately distinct from
bands — bands are what a sensor measured at one moment, timesteps are the same
measurement repeated — so a spectral stack and a temporal stack can never be
confused for one another.

:func:`eeo.time_series` is the call to reach for. It reads whatever holds the
scenes and delegates: a STAC search result, its items, a folder of rasters, or
datasets you loaded yourself.

A catalog search is the shortest path, because every item already states its
acquisition time:

.. code-block:: python

    import eeo

    results = eeo.stac_search(
        "sentinel-2-l2a",
        bbox=(11.0, 46.5, 11.2, 46.7),
        datetime="2023-04-01/2023-09-30",
        cloud_cover=20,
    )
    ts = eeo.time_series(results, assets=["B04", "B08"])

    len(ts)                 # timesteps, oldest first
    ts.timestamps[0]        # when the first one was acquired
    ndvi = ts[0].ndvi(red="B04", nir="B08")   # each timestep is an ordinary dataset

Scenes read from a catalog are written to a temporary cache and reopened from
there, so the series holds file handles rather than arrays and a long series
stays bounded in memory; :meth:`eeo.EEOTimeSeries.close` removes them. Pass
``cache=`` a directory to keep the files instead — signed catalog URLs expire,
cached GeoTIFFs do not — or ``cache=False`` to keep every scene in memory,
which is faster for a small area.

A folder of rasters
-------------------

:meth:`eeo.EEOTimeSeries.from_folder` builds a series from files on disk — one
raster per timestep — which is what scenes downloaded, exported from another
tool, or written by an earlier step of your own look like:

.. code-block:: python

    ts = eeo.time_series("scenes/")                        # scenes/*.tif
    ts = eeo.time_series("scenes/", pattern="**/*.tif")    # subdirectories too

The files are opened, not read, so the series is file-backed from the start and
holds no pixels.

A file, unlike a catalog item, does not state when it was acquired — but its name
usually does, and that is what is read: ``20230412``, ``2023-04-12``, or either
with a time after it (``20230412T100621``), which covers Sentinel-2 and Landsat
names as they are delivered. The first real date in the name wins, so the
acquisition date of a Landsat product id is taken rather than its processing
date. Only the filename is read, never the directories above it — a folder named
by date is a convention, and guessing at it would make a timestep's date depend
on where its file was stored.

``timestamp=`` takes over where the filename does not carry one. It is a
function of the path, so it can read the date from wherever it actually lives:

.. code-block:: python

    ts = eeo.time_series(
        "scenes/",
        pattern="*/B04.tif",
        timestamp=lambda path: dt.datetime.fromisoformat(path.parent.name),
    )

``chunks=`` opens the files on the lazy backend instead, and the
grid-consistency arguments apply as everywhere else.

One acquisition, one timestep
-----------------------------

A catalog publishes a scene again each time the mission reprocesses its archive,
and every copy matches a search. That is correct of the catalog and wrong for a
series: a statistic across time counts each timestep once, so a repeated moment
is weighted twice in a median and the composite leans toward whichever dates
happen to be duplicated.

Drop the extra copies before anything is read:

.. code-block:: python

    ts = eeo.time_series(results.deduplicate(), assets=["B04", "B08", "SCL"])

:meth:`eeo.io.STACSearchResult.deduplicate` decides on metadata the search
already returned, so a duplicate never costs a read.
:meth:`eeo.EEOTimeSeries.deduplicate` applies the same rule to a series already
built, which is what the folder and hand-assembled paths have.

Two items are the same acquisition when they share a collection, an acquisition
time, and the ground they cover — read from ``grid:code`` (``"MGRS-33TUL"`` for
Sentinel-2, ``"WRS2-192029"`` for Landsat), or from the footprint where the
catalog declares no grid. Of the copies, the winner is:

1. the **highest processing version** — for Sentinel-2 the processing baseline;
2. else, the **most recently processed**, from ``processing:datetime``,
   or from ``updated`` or ``created`` where the catalog states none;
3. else, whichever the catalog listed first.

Version comes before time because ``created`` and ``updated`` describe the STAC
record and not the data — the specification says so — and a metadata-only fix
must not let an older processing outrank a better one.

**Two tiles of one overpass are not duplicates.** They share an acquisition time
and cover different ground, so both are kept: over an area exceeding a tile
boundary each holds a different part of it, and they want :func:`eeo.mosaic`
rather than dropping. This is why nothing is deduplicated automatically — a
series that repeats a moment only *warns*, naming both causes, because the two
have opposite fixes.

One grid, one set of bands
--------------------------

A series is checked when it is built, on metadata alone: every timestep must be
in the same CRS, on the same pixel grid, and hold the same bands. A mismatch is
refused rather than papered over — timesteps that do not share a grid cannot be
compared pixel by pixel, and a band that means "red" at one timestep and "nir"
at another makes every index computed across them wrong.

Alignment is opt-in, as it is elsewhere in Easy-EO:

.. code-block:: python

    ts = eeo.time_series(results, assets=["B04", "B08"], auto_align=True)
    ts = eeo.time_series(results, assets=["B04", "B08"], auto_reproject=True)

``auto_align=True`` warps timesteps on a different grid onto the reference's;
``auto_reproject=True`` does the same across a CRS change — which a search wide
enough to cross a UTM zone boundary will need. Either way the warp lands on the
reference grid exactly, not merely on its shape. ``method=`` chooses the
resampling, and defaults to ``"nearest"`` because a series built for cloud
masking carries a quality band whose values are class numbers. ``reference=``
picks which timestep sets the grid, in time order; the earliest by default.

Sentinel-2 and the 2022 baseline change
---------------------------------------

Easy-EO reads the values a product stores, and does not decode them to
reflectance. That matters for one date: from processing baseline 04.00, deployed
on 25 January 2022, a Sentinel-2 Level-2A product shifts its stored values by
``BOA_ADD_OFFSET`` — in practice −1000 DN for every band — so the same ground
reads about 1000 DN higher after the change than before it.

A series that spans that boundary therefore mixes two conventions: a composite
over it is biased by however many scenes fall on each side, and an index
trajectory shows a step at the boundary that is not in the ground. Easy-EO warns
when it sees one, reading each scene's recorded baseline where there is one
(the reprocessed archive carries 04.00 or later on much older acquisitions, so
the acquisition date alone would misjudge those) and falling back to the date
otherwise. Keep the series on one side of that date, or use scenes reprocessed
to a single baseline.

One operation, every timestep
-----------------------------

:meth:`eeo.EEOTimeSeries.map` applies any operation that takes a dataset and
returns one — the spectral indices, the algebra, clipping, masking, or a
function of your own — to every timestep, and returns a new series. The
operation is unchanged: it is the same function you would call on one scene, and
it behaves exactly as it does there, carrying band names, timestamp and attrs
onto each result:

.. code-block:: python

    ndvi = ts.map(eeo.ndvi, red="B04", nir="B08")
    clear = ts.map(eeo.mask_clouds).map(eeo.ndvi, red="B04", nir="B08")
    doubled = ts.map(lambda scene: scene.multiply(2))

The results are held in memory, which is fine for an area of interest and not
for whole scenes. ``save_dir=`` writes each result to a GeoTIFF and returns a
series reading those files instead, so peak memory is one result rather than the
whole series:

.. code-block:: python

    ndvi = ts.map(eeo.ndvi, red="B04", nir="B08", save_dir="ndvi/")

Datasets loaded by hand work the same way, as long as each carries a timestamp:

.. code-block:: python

    scenes = [
        eeo.load_sentinel2(path, bands=["red", "nir", "scl"])
        for path in products
    ]
    ts = eeo.time_series(scenes)

Collapsing a series to one raster
---------------------------------

Four reducers take a series and return a single dataset: ``median()``,
``mean()``, ``min()`` and ``max()``.

.. code-block:: python

    composite = ts.median()
    peak_greenness = ts.map(eeo.ndvi, red="B04", nir="B08").max()
    ts.median(save_path="composite.tif")     # never held in memory

**Nodata across time is absent, not contagious.** Elsewhere in Easy-EO a pixel
missing from any operand is missing from the output; that rule is for operations
*combining* rasters. A reducer computes a statistic, and a statistic treats
nodata as absent — so a pixel clouded at two of five timesteps still gets a
median.

``median()`` and ``mean()`` are float32 with NaN for such a pixel: a median over
an even number of timesteps averages the two middle values. ``min()`` and
``max()`` keep the timesteps' own dtype, because they select a value that was
measured rather than computing a new one, and mark a missing pixel with the
timesteps' nodata value. All four return a plain dataset that chains like any
other.

The result carries no timestamp — a composite was not acquired at any one moment
— and records what it reduced in ``attrs``.

Monthly composites: binning in time
-----------------------------------

Forty acquisitions is rarely the number a question is asked in. *"How did this
field green up?"* is about months. :meth:`eeo.EEOTimeSeries.resample_time`
groups the timesteps into periods, and the reducers then work *within* each
period instead of across the whole series:

.. code-block:: python

    monthly = ts.resample_time("MS").median()      # one raster per calendar month
    monthly = ts.resample_time("MS").composite()   # ...cloud-free
    weekly = ts.resample_time("7D").max()

What comes back is a series like any other — one timestep per period — so it can
be mapped over, sampled, sliced, saved, or reduced again:

.. code-block:: python

    monthly = ts.resample_time("MS").composite()
    greenest = monthly.map(eeo.ndvi, red="B04", nir="B08").max()

Periods are pandas offset aliases, passed to pandas untouched: ``"D"`` a day,
``"7D"`` seven days, ``"W"`` a week, ``"MS"`` a calendar month, ``"QS"`` a
quarter, ``"YS"`` a year, and anchored forms such as ``"W-MON"``.

.. note::

   Prefer the **start-of-period** spellings above. pandas 2.2 renamed the
   end-of-period aliases — ``"M"`` became ``"ME"``, ``"Q"`` became ``"QE"``,
   ``"Y"`` became ``"YE"`` — and Easy-EO supports pandas on both sides of that
   change, so ``"MS"`` works on every installation while ``"M"`` does not. A
   period Easy-EO cannot use is refused with that rename named.

A period holding no acquisition is dropped rather than carried: a series cannot
hold a timestep with no raster behind it, so a cloudy May simply is not in the
result. Grouping itself reads nothing — it is arithmetic on the timestamps — and
the grouping is inspectable before you commit to reducing it:

.. code-block:: python

    periods = ts.resample_time("MS")
    len(periods)                       # how many months have anything in them
    [len(period) for period in periods]  # how many acquisitions each holds

Each result is stamped with its **period's label**, because a reducer states no
timestamp of its own — a composite was not acquired at any one moment — and
records the span it actually covers in ``attrs`` (``time_start``, ``time_end``,
``timesteps``), plus the period under ``temporal_bin``. ``save_dir=`` writes one
raster per period and reads the series back from those files, which is what makes
a season of full scenes workable.

A cloud-free composite
----------------------

``composite()`` is the reason a time series is worth having. Each timestep is
masked with its own quality band — Sentinel-2's ``SCL`` or Landsat's
``QA_PIXEL``, whichever it carries — and the masked timesteps are reduced across
time:

.. code-block:: python

    import eeo

    results = eeo.stac_search(
        "sentinel-2-l2a",
        bbox=(11.0, 46.5, 11.2, 46.7),
        datetime="2023-04-01/2023-09-30",
    )
    ts = eeo.time_series(results, assets=["B04", "B08", "SCL"])

    clear = ts.composite()          # ['B04', 'B08'] — no SCL band
    ndvi = clear.ndvi(red="B04", nir="B08")

The quality band is **not** in the result: it has done its work, and a median of
scene-class numbers would be a class no classifier ever assigned. A pixel that
was clouded at *every* timestep is the one a composite cannot fill, and comes
back as nodata rather than as whatever the cloud looked like.

``how=`` chooses the statistic (median by default — a missed cloud edge at one
timestep is an outlier a median discards and a mean averages in), and
``classes=``, ``flags=``, ``min_cloud_confidence=``, ``mission=`` and ``nodata=``
pass through to :func:`eeo.mask_clouds` unchanged. Masking reads a whole scene,
so ``mask_dir=`` writes the masked timesteps out instead of holding them all,
and ``save_path=`` does the same for the composite itself.

It is the same thing as ``ts.map(eeo.mask_clouds).median()`` minus the quality
band, spelled as one call because it is the workflow the series exists for. Do
it by hand when a timestep's mask lives in a separate raster.

What happened here: a trajectory
--------------------------------

The counterpart to a composite. Where the reducers collapse time into one
raster, :meth:`eeo.EEOTimeSeries.extract_at` collapses space into one table —
what happened at a single location, indexed by time:

.. code-block:: python

    ndvi = ts.map(eeo.ndvi, red="B04", nir="B08", name="ndvi")
    trajectory = ndvi.extract_at((11.1, 46.6), crs="EPSG:4326")

    trajectory                      # a pandas DataFrame indexed by acquisition time
    trajectory["ndvi"].idxmax()     # when it was greenest
    trajectory.plot()               # straight into matplotlib

It reads one pixel per timestep and band — never a band, never a scene — so a
trajectory over a season of full Sentinel-2 tiles costs a few dozen pixels.
``crs=`` transforms the point for you, which saves converting the lon/lat a
catalog search handed back. Columns are named after the bands, and a pixel that
was nodata at a timestep is ``NaN`` there rather than its fill value, so a
cloudy date reads as a gap.

Several locations are a concat of several calls:

.. code-block:: python

    import pandas as pd

    plots = {"north": (11.10, 46.60), "south": (11.15, 46.55)}
    table = pd.concat(
        {name: ndvi.extract_at(point, crs="EPSG:4326") for name, point in plots.items()},
        names=["plot"],
    )

.. autofunction:: eeo.time_series

.. autoclass:: eeo.EEOTimeSeries
    :members:
    :special-members: __len__, __getitem__

.. autoclass:: eeo.timeseries.TemporalBins
    :members:
    :special-members: __len__, __iter__
