Time Series Module
==================

An ordered, timestamped collection of rasters covering one area:
:class:`eeo.EEOTimeSeries`. Timesteps are kept deliberately distinct from
bands — bands are what a sensor measured at one moment, timesteps are the same
measurement repeated — so a spectral stack and a temporal stack can never be
confused for one another.

:func:`eeo.time_series` is the call to reach for. It reads whatever holds the
scenes and delegates: a STAC search result, its items, or datasets you loaded
yourself.

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

Building a series from a folder of rasters
(:meth:`eeo.EEOTimeSeries.from_folder`) is not implemented yet; the call exists
and names the code that does the same thing today.

.. autofunction:: eeo.time_series

.. autoclass:: eeo.EEOTimeSeries
    :members:
    :special-members: __len__, __getitem__
