Loading Satellite Data
======================

Most Earth-observation work starts the same way: find the scenes covering your
area over some period, then get the bands you need out of them. Easy-EO does
both against any **STAC** catalog — Microsoft Planetary Computer by default —
without downloading a single full scene.

:func:`~eeo.stac_search` finds the scenes. :meth:`~eeo.io.STACItem.load` reads
them, **cropped to your area of interest**, straight into an
:class:`~eeo.core.EEORasterDataset` you can chain operations onto.

This needs the optional ``stac`` extra:

.. code-block:: bash

   pip install "easy-eo[stac]"

.. note::

   **Which service is contacted.** :func:`~eeo.stac_search` defaults to
   ``catalog="https://planetarycomputer.microsoft.com/api/stac/v1"`` — Microsoft
   Planetary Computer. That is the only catalog Easy-EO ever queries unless you
   pass a different ``catalog=`` yourself; the value is also available as
   :data:`eeo.io.PLANETARY_COMPUTER_STAC_URL`.

   Loading pixels then reads from wherever the catalog says its assets live
   (for Planetary Computer, its Azure blob storage). Easy-EO does not choose
   that host — it is whatever URL the scene's metadata published.

.. seealso::

   :doc:`spectral_indices` for the indices computed here, :doc:`band_names` for
   addressing bands by name, and :doc:`sample_data` for working offline from a
   bundled sample instead.

-----

Quickstart: NDVI from a satellite archive
-----------------------------------------

Search, load, compute, plot — nothing downloaded but the pixels you asked for:

.. code-block:: python

   import eeo

   results = eeo.stac_search(
       "sentinel-2-l2a",
       bbox=(11.0, 46.5, 11.2, 46.7),
       datetime="2023-06-01/2023-08-31",
       cloud_cover=20,
       limit=1,
   )
   scene = results[0].load(["B04", "B08"])
   ndvi = scene.ndvi(red="B04", nir="B08")
   ndvi.plot_raster()

That is the whole workflow. The scene it reads is a Sentinel-2 tile of roughly
11,000 × 11,000 pixels per band — around 240 MB — but only the window covering
your bounding box is fetched, so the load moves a few megabytes and finishes in
seconds.

-----

Searching a catalog
-------------------

.. code-block:: python

   results = eeo.stac_search(
       "sentinel-2-l2a",
       bbox=(11.0, 46.5, 11.2, 46.7),
       datetime="2023-06-01/2023-08-31",
       cloud_cover=20,
       limit=10,
   )

``bbox``
   Your area of interest, ``(minx, miny, maxx, maxy)`` in **WGS 84 lon/lat
   degrees** — the STAC API expects degrees regardless of the imagery's own
   projection. It is also remembered as the default crop for loading (below).

``datetime``
   A single instant, an interval string such as ``"2023-06-01/2023-08-31"``, or
   a ``(start, end)`` pair. Use ``".."`` to leave an end open
   (``"2023-06-01/.."``). A bare date covers its whole day, so the closing date
   is inclusive.

``cloud_cover``
   Maximum scene cloud cover in percent.

``limit``
   Maximum number of scenes to return. Without it you get every match, which
   can be thousands.

``catalog``
   Which STAC API to search. **Defaults to Microsoft Planetary Computer**
   (``https://planetarycomputer.microsoft.com/api/stac/v1``); pass another
   endpoint to search somewhere else.

``sign``
   Whether to sign asset URLs. Left alone, it signs when — and only when —
   ``catalog`` is a Planetary Computer endpoint, which is what makes its assets
   readable; other catalogs are untouched.

The result is an ordered, timestamped sequence — oldest first, undated scenes
last — so it indexes, slices, and iterates like a list:

.. code-block:: python

   len(results)          # how many scenes matched
   results[0]            # the oldest
   results[-1]           # the most recent
   results[:3]           # another result holding the first three
   results.timestamps    # acquisition times, in order

Each entry describes one scene:

.. code-block:: python

   item = results[0]

   item.id            # the catalog's scene id
   item.timestamp     # acquisition time (timezone-aware)
   item.cloud_cover   # scene cloud cover, in percent
   item.asset_names   # the bands and products this scene offers
   item.properties    # everything else the catalog published
   item.item          # the underlying pystac Item, if you need the full API

Searching is metadata-only: it makes HTTP requests to the catalog but reads no
pixels, so scanning a long time series costs almost nothing.

-----

Choosing bands
--------------

``item.asset_names`` always lists exactly what a scene offers, and it is the
authority — asset keys are a property of the catalog, not of the satellite.

Sentinel-2 (``sentinel-2-l2a``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The two catalogs disagree here: Planetary Computer keys assets by the
Sentinel-2 band number, Earth Search by common name.

.. list-table::
   :header-rows: 1
   :widths: 30 20 25 25

   * - Band
     - Sentinel-2
     - Planetary Computer
     - Earth Search
   * - Blue
     - B02
     - ``B02``
     - ``blue``
   * - Green
     - B03
     - ``B03``
     - ``green``
   * - Red
     - B04
     - ``B04``
     - ``red``
   * - NIR
     - B08
     - ``B08``
     - ``nir``
   * - NIR (narrow)
     - B8A
     - ``B8A``
     - ``nir08``
   * - SWIR 1.6 µm
     - B11
     - ``B11``
     - ``swir16``
   * - SWIR 2.2 µm
     - B12
     - ``B12``
     - ``swir22``
   * - Scene classification
     - SCL
     - ``SCL``
     - ``scl``

Landsat 8/9 (``landsat-c2-l2``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both catalogs serve Landsat Collection 2 Level-2 under the same collection id
and — unlike Sentinel-2 — with the same common-name asset keys, so one set of
names works on either (though see the note on credentials below):

.. list-table::
   :header-rows: 1
   :widths: 34 26 40

   * - Band
     - Landsat 8/9 (OLI/TIRS)
     - Asset key (both catalogs)
   * - Coastal / aerosol
     - B1
     - ``coastal``
   * - Blue
     - B2
     - ``blue``
   * - Green
     - B3
     - ``green``
   * - Red
     - B4
     - ``red``
   * - NIR
     - B5
     - ``nir08``
   * - SWIR 1.6 µm
     - B6
     - ``swir16``
   * - SWIR 2.2 µm
     - B7
     - ``swir22``
   * - Thermal (surface temperature)
     - B10
     - ``lwir11`` (Planetary Computer only)
   * - QA / cloud mask
     - QA_PIXEL
     - ``qa_pixel``

Mind the band numbers when translating a formula between missions — Landsat's
NIR is B5 where Sentinel-2's is B8 — but the asset keys spare you that
arithmetic entirely. All Landsat Level-2 optical bands share a 30 m grid, so
stacking them needs no resampling:

.. code-block:: python

   results = eeo.stac_search(
       "landsat-c2-l2",
       bbox=(11.0, 46.5, 11.2, 46.7),
       datetime="2023-06-01/2023-08-31",
       cloud_cover=20,
       limit=10,
   )
   scene = results[0].load(["red", "nir08"])
   ndvi = scene.ndvi(red="red", nir="nir08")

**The collection is not only Landsat 8 and 9.** ``landsat-c2-l2`` spans the
whole Landsat archive, so a search over a recent summer happily returns
Landsat 7 scenes alongside 8 and 9. The asset keys stay the same — which is
precisely their value, since Landsat 7's red band is B3 where Landsat 8/9's is
B4 — but the sensors are not interchangeable, and Landsat 7 imagery after 2003
carries scan-line-corrector gaps. Filter by platform when the mission matters:

.. code-block:: python

   modern = [
       item
       for item in results
       if item.properties.get("platform") in ("landsat-8", "landsat-9")
   ]
   scene = modern[0].load(["red", "nir08"])

**Landsat on Earth Search needs AWS credentials.** Its assets are ``s3://``
hrefs into the requester-pays ``usgs-landsat`` bucket, so a load fails with a
GDAL credentials error unless your environment is configured for it. Planetary
Computer serves the same scenes over HTTPS and signs them automatically, which
makes it the path of least resistance for Landsat.

.. seealso::

   :doc:`spectral_indices` lists the band each index expects, per mission.

-----

Loading, cropped to your area
-----------------------------

A load returns only the area of interest. The assets are read remotely as HTTP
range requests against the cloud-optimized GeoTIFFs, so the pixels outside your
box are never transferred:

.. code-block:: python

   scene = results[0].load(["B04", "B08"])

By default it crops to the ``bbox`` you searched with — the item remembers it as
``item.search_bbox``, so you never repeat yourself. Override or opt out per
load:

.. code-block:: python

   # A different (usually smaller) area
   field = results[0].load("B04", bbox=(11.05, 46.55, 11.08, 46.58))

   # The entire scene - a full Sentinel-2 band, so expect hundreds of MB
   whole_tile = results[0].load("B04", crop=False)

Ask for several assets and they come back as one multi-band dataset, each band
named after the asset it came from, ready to address by name:

.. code-block:: python

   scene = results[0].load(["B04", "B08", "B11"])

   scene.band_names            # ['B04', 'B08', 'B11']
   scene.timestamp             # the scene's acquisition time
   scene.attrs["stac_item"]    # which catalog scene it came from

Sentinel-2 bands do not share a resolution — B11 is 20 m where B04 and B08 are
10 m — so assets that do not match the **first** asset's grid are resampled onto
it (nearest neighbour by default, ``resampling="bilinear"`` and friends if you
prefer). List your finest band first to keep full resolution, or a coarser one
first to downsample everything cheaply.

The loaded dataset is an ordinary Easy-EO raster, so the rest of the library
follows:

.. code-block:: python

   result = (
       results[0]
       .load(["B04", "B08"])
       .ndvi(red="B04", nir="B08")
       .normalize_min_max()
   )
   result.save_raster("ndvi.tif")

-----

Using a different catalog
-------------------------

Searches go to Microsoft Planetary Computer unless you say otherwise. To use
another archive, name it — any STAC API works. Earth Search (AWS Open Data)
needs no account:

.. code-block:: python

   results = eeo.stac_search(
       "sentinel-2-l2a",
       bbox=(11.0, 46.5, 11.2, 46.7),
       datetime="2023-06-01/2023-08-31",
       cloud_cover=20,
       limit=1,
       catalog="https://earth-search.aws.element84.com/v1",
   )
   scene = results[0].load(["red", "nir"])
   ndvi = scene.ndvi(red="red", nir="nir")

Planetary Computer asset URLs are signed automatically; other catalogs are left
alone. Pass ``sign=False`` to skip signing, or ``sign=True`` to force it.

-----

Things worth knowing
--------------------

**Signed URLs expire.** Planetary Computer signatures are short-lived, so
search and load in the same session rather than storing search results for
later.

**The cloud filter is the catalog's, not ours.** Some catalogs match against
their own indexed value, so a returned scene can sit slightly above your
threshold. When an exact cut-off matters, filter the result yourself:

.. code-block:: python

   clear = [item for item in results if item.cloud_cover <= 20]

**Catalogs return reprocessed duplicates.** One acquisition can appear as
several scenes with the same timestamp and different processing baselines.
They are kept as separate items; pick by ``item.id`` if it matters.

**Catalog outages surface as errors.** A search hits somebody else's service.
If it is down you will see the client's transport error, not an Easy-EO one —
retry later, or point ``catalog=`` elsewhere.

**Memory.** Only the window you asked for is held in memory, which is exactly
why cropping is the default. ``crop=False`` on a Sentinel-2 band means a
full-resolution band in RAM.
