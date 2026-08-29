Loading Downloaded Scenes
=========================

:doc:`loading_satellite_data` reads scenes from a catalog over the network.
This page covers the other case: a product you have already downloaded, sitting
on disk as a folder or as the archive it arrived in.

:func:`~eeo.load_sentinel2` reads a Copernicus ``.SAFE`` product.
:func:`~eeo.load_landsat` reads a USGS Collection 2 Level-2 product — one
function for Landsat 4, 5, 7, 8, and 9. Both return the same
:class:`~eeo.core.EEORasterDataset` as a catalog load, on the same grid, with
the same band names, so a workflow does not care which route the data took.

Neither needs an optional extra.

.. seealso::

   :doc:`loading_satellite_data` for reading the same scenes from a catalog,
   :doc:`band_names` for addressing bands by name, and :doc:`spectral_indices`
   for the indices computed here.

-----

Quickstart
----------

.. code-block:: python

   import eeo

   scene = eeo.load_sentinel2(
       path="S2B_MSIL2A_20240830T100559_N0511_R022_T32TPS_20240830T134009.SAFE",
       bands=["red", "nir"],
       bbox=(11.0, 46.5, 11.2, 46.7),
   )
   ndvi = scene.ndvi(red="red", nir="nir")

Landsat is the same call:

.. code-block:: python

   scene = eeo.load_landsat(
       path="LC09_L2SP_193028_20260822_20260823_02_T1",
       bands=["red", "nir08"],
       bbox=(11.0, 46.5, 11.2, 46.7),
   )
   ndvi = scene.ndvi(red="red", nir="nir08")

.. important::

   Both return the product's **stored integers**, not reflectance — the same
   values QGIS shows when you open the file. See
   `What the values mean`_ before computing an index from them.

-----

Where the products come from
----------------------------

Sentinel-2
   The `Copernicus Data Space Ecosystem <https://dataspace.copernicus.eu/>`_.
   Download the **Level-2A** product; it arrives as a ``.zip`` containing one
   ``.SAFE`` directory.

Landsat
   `USGS EarthExplorer <https://earthexplorer.usgs.gov/>`_. Order the
   **Collection 2 Level-2** product; it arrives as a ``.tar``.

Both are free and both require an account.

-----

What a path can be
------------------

Either loader accepts the product as it arrived, unpacked or not:

.. code-block:: python

   # the product itself
   eeo.load_sentinel2(path="S2B_MSIL2A_….SAFE", bands=["red"])
   # the folder it was unpacked into
   eeo.load_sentinel2(path="downloads/", bands=["red"])
   # still zipped, as downloaded
   eeo.load_sentinel2(path="S2B_MSIL2A_….zip", bands=["red"])
   # the manifest itself
   eeo.load_sentinel2(path="S2B_MSIL2A_….SAFE/MTD_MSIL2A.xml", bands=["red"])

   # still tarred, as downloaded
   eeo.load_landsat(path="LC09_L2SP_….tar", bands=["red"])

Reading straight from the archive means you don't have to unpack the
archive yourself, which is worth it for a scene you will read once.
A **compressed** archive is refused rather than read slowly:

.. code-block:: text

   ValidationError: 'LC09_….tar.gz' is a compressed archive, which cannot be
   read in place: it holds no index, so reading one band would decompress every
   byte before it, once per band. Extract it first — `tar -xf LC09_….tar.gz` —
   and pass the 'LC09_….tar' directory.

Pointing at a folder is only unambiguous while it holds **one** product. A
folder with several is refused by name rather than loading whichever sorts
first, which would silently analyse the wrong date.

-----

Bands are named, not numbered
-------------------------------

``["red", "nir08"]`` means the same thing on every sensor. Band *numbers* do
not: red is band 4 on Landsat 8 and 9, and band 3 on Landsat 7. Naming a band
is what lets one script run over both.

.. code-block:: python

   # Landsat 9: red is SR_B4, nir08 is SR_B5
   eeo.load_landsat(path="LC09_L2SP_….tar", bands=["red", "nir08"])
   # Landsat 7: red is SR_B3, nir08 is SR_B4
   eeo.load_landsat(path="LE07_L2SP_….tar", bands=["red", "nir08"])

Anything in the first column can be passed to ``bands=``, and so can any native
id on the same row. A dash means the sensor has no such band, and asking for it
fails with a message listing what the product does hold.

.. list-table::
   :header-rows: 1
   :widths: 22 18 30 30

   * - Pass this
     - Sentinel-2
     - Landsat 4, 5, 7
     - Landsat 8, 9
   * - ``coastal``
     - ``B01``
     - —
     - ``SR_B1``
   * - ``blue``
     - ``B02``
     - ``SR_B1``
     - ``SR_B2``
   * - ``green``
     - ``B03``
     - ``SR_B2``
     - ``SR_B3``
   * - ``red``
     - ``B04``
     - ``SR_B3``
     - ``SR_B4``
   * - ``rededge1``
     - ``B05``
     - —
     - —
   * - ``rededge2``
     - ``B06``
     - —
     - —
   * - ``rededge3``
     - ``B07``
     - —
     - —
   * - ``nir``
     - ``B08``
     - —
     - —
   * - ``nir08``
     - ``B8A``
     - ``SR_B4``
     - ``SR_B5``
   * - ``nir09``
     - ``B09``
     - —
     - —
   * - ``swir16``
     - ``B11``
     - ``SR_B5``
     - ``SR_B6``
   * - ``swir22``
     - ``B12``
     - ``SR_B7``
     - ``SR_B7``
   * - ``lwir11``
     - —
     - —
     - ``ST_B10``
   * - ``lwir``
     - —
     - ``ST_B6``
     - —

The quality and ancillary layers are requested the same way:

.. list-table::
   :header-rows: 1
   :widths: 22 18 30 30

   * - Pass this
     - Sentinel-2
     - Landsat 4, 5, 7
     - Landsat 8, 9
   * - ``scl``
     - ``SCL``
     - —
     - —
   * - ``aot``
     - ``AOT``
     - —
     - —
   * - ``wvp``
     - ``WVP``
     - —
     - —
   * - ``tci``
     - ``TCI``
     - —
     - —
   * - ``qa_pixel``
     - —
     - ``QA_PIXEL``
     - ``QA_PIXEL``
   * - ``qa_radsat``
     - —
     - ``QA_RADSAT``
     - ``QA_RADSAT``
   * - ``qa_aerosol``
     - —
     - —
     - ``SR_QA_AEROSOL``
   * - ``atmos_opacity``
     - —
     - ``SR_ATMOS_OPACITY``
     - —
   * - ``cloud_qa``
     - —
     - ``SR_CLOUD_QA``
     - —

Note ``nir`` and ``nir08``: Sentinel-2 carries both, a wide band and a narrow
one, while Landsat's near-infrared is the narrow kind. So ``"nir08"`` is the
name that works on every mission, and ``"nir"`` is Sentinel-2 only.

The native ids work with or without their prefix, so ``"SR_B4"`` and ``"B4"``
both reach the same band — useful when following a formula written in those
terms. See :doc:`band_names` for the full vocabulary.

**There is no default band list.** A full Sentinel-2 Level-2A product holds
several gigabytes of imagery: one 10 m band alone is 10,980 × 10,980 pixels at
two bytes each, about 241 MB, and a Landsat scene is roughly 7,700 × 7,900,
about 120 MB per band. A default of "everything" would turn a two-band NDVI
into a load that contains many unneeded items in memory , so the set to read is always an
explicit choice — and ``bbox`` is what keeps it bounded.

-----

Resolution
----------

Sentinel-2 writes its bands at 10, 20, and 60 m. ``load_sentinel2`` defaults to
the finest resolution among the bands you actually asked for, so a load never
upsamples everything to a resolution only one band was sensed at:

.. code-block:: python

   # both 10 m, so 10 m
   eeo.load_sentinel2(path="S2B_MSIL2A_….SAFE", bands=["red", "nir"])
   # both 20 m, so 20 m
   eeo.load_sentinel2(path="S2B_MSIL2A_….SAFE", bands=["swir16", "swir22"])
   # mixed, so the finest of them: 10 m
   eeo.load_sentinel2(path="S2B_MSIL2A_….SAFE", bands=["red", "swir16"])
   # or say which you want
   eeo.load_sentinel2(
       path="S2B_MSIL2A_….SAFE", bands=["red", "swir16"], resolution=20
   )
   # nearest is used for "scl" whatever this says, since it holds classes
   eeo.load_sentinel2(
       path="S2B_MSIL2A_….SAFE",
       bands=["red", "scl"],
       resolution=10,
       resampling="bilinear",
   )

A band the product does not hold at the chosen resolution is read at the finest
it does hold and warped onto the grid.

``load_landsat`` has **no** ``resolution`` or ``resampling`` argument, because
nothing is ever resampled: every Collection 2 Level-2 band, thermal and quality
included, is delivered on one 30 m grid.

-----

Which levels are read
---------------------

Surface reflectance only:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Mission
     - Read
     - Refused
   * - Sentinel-2
     - Level-2A
     - Level-1C
   * - Landsat
     - ``L2SP``, ``L2SR``
     - ``L1TP``, ``L1GT``, ``L1GS``

The level is read from the metadata, not from
the filename, so renaming a product cannot talk the loader into the wrong one.

-----

.. _what-the-values-mean:

What the values mean
--------------------

**A load returns the product's stored integers.** Not reflectance — the digital
numbers the file holds, which are the same values QGIS shows. This is true of
:func:`~eeo.load_sentinel2`, :func:`~eeo.load_landsat`, and
:meth:`~eeo.io.STACItem.load` alike: the assets declare a scale of ``1.0`` and
an offset of ``0.0`` to GDAL.

Converting is one line, and each mission encodes differently:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Product
     - Conversion
   * - Sentinel-2 L2A, baseline 04.00 and later
     - ``reflectance = DN * 1e-4 - 0.1``
   * - Sentinel-2 L2A, before baseline 04.00
     - ``reflectance = DN * 1e-4``
   * - Landsat C2 L2 surface reflectance
     - ``reflectance = DN * 2.75e-5 - 0.2``
   * - Landsat C2 L2 surface temperature
     - ``kelvin = DN * 0.00341802 + 149.0``

The coefficients for the product in hand travel on the result, so you do not
have to hardcode them:

.. code-block:: python

   scene = eeo.load_sentinel2(path="S2B_MSIL2A_….SAFE", bands=["red", "nir"])
   reflectance = scene * 1e-4 - 0.1        # baseline 04.00 and later

   scene.attrs["processing_baseline"]      # '05.XX'
   scene.attrs["band_offsets"]             # {'B04': -1000.0, ...}

.. warning::

   **An index computed over digital numbers is biased.** A *multiplicative*
   scale cancels in a normalised difference — the same factor appears in the
   numerator and the denominator — so it is easy to assume the offset cancels
   too. It does not: an additive offset survives, and pulls the result toward
   zero.

   Measured on Sentinel-2 farmland in the Po valley, the same NDVI came out at
   **0.438** over raw digital numbers and **0.668** over reflectance — a median
   error of +0.231, and up to 0.593 on individual pixels. That is the
   difference between "sparse vegetation" and "dense canopy".

   Convert before computing an index if the absolute value matters.

Sentinel-2's offset applies to products processed with baseline 04.00 (January
2022) or later; earlier products have none. A time series spanning that date
mixes two encodings 1,000 DN apart, and ``attrs["processing_baseline"]`` is
what tells them apart.

-----

What the result carries
-----------------------

Beyond the pixels, a load records where they came from:

.. code-block:: python

   scene = eeo.load_landsat(path="LC09_L2SP_….tar", bands=["red", "nir08"])

   scene.band_names            # ['red', 'nir08']
   scene.timestamp             # acquisition time, UTC
   scene.attrs["product"]      # 'LC09_L2SP_193028_20260822_20260823_02_T1'
   scene.attrs["mission"]      # 'Landsat 9'
   scene.attrs["level"]        # 'L2SP'

Sentinel-2 additionally records ``tile`` and ``processing_baseline``; Landsat
records ``wrs_path``, ``wrs_row``, and the scaling coefficients above.

.. note::

   **Landsat 7 after 31 May 2003 is missing about 22% of each scene.** The scan
   line corrector failed, leaving wedge-shaped gaps that widen toward the scene
   edges. Those pixels carry the fill value, not a measurement, so they are
   nodata and must be excluded from statistics rather than read as zero
   reflectance. See :doc:`nodata_and_dtype`.
