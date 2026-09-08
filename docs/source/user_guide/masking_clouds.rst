Masking Clouds
==============

A satellite sees the top of the atmosphere, not the ground. On a given day a
good part of any scene may be cloud, the shadow a cloud throws, or thin haze —
and none of those pixels tell you anything about the land underneath.

If you leave them in, they do not announce themselves. They quietly change
your answer. On a real Landsat 9 scene, the average NDVI (a common measure of
vegetation) over the cloudy pixels is **0.126**, while over the clear pixels it
is **0.302**. Nothing errors, nothing looks broken — the vegetation of that
scene simply appears less than half as healthy as it is.

Removing those pixels is what this page is about.

.. seealso::

   :doc:`loading_downloaded_scenes` and :doc:`loading_satellite_data` for
   getting a scene in the first place, :doc:`nodata_and_dtype` for what
   "removed" actually means to the rest of the library, and
   :doc:`spectral_indices` for the indices that then benefit.

-----

The short version
-----------------

Load the scene with its quality band, then call one method:

.. code-block:: python

   import eeo

   scene = eeo.load_sentinel2("S2A_....SAFE.zip", bands=["red", "nir", "scl"])
   clear = scene.mask_clouds()

   ndvi = clear.ndvi("red", nir="nir")     # cloudy pixels excluded

The same two lines work on Landsat — only the band names change:

.. code-block:: python

   scene = eeo.load_landsat("LC09_....tar", bands=["red", "nir08", "qa_pixel"])
   clear = scene.mask_clouds()

That is the whole feature. The rest of this page explains what it decided on
your behalf, and how to decide differently.

-----

What "masking" actually does
----------------------------

:func:`~eeo.mask_clouds` does not delete pixels or change the shape of your
raster. It marks the cloudy ones as **nodata** — the library's word for "there
is no measurement here" — in *every* band.

Everything downstream then leaves them alone automatically. An average skips
them, a percentile stretch ignores them, an index returns "not a number" for
them. You do not have to remember they exist.

.. note::

   The quality band gets masked along with the rest, so a masked scene cannot
   be used as its own mask a second time. Keep the original if you need it
   again.

-----

Where the answer comes from
---------------------------

You are not asking Easy-EO to find the clouds. Both missions already did that
and shipped the answer inside the product, in an extra layer beside the colour
bands. Easy-EO reads that layer.

The two missions encode it completely differently:

.. list-table::
   :header-rows: 1
   :widths: 18 20 62

   * - Mission
     - Layer
     - How it stores the answer
   * - Sentinel-2
     - ``SCL``
     - One **class number** per pixel: a single label from a list of twelve,
       such as "vegetation" or "cloud, high probability".
   * - Landsat
     - ``QA_PIXEL``
     - Several **independent flags packed into the bits** of one number, so a
       single pixel can be flagged cloud *and* water *and* have a confidence
       level, all at once.

You never have to care which. ``mask_clouds`` looks at the quality band's name
and picks the right reader, so the same code works either way — from a
downloaded file or from an online catalog.

Sentinel-2: the twelve classes
------------------------------

Every pixel of the ``SCL`` band carries exactly one of these numbers. Names are
ESA's own, from the `Copernicus Scene Classification table
<https://sentiwiki.copernicus.eu/web/s2-processing>`_:

.. list-table::
   :header-rows: 1
   :widths: 6 30 50 14

   * - #
     - Name
     - Meaning
     - Masked by default
   * - 0
     - ``NO_DATA``
     - Outside the image, or no measurement at all.
     - **yes**
   * - 1
     - ``SATURATED_OR_DEFECTIVE``
     - The sensor overloaded, or the detector was faulty.
     - **yes**
   * - 2
     - ``CAST_SHADOWS``
     - Ground in the shadow of a mountain, not a cloud.
     - no
   * - 3
     - ``CLOUD_SHADOWS``
     - Ground in the shadow of a cloud.
     - **yes**
   * - 4
     - ``VEGETATION``
     - Plants.
     - no
   * - 5
     - ``NOT_VEGETATED``
     - Anything that is not plants and not water — soil, rock, roads,
       buildings. Defined by what it is *not*.
     - no
   * - 6
     - ``WATER``
     - Water.
     - no
   * - 7
     - ``UNCLASSIFIED``
     - The classifier could not decide. Not the same as "clear".
     - no
   * - 8
     - ``CLOUD_MEDIUM_PROBABILITY``
     - Probably cloud.
     - **yes**
   * - 9
     - ``CLOUD_HIGH_PROBABILITY``
     - Almost certainly cloud.
     - **yes**
   * - 10
     - ``THIN_CIRRUS``
     - High, thin ice cloud. You can partly see through it, so the pixel holds
       a real measurement — of a real surface, seen through interference.
     - **yes**
   * - 11
     - ``SNOW_ICE``
     - Snow or ice on the ground.
     - no

.. note::

   ESA has renamed two of these over the years without changing their numbers.
   Class 2 used to be ``DARK_FEATURES``, and class 5 is still widely called
   "bare soil", which is narrower than what it really holds. Easy-EO accepts
   the old names too, so older scripts and tutorials keep working:
   ``classes=["bare_soil"]`` and ``classes=["not_vegetated"]`` mean the same
   thing.

Landsat: the bits
-----------------

``QA_PIXEL`` packs eight yes/no flags and four confidence levels into one
16-bit number. The tables below are transcribed from the USGS Collection 2
Level-2 Science Product Guides — `LSDS-1619 Table 6-2
<https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/LSDS-1619_Landsat8-9-Collection2-Level2-Science-Product-Guide-v6.pdf>`_
for Landsat 8–9 and `LSDS-1618 Table 5-5
<https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/LSDS-1618_Landsat-4-7_C2-L2-ScienceProductGuide-v4.pdf>`_
for Landsat 4–7. Those links pin the versions these tables were read from; USGS
publishes newer revisions on the `Landsat Collection 2 site
<https://www.usgs.gov/landsat-missions/landsat-collection-2>`_.

.. list-table:: Single-bit flags
   :header-rows: 1
   :widths: 8 26 52 14

   * - Bit
     - Flag
     - Meaning
     - Masked by default
   * - 0
     - ``FILL``
     - Outside the imaged area — the empty corners of the scene.
     - **yes**
   * - 1
     - ``DILATED_CLOUD``
     - A safety buffer grown around each detected cloud, catching the fuzzy
       edge the cloud flag itself misses.
     - **yes**
   * - 2
     - ``CIRRUS``
     - Thin high cloud. **Landsat 8 and 9 only** — see below.
     - **yes**
   * - 3
     - ``CLOUD``
     - Cloud.
     - **yes**
   * - 4
     - ``CLOUD_SHADOW``
     - The shadow a cloud casts.
     - **yes**
   * - 5
     - ``SNOW``
     - Snow or ice.
     - no
   * - 6
     - ``CLEAR``
     - Neither cloud nor dilated cloud. Derived from bits 1 and 3, so it adds
       nothing new.
     - no
   * - 7
     - ``WATER``
     - Water rather than land.
     - no

.. list-table:: Confidence levels (two bits each)
   :header-rows: 1
   :widths: 12 30 58

   * - Bits
     - Field
     - Values
   * - 8–9
     - Cloud confidence
     - ``0`` none, ``1`` low, ``2`` **medium**, ``3`` high
   * - 10–11
     - Cloud shadow confidence
     - ``0`` none, ``1`` low, ``2`` *reserved*, ``3`` high
   * - 12–13
     - Snow/ice confidence
     - ``0`` none, ``1`` low, ``2`` *reserved*, ``3`` high
   * - 14–15
     - Cirrus confidence
     - ``0`` none, ``1`` low, ``2`` *reserved*, ``3`` high

Two details in that table matter more than they look.

**Only cloud confidence has a "medium".** For the other three, the value 2 is
reserved and never appears, so treating them as a three-step scale would be
reading a level that does not exist.

**The same bit is not the same flag on every Landsat.** Landsat 4, 5 and 7
carry no cirrus-detecting instrument, so bits 2 and 14–15 are unused on those
satellites. Easy-EO therefore needs to know which satellite took the scene. It
reads that from the scene itself, so normally you never think about it — but
if you build a dataset by hand you may have to say:

.. code-block:: python

   scene.mask_clouds(mission=9)

Asking for cirrus on a Landsat 7 scene raises an error rather than quietly
reporting "no cirrus anywhere", which would look like an answer and is not one.

-----

Why those defaults
------------------

The default is deliberately **cautious**: it would rather throw away some good
ground than let cloud through. Discarding a clear pixel costs you data;
keeping a cloudy one corrupts your result.

So by default Easy-EO masks cloud, the buffer around it, cloud shadow, thin
cirrus, and anything that holds no measurement at all. It keeps every genuine
surface — vegetation, soil, water, snow, and terrain shadow.

Two choices in there are worth spelling out.

**Thin cirrus is masked.** You can see the ground through it, so it is tempting
to keep. But the measurement underneath is distorted, and a distorted number is
still a wrong one.

**Medium-confidence cloud is masked** — and on Landsat, this takes some care.
The single-bit ``CLOUD`` flag is only set where the confidence is *high*. A
mask built from the flags alone therefore lets medium-confidence cloud straight
through: USGS's own documentation lists pixel value 22280 as "Mid conf cloud"
with **no cloud flag set at all**. On a real Landsat 9 scene that is 30,711
pixels a naive mask would have missed.

Easy-EO therefore also masks on the cloud *confidence* field at medium and
above. This is what USGS means when it advises using the confidence levels
rather than the clear/cloud bits, which it calls the truer measure of cloud
extent. It also keeps the two missions consistent, since Sentinel-2's
``CLOUD_MEDIUM_PROBABILITY`` was already being masked.

-----

Deciding differently
--------------------

The defaults are a judgement, not a fact, and you can disagree with them.

Keep thin cirrus on Sentinel-2, because you would rather have a hazy
measurement than none:

.. code-block:: python

   scene.mask_clouds(classes=[0, 1, 3, 8, 9])       # everything except cirrus (10)

Classes can be given as numbers, as names, or as the enumeration:

.. code-block:: python

   from eeo import SCLClass

   scene.mask_clouds(classes=["cloud_shadows", "cloud_high_probability"])
   scene.mask_clouds(classes=[SCLClass.CLOUD_HIGH_PROBABILITY])

Mask snow as well, if snow would confuse the thing you are measuring:

.. code-block:: python

   scene.mask_clouds(classes=[0, 1, 3, 8, 9, 10, 11])

On Landsat, choose flags instead — and set the confidence threshold:

.. code-block:: python

   from eeo import QAConfidence

   # Only high-confidence cloud, matching the flag bits alone.
   scene.mask_clouds(min_cloud_confidence=QAConfidence.HIGH)

   # Ignore confidence entirely and use just the flags you name.
   scene.mask_clouds(flags=["cloud", "cloud_shadow"], min_cloud_confidence=None)

   # Stricter: mask anything with even low cloud confidence.
   scene.mask_clouds(min_cloud_confidence=QAConfidence.LOW)

If your quality layer lives in a separate file, pass it in:

.. code-block:: python

   scl = eeo.load_raster("SCL.tif")
   scl.band_names = ["scl"]
   clear = scene.mask_clouds(mask=scl)

It must already be on the same pixel grid. Easy-EO will not resample it for
you, because resampling a mask is dangerous: an "averaging" method applied to
class numbers would blend cloud (9) and vegetation (4) into bare soil (5) — a
class nobody measured.

-----

How much of the scene survived?
-------------------------------

:func:`~eeo.clear_fraction` tells you what is left, which is how you decide
whether a scene is worth using at all:

.. code-block:: python

   scene.mask_clouds().clear_fraction()      # 0.6036  -> 60% usable

.. warning::

   **That number is not "how cloudy it was."** It counts every pixel with no
   measurement — including the empty corners every satellite scene has, because
   the satellite flies at an angle while the image is stored as an
   north-aligned rectangle.

   The scene above is only 63.1% data to begin with. Of the 39.6 percentage
   points it appears to "lose", just **2.7 are cloud**. The rest was never
   imaged.

   Clip to the area you actually care about first, and the number answers the
   question you meant:

   .. code-block:: python

      inside = scene.clip_raster_with_bbox(my_area)
      inside.clear_fraction()                    # 1.0000  -- no empty corners
      inside.mask_clouds().clear_fraction()      # 0.9677  -- 3% cloud

-----

How good are these masks, honestly?
-----------------------------------

Moderate. They are free, they ship with every scene, and they are good enough
for a great deal of work — but they are automated guesses, and both agencies
document where they go wrong.

**Sentinel-2.** ESA notes that `snow is most of the time identified as opaque
clouds <https://sentiwiki.copernicus.eu/web/s2-processing>`_, because snow on a
mountain is as bright as a cloud in the bands used to detect them. Cirrus
detection yields "a probability and not a certainty". And the cirrus mask is
only computed **below 3,000 m elevation** — above that, cirrus and ground
cannot be told apart, so high mountains get no cirrus detection at all.

**Landsat.** The USGS product guide (LSDS-1619 §6.1.1.1) lists three known
issues with the CFMask algorithm behind ``QA_PIXEL``:

1. It misjudges cloud when the temperature difference from the ground is too
   large or too small — a warm cloud over very cold ground may be missed, and
   leftover ice on unusually warm ground may be called cloud.
2. It struggles over **bright targets**: building tops, beaches, snow and ice,
   sand dunes, salt flats.
3. "Optically thin clouds will always be challenging to identify and have a
   chance of being omitted."

The pattern across both: **bright things get over-flagged, faint things get
missed.** If your study area is a beach, a salt flat, a city, or a snowfield,
expect the mask to be pessimistic. If you are chasing subtle change, expect
some thin cloud to slip through.

Class 7, ``UNCLASSIFIED``, is worth watching for the same reason. It means the
classifier gave up — which is not a claim that the pixel is clear.

When you need better
--------------------

Two well-regarded alternatives, and their catches:

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - Tool
     - What it gives you
     - The catch
   * - `s2cloudless
       <https://github.com/sentinel-hub/sentinel2-cloud-detector>`_
     - A machine-learning cloud detector for Sentinel-2 from Sentinel Hub,
       giving a probability rather than a label. Installable with ``pip``.
     - It needs band ``B10``, which **Level-2A products do not contain** — we
       checked a real product to be sure. You would have to download the
       Level-1C version of the scene instead.
   * - `Cloud Score+
       <https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED>`_
     - A continuous 0–1 usability score per pixel, from Google. Handles haze
       and cloud edges better than a hard label can, and can be applied to the
       Level-2A scenes you already have.
     - It lives in Google Earth Engine, so it means an Earth Engine account
       and working in that platform rather than locally.

For a great many purposes — a seasonal NDVI comparison, a flood map, a
land-cover sketch — the built-in mask is entirely adequate, and it costs you
nothing but the extra band. Reach for the alternatives when your result turns
on the pixels these masks get wrong.
