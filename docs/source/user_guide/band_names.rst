Naming Bands
============

A band index tells you *where* a band is; a band name tells you *what it is*.
``scene.ndvi(red=4, nir=8)`` is correct but opaque — six months later you have
to go looking for the band order again. Easy-EO lets you give bands
recognisable names once, then address them by name anywhere a 1-based index is
accepted:

.. code-block:: python

   from eeo import load_raster

   scene = load_raster("sentinel2.tif", band_names=["blue", "green", "red", "nir"])

   ndvi = scene.ndvi(red="red", nir="nir")
   scene.plot_composite(bands=["red", "green", "blue"])

Names are metadata, not a new data model: nothing about the pixels, the band
order, or the indices changes. Every index-based call keeps working exactly as
before.

.. seealso::

   :doc:`spectral_indices` for the index methods and their band arguments, and
   :doc:`visualization` for band selection in the plotting functions.

-----

Assigning names
---------------

Names come from one of three places.

**At load time.** ``load_raster`` and ``load_array`` take an optional
``band_names`` list with one entry per band:

.. code-block:: python

   scene = load_raster("sentinel2.tif", band_names=["blue", "green", "red", "nir"])
   rgb = load_array(array, crs=4326, transform=t, band_names=["red", "green", "blue"])

**From the file itself.** If the raster already carries GDAL band descriptions
— many real products do — they are read automatically, so ``load_raster``
alone is often enough. An explicit ``band_names`` argument overrides whatever
the file declares.

**After the fact.** Names are not fixed at construction. Assign the whole list
at once, or rename a single band:

.. code-block:: python

   scene.band_names = ["blue", "green", "red", "nir"]   # replaces all names
   scene.set_band_name(4, "nir")                        # renames one band
   scene.band_names = None                              # clears every name

Assigning a list validates its length against the band count. A name is
stripped of surrounding whitespace, and a blank one becomes ``None`` — the
canonical "this band is unnamed" value. Use ``None`` for bands you do not want
to name; a partially named raster is perfectly valid.

.. note::

   Names live in memory on the dataset. A file opened for reading is never
   modified — names reach a file only when you call ``save_raster``.

-----

Addressing bands by name
------------------------

Anywhere a 1-based band index is accepted, a name works too:

.. code-block:: python

   scene.get_band("nir")                          # read one band
   scene.get_maximum_pixel(band_idx="nir")        # pixel statistics
   scene.extract_value_at_coordinate((x, y), band_idx="red")

   scene.ndvi(red="red", nir="nir")               # any index band argument
   scene.evi(red="red", blue="blue", nir="nir")

   scene.plot_raster(bands=["red", "nir"])        # plotting
   scene.plot_composite(bands=["red", "green", "blue"])

Lists may mix the two forms freely — ``bands=["red", 2, "blue"]`` is fine.

Matching is **case-insensitive** and ignores surrounding whitespace, but is
otherwise exact: ``"NIR"``, ``"nir"``, and ``" Nir "`` all find a band named
``nir``, while ``"near infrared"`` does not. There is no fuzzy matching and no
sensor-alias table — Easy-EO never guesses which band you meant.

Two rules keep the lookup unambiguous:

- **A string is always a name; an integer is always an index.** The two lookup
  spaces never overlap. A band literally named ``"4"`` is reached only by the
  string ``"4"``, never by the integer ``4`` — and ``4`` always means the
  fourth band, whatever it is called.
- **An unknown or ambiguous name is an error, not a guess.** Asking for a name
  no band has raises ``ValidationError`` listing the names that do exist.
  Real products sometimes repeat a description across bands; loading such a
  file always works, but resolving the repeated name raises rather than
  silently picking the first match. Rename the duplicates, or use the index.

-----

How names survive operations
----------------------------

Band names cannot simply be copied onto every result the way ``timestamp`` and
``attrs`` are: operations change what the bands *are*. Easy-EO applies one
rule per category.

.. list-table::
   :header-rows: 1
   :widths: 22 33 45

   * - Category
     - Operations
     - What happens to the names
   * - Identity-preserving
     - scalar algebra, ``clip_raster_with_bbox``, ``clip_raster_with_vector``,
       ``resample``, ``reproject_raster``, the normalizations
     - Carried through unchanged — band *i* still means what it meant.
   * - Synthesizing
     - ``ndvi``, ``ndwi``, ``ndmi``, ``ndbi``, ``evi``, ``savi``,
       ``normalized_difference``
     - The output is a **new** measurement that maps to no input band, so it is
       unnamed unless you pass ``name=``.
   * - Concatenating
     - ``stack``, ``mosaic``
     - ``stack`` concatenates its inputs' names in band order; ``mosaic``
       keeps the primary's. Override either with ``names=``.

.. code-block:: python

   scene.band_names = ["blue", "green", "red", "nir"]

   scene.add(1).band_names             # ['blue', 'green', 'red', 'nir']
   scene.resample(scale_factor=2).band_names   # ['blue', 'green', 'red', 'nir']

   scene.ndvi(red="red", nir="nir").band_names             # [None]
   scene.ndvi(red="red", nir="nir", name="ndvi").band_names  # ['ndvi']

   red.stack([green, blue]).band_names                     # ['red', 'green', 'blue']
   red.stack([green, blue], names=["r", "g", "b"]).band_names  # ['r', 'g', 'b']

Index results are deliberately **not** auto-named after the operation that
produced them. If they were, stacking five NDVI rasters from five dates would
give you five bands all called ``ndvi`` — every one of them ambiguous to
resolve. Naming stays your decision, either through ``name=`` at the call site
or through ``set_band_name`` afterwards.

-----

Saving and reloading
--------------------

``save_raster`` writes the names to the output's GDAL band descriptions, and
``load_raster`` reads them back, so names round-trip through a GeoTIFF without
a sidecar file:

.. code-block:: python

   scene.band_names = ["blue", "green", "red", "nir"]
   scene.save_raster("named.tif")

   reloaded = load_raster("named.tif")
   reloaded.band_names        # ['blue', 'green', 'red', 'nir']

Unnamed bands write no description. Output formats that cannot store band
descriptions simply drop them; the pixels are unaffected.

-----

Seeing the names
----------------

``describe()`` lists the names alongside the rest of the metadata, and labels
each per-band statistics row with its name:

.. code-block:: text

   EEORasterDataset
     source      : sentinel2.tif
     bands       : 4
     band names  : 1: blue, 2: green, 3: red, 4: nir
     ...

     statistics    : exact — full read
     band 1 (blue) : min 0   max 8191   mean 1024.5   ...
     band 4 (nir)  : min 0   max 9998   mean 3210.7   ...

``repr()`` shows a short list (elided past four bands), and the plotting
functions title each subplot ``Band 3 (red)`` when the band has a name.
