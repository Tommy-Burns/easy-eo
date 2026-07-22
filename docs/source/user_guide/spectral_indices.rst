Spectral Indices
================

Spectral indices combine a few bands into a single, physically meaningful
layer — vegetation, surface water, moisture, or built-up density.
Easy-EO ships six ready-made indices as **chainable, nodata-safe,
float32-output** methods on :class:`~eeo.core.core.EEORasterDataset`, built on
the same algebra primitives you can use directly.

.. seealso::

   :doc:`nodata_and_dtype` for how every index treats nodata pixels and why
   the output is always float32.

-----

Two ways to supply bands
------------------------

Every band argument accepts **either** a separate single-band
``EEORasterDataset`` **or** a 1-based ``int`` band index into the receiver, so
the same method works whether your bands are individual files or one stacked
multi-band scene. The *primary* band defaults to index ``1``, which keeps the
separate-band case a clean one-liner.

.. code-block:: python

   from eeo import load_raster

   # (a) one band per raster -- call on the primary band, pass the others
   nir = load_raster("B08.tif")
   red = load_raster("B04.tif")
   ndvi = nir.ndvi(red)

   # (b) one stacked scene -- select bands by 1-based index
   scene = load_raster("sentinel2_stack.tif")   # e.g. [B02, B03, B04, B08]
   ndvi = scene.ndvi(red=3, nir=4)

When a band is a separate raster on a different grid, it is resampled onto the
receiver's grid automatically (``auto_align=True``, the default); pass
``auto_align=False`` to require an exact match instead.

Each method returns a new single-band ``EEORasterDataset`` and is fully
chainable; pass ``return_as_ndarray=True`` to get the raw 2D array instead.

.. code-block:: python

   # chain straight into a stretch and a plot
   (
       nir.ndvi(red)
       .normalize_percentile(lower_percentile=2, upper_percentile=98)
       .plot_raster()
   )

-----

The indices and their bands
---------------------------

The table gives the formula and the band each argument maps to on two common
sensors. "Call on" names the band the method is invoked on (the primary band,
default index ``1``).

.. list-table::
   :header-rows: 1
   :widths: 10 34 18 19 19

   * - Index
     - Formula
     - Call on
     - Sentinel-2
     - Landsat 8/9 (OLI)
   * - ``ndvi``
     - ``(NIR - Red) / (NIR + Red)``
     - NIR
     - NIR B8, Red B4
     - NIR B5, Red B4
   * - ``ndwi``
     - ``(Green - NIR) / (Green + NIR)``
     - Green
     - Green B3, NIR B8
     - Green B3, NIR B5
   * - ``ndmi``
     - ``(NIR - SWIR1) / (NIR + SWIR1)``
     - NIR
     - NIR B8, SWIR1 B11
     - NIR B5, SWIR1 B6
   * - ``ndbi``
     - ``(SWIR1 - NIR) / (SWIR1 + NIR)``
     - SWIR1
     - SWIR1 B11, NIR B8
     - SWIR1 B6, NIR B5
   * - ``evi``
     - ``2.5 (NIR - Red) / (NIR + 6 Red - 7.5 Blue + 1)``
     - NIR
     - NIR B8, Red B4, Blue B2
     - NIR B5, Red B4, Blue B2
   * - ``savi``
     - ``(1 + L)(NIR - Red) / (NIR + Red + L)``
     - NIR
     - NIR B8, Red B4
     - NIR B5, Red B4

.. note::

   ``ndwi`` here is McFeeters' **water** NDWI ``(Green - NIR)``. The moisture
   variant ``(NIR - SWIR1)`` is exposed separately as :meth:`ndmi`.

   ``evi`` expects surface reflectance scaled to roughly ``[0, 1]``. If your
   bands are integer DN or reflectance scaled by 10000, rescale first, e.g.
   ``scene.divide(10000).evi(red=3, blue=1, nir=4)``.

   ``savi`` takes a soil-brightness factor ``soil_factor`` (``L``): ``0`` for
   dense cover (then SAVI equals NDVI), ``1`` for very sparse cover, ``0.5``
   (the default) for intermediate cover. Sentinel-2's SWIR1 (B11) is 20 m;
   auto-alignment resamples it onto a 10 m NIR grid for ``ndmi`` / ``ndbi``.

-----

Nodata and dtype
----------------

Every index follows the library :doc:`nodata_and_dtype`:

- Output is always **float32**.
- Nodata is **contagious**: a pixel that is nodata in *any* input band is
  ``NaN`` in the output, and the result's ``nodata`` is ``NaN`` when any input
  declares nodata (otherwise ``None``).
- A zero denominator (e.g. ``NIR + Red == 0``) is guarded to ``0`` rather than
  producing ``inf``/``NaN``.

-----

Comparing the indices
---------------------

The panel below runs all six indices over the same small synthetic scene — a
vegetation patch (top-left), open water (top-right), and a built-up area
(bottom) — so you can see how each responds to the same ground cover.

.. plot::
   :caption: The six built-in indices computed over one synthetic scene.

   import numpy as np
   import matplotlib.pyplot as plt
   from affine import Affine
   from eeo import load_array

   # A 60x60 scene split into vegetation, water, and built-up regions, with
   # per-region reflectance for Blue, Green, Red, NIR, and SWIR1.
   h = w = 60
   region = np.zeros((h, w), dtype=int)
   region[: h // 2, w // 2 :] = 1          # water   (top-right)
   region[h // 2 :, :] = 2                  # built-up (bottom)

   # reflectance [blue, green, red, nir, swir1] per region
   sig = {
       0: [0.04, 0.07, 0.05, 0.45, 0.20],   # vegetation: high NIR, low red
       1: [0.06, 0.09, 0.05, 0.03, 0.02],   # water: very low NIR/SWIR
       2: [0.18, 0.20, 0.24, 0.28, 0.34],   # built-up: high SWIR, flat
   }
   rng = np.random.default_rng(0)
   bands = {}
   for i, name in enumerate(["blue", "green", "red", "nir", "swir"]):
       arr = np.zeros((h, w), dtype=np.float32)
       for r, vals in sig.items():
           arr[region == r] = vals[i]
       arr += rng.normal(0, 0.01, size=arr.shape).astype(np.float32)
       bands[name] = arr

   transform = Affine.translation(0, h) * Affine.scale(1, -1)
   ds = {n: load_array(a, transform=transform, crs=4326) for n, a in bands.items()}

   panels = [
       ("NDVI", ds["nir"].ndvi(ds["red"], return_as_ndarray=True), "RdYlGn", -1, 1),
       ("NDWI", ds["green"].ndwi(ds["nir"], return_as_ndarray=True), "BrBG_r", -1, 1),
       ("NDMI", ds["nir"].ndmi(ds["swir"], return_as_ndarray=True), "BrBG", -1, 1),
       ("NDBI", ds["swir"].ndbi(ds["nir"], return_as_ndarray=True), "pink", -1, 1),
       ("EVI", ds["nir"].evi(ds["red"], ds["blue"], return_as_ndarray=True), "YlGn", -1, 1),
       ("SAVI", ds["nir"].savi(ds["red"], return_as_ndarray=True), "YlGn", -1, 1),
   ]

   fig, axes = plt.subplots(2, 3, figsize=(9, 6))
   for ax, (title, data, cmap, vmin, vmax) in zip(axes.ravel(), panels):
       im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
       ax.set_title(title)
       ax.set_xticks([])
       ax.set_yticks([])
       fig.colorbar(im, ax=ax, shrink=0.7)
   fig.suptitle("Vegetation (top-left) · Water (top-right) · Built-up (bottom)")
   fig.tight_layout()

-----

Doing it by hand
----------------

Because the indices are thin wrappers over the algebra primitives, you can
always build a custom index yourself. The normalized-difference family is one
call to :func:`~eeo.analysis.indices.normalized_difference`:

.. code-block:: python

   # NDVI, written out
   ndvi = nir.normalized_difference(red)

   # a custom ratio index
   custom = nir.subtract(red).divide(nir.add(red).add(0.5))
