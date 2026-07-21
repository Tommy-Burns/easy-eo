Nodata & Dtype Contract
=======================

Real earth-observation rasters are full of gaps: cloud masks, scene edges,
sensor dropouts, and clipped areas are all marked with a **nodata** value.
Easy-EO applies one consistent contract for how operations treat those pixels
and what data type they return, so results stay trustworthy across a chain.

This page is the user-facing summary. The normative version — the one
contributors implement and test against — lives in ``CODE_STYLE.md`` under
"Nodata & Dtype Contract".

Nodata handling
---------------

**Nodata pixels are treated as absent, not as data.**
Pixel-wise operations (arithmetic, indices, normalization, standardization)
and every statistic (minimum, maximum, mean, percentile) exclude nodata
pixels from the computation. A ``-9999`` fill value never drags down a mean,
shifts a percentile stretch, or leaks into an arithmetic result.

**Nodata is contagious.**
When an operation combines two rasters, a pixel that is nodata in *either*
input is nodata in the output. This keeps gaps from silently filling with
meaningless numbers when you add, subtract, or ratio two scenes.

**Nodata is preserved and recorded.**
Output nodata pixels are written back as the output's nodata value, and that
value is stored in the result's metadata. Spatial operations that only move
pixels around — :func:`clip <eeo.preprocessing.clip.clip_raster_with_bbox>`,
:func:`resample <eeo.preprocessing.resample.resample>`,
:func:`reproject <eeo.preprocessing.reproject.reproject_raster>`,
``mosaic``, and ``stack`` — carry the nodata value through unchanged and never
blend it (resample and reproject default to nearest-neighbour so nodata edges
are not smeared into valid data).

**How nodata is represented depends on the output dtype.**

- **Floating-point outputs** use ``NaN`` as the nodata marker and set
  ``nodata = nan`` in the metadata. ``NaN`` is NumPy's native missing-value
  marker and works directly with ``numpy.nanmean``, ``numpy.nanpercentile``,
  and friends.
- **Integer outputs** cannot store ``NaN``, so they keep the input's integer
  nodata sentinel (for example ``0`` or ``-9999``) in both the array and the
  metadata.

**A raster with no nodata value has no gaps.**
If a dataset's ``nodata`` is ``None``, every pixel is valid and every
operation computes over all of them. That is expected, not an error — set a
nodata value on your source if some pixels should be ignored.

Data types
----------

**Fractional results come back as float32.**
Operations that inherently produce fractions —
:func:`divide <eeo.ops.algebra.divide>`,
:func:`normalized_difference <eeo.analysis.indices.normalized_difference>`
and the spectral indices built on it,
:func:`normalize_min_max <eeo.preprocessing.normalize.normalize_min_max>`,
:func:`normalize_percentile <eeo.preprocessing.normalize.normalize_percentile>`,
:func:`standardize <eeo.preprocessing.normalize.standardize>`,
:func:`sqrt <eeo.ops.algebra.sqrt>`, and
:func:`log <eeo.ops.algebra.log>` — return **float32**, whatever the input
dtype was. Easy-EO never silently truncates an NDVI of ``0.42`` to ``0``
because the source band was an integer.

**Exact arithmetic keeps its natural type.**
For :func:`add <eeo.ops.algebra.add>`,
:func:`subtract <eeo.ops.algebra.subtract>`,
:func:`multiply <eeo.ops.algebra.multiply>`,
:func:`power <eeo.ops.algebra.power>`, and
:func:`absolute <eeo.ops.algebra.absolute>`, the output dtype follows NumPy's
type-promotion rules, with floating results emitted as float32 rather than
float64. Integer arithmetic that stays integer (an integer raster plus an
integer) is exact and keeps its integer dtype; mixing in a float (``ds + 0.5``)
promotes to float32 rather than truncating.

**float32 is the default float.**
Rasters are large, and float64 doubles the memory for little analytical gain
in typical EO work, so Easy-EO prefers float32 unless a computation genuinely
needs the extra precision.

Every operation states its exact nodata and dtype behavior in the ``Returns``
and ``Notes`` sections of its docstring — see the :doc:`API reference
</modules/ops>` for the per-operation guarantees.
