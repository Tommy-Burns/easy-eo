# Changelog

All notable changes to Easy-EO are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is pre-1.0, breaking changes may occur in minor releases and
are called out under a **Breaking** heading.

## [Unreleased]

### Added

- New optional `lazy` extra (`pip install "easy-eo[lazy]"`): xarray, rioxarray
  and `dask[array]`, the dependencies of the dask-chunked backend being built
  in WP-17. Installing it enables nothing yet; the backend itself follows. The
  conda equivalent is `conda install -c conda-forge easy-eo xarray rioxarray
  dask-core` — `dask-core`, not `dask`, which on conda-forge is a metapackage
  that also installs `distributed` and `bokeh`.
- The floor `dask>=2024.8` is tested, not assumed: CI's minimum-versions job
  now installs the `lazy` extra at its lower bounds (with numpy 1.26,
  rasterio 1.4, xarray 2024.7 and rioxarray 0.17 on Python 3.10), and the test
  matrix and the monthly latest-dependencies run install it too.
- A lazy, dask-chunked backend: `load_raster(path, chunks=...)` opens the file
  as a `rioxarray.open_rasterio` DataArray behind the new `XarrayAdapter`
  instead of with rasterio. Opening and every metadata accessor compute
  nothing; `read()` computes only the bands and window requested, so a window
  of a scene larger than memory stays bounded; `save_raster()` computes and
  writes one chunk at a time. `chunks` takes `"auto"`, one int, or a dict over
  `"band"`/`"y"`/`"x"`, and is validated before anything is imported. Leaving
  it out keeps the rasterio backend, so no existing call changes.
- Metadata on the lazy backend matches the rasterio backend's exactly —
  including nodata reported as a float and band descriptions read back as
  names — checked on synthetic rasters and on a real Landsat 9 band
  (8081 x 7991), whose full read, windowed reads and saved file are identical
  across both backends.
- Every operation now runs on a lazy dataset without reading it. A lazy
  dataset opened from a file is promoted to the rasterio backend by reopening
  that file rather than by reading its pixels, so the promotion every
  operation performs costs nothing and the block-wise engine then streams from
  the file as it always has. Audited by calling all 33 bound operations and
  plots on a lazy dataset: 30 now compute nothing at all through dask.
- `mosaic`, `stack`, `clip_raster_with_vector`, `clip_raster_with_bbox` and
  `reproject_raster` accept a lazy dataset instead of refusing it. They still
  refuse a NumPy-backed one, whose pixels are already in memory and which
  promotion would therefore copy — that stays the caller's decision.
- Plots decimate a large lazy raster rather than reading it whole, as they
  already did for rasterio-backed ones.
- Three operations still read a lazy raster in full, each because it reads in
  full on every backend, not because of the backend: `stack` (it builds one
  in-memory array by definition), `clear_fraction` and `plot_histogram`.

### Changed

- `load_raster` now reports an unreadable file as "could not be opened as a
  raster" rather than "... as a rasterio dataset", since it may no longer be
  rasterio that opens it.

## [0.4.2] - 2026-09-13

### Added

- A block-wise execution engine (`eeo.core.blockwise`) that runs a pixel-wise
  function over a raster one window at a time and writes each result straight
  into the output, so peak memory follows the block size rather than the
  scene. `apply_blockwise` takes the operands as `BlockSource`s — a whole
  raster, a single band, or a scalar — and applies the nodata & dtype contract
  per block. It is also public: call it directly to run your own pixel-wise
  function the same way, including straight to a file with `save_path=`.
- The contract's output dtype and nodata value are resolved once for the whole
  output rather than per block. That is what makes blocking invisible: a block
  containing no nodata pixels would otherwise leave its own slice of the
  output declaring no nodata, and a Sentinel-2 tile that declares fill but is
  fully imaged would end up recording none at all — after which a mosaic
  against a partly-filled neighbour would blend fill in as if it were data.
- `save_path=` streams the result to a file instead of an in-memory raster,
  the only route whose memory stays bounded when the output is also larger
  than memory.
- Verified against both real products the maintainer keeps: block-wise NDVI is
  bit-for-bit identical to the eager computation on a Landsat 9 scene
  (8081 x 7991, read from its tar) and on a Sentinel-2 tile (10980 x 10980,
  read as JP2), at every block shape tried, including shapes that divide the
  scene unevenly. On the Landsat scene the NaN pixels of the result are
  exactly the union of the two bands' off-swath fill, which runs diagonally
  across every block seam. Peak RSS for that NDVI fell from 1945 MiB eager to
  923 MiB block-wise, and to 685 MiB streaming to disk — the remainder there
  being GDAL's own block cache, which defaults to 5% of RAM. Wall time on the
  Sentinel-2 tile went from 23.1 s to 25.4 s, so the memory comes at about a
  10% cost.

### Changed

- `standardize`, `normalize_percentile`, and the pixel statistics
  (`get_maximum_pixel`, `get_minimum_pixel`, `get_mean_pixel`,
  `get_percentile_pixel`) now stream. Each needs a statistic over every pixel
  before it can rescale or locate anything, so each makes two bounded passes —
  one streaming reduction to measure, one streaming pass to apply or to find
  the pixel — instead of holding the band. The new reductions live in
  `eeo.core.streaming`.
- `get_percentile_pixel` and `normalize_percentile` measure integer rasters
  exactly, from a streaming histogram with one counter per distinct value — no
  binning, so the thresholds equal `numpy.percentile` rather than
  approximating it. That covers every raw Sentinel-2 and Landsat band, and so
  every raster large enough for the memory to matter. **Floating-point rasters
  still read the band**, and say so: an exact percentile of float data is not a
  running accumulation like a minimum or a mean, and has no finite set of
  values to count, so it cannot be had in bounded memory. Returning an
  approximation without saying so would have been worse.
- `extract_value_at_coordinate` reads a 1x1 window instead of the whole band.
  Sampling one point in a 10 m Sentinel-2 band no longer costs 241 MB.
- Ties in the pixel statistics are broken by position rather than by block, so
  the pixel reported is the first in row-major order — what `numpy.nanargmin`
  would return — no matter how the raster was cut into blocks.
- The normalizers now do their intermediate arithmetic in float64 whether or
  not the raster declares nodata. Previously `mask_nodata` promoted to float64
  only when substituting NaN for a declared sentinel, so a float32 raster was
  rescaled in float32 or float64 depending on nothing but that. Output is
  float32 either way.
- Raster algebra (`add`, `subtract`, `multiply`, `divide`, `power`, `sqrt`,
  `log`, `absolute`), every spectral index (`normalized_difference`, `ndvi`,
  `ndwi`, `ndmi`, `ndbi`, `evi`, `savi`), and `normalize_min_max` now stream
  block-wise instead of reading whole rasters into memory. Nothing to enable
  and no API change: a raster under the block budget is simply one block, so
  small rasters behave exactly as before.
- `normalize_min_max` now reads every pixel twice: one streaming pass finds the
  data range, a second rescales against it. Memory stays bounded by the block
  in both, where before the whole raster was resident for both.
- The engine is the single place the nodata & dtype contract is applied for
  these ops, replacing three near-identical copies of "compute, mask, write to
  a MemoryFile" in `eeo/ops/algebra.py`, `eeo/analysis/indices.py` and
  `eeo/preprocessing/normalize.py`.
  The whole-array helper they shared, `eeo.common.apply_nodata_contract`, had
  no callers left and is removed; the engine uses `resolve_output_nodata` and
  `apply_nodata_mask` directly.
- Op docstrings no longer carry the "reads the full array into memory" note,
  which is no longer true of them; the guarantee is stated once in the
  operations guide instead.

### Fixed

- **Chained operations on a full scene slowed from about a second per step to
  minutes.** Every in-memory raster the library produced — each op result,
  and every scene loader's output — was returned as an open writer, which
  leaves its freshly written blocks *dirty* in GDAL's block cache. Each op in a
  chain added another raster's worth, and once that outgrew the cache (5% of
  RAM by default, so about 400 MB on an 8 GB laptop) GDAL had to flush dirty
  blocks one band at a time, which for a pixel-interleaved GeoTIFF is
  catastrophically slow. On a real 6-band, 5490x5490 Sentinel-2 stack with
  default settings, `ds.add(5).add(5).multiply(2)` took 1.0 s, 147 s and
  465 s per step on 0.4.1. It now takes about 1.3 s each: every in-memory
  raster goes through `RasterioAdapter.write_in_memory` or
  `RasterioAdapter.from_memory_file`, which close the writer and reopen the
  result read-only. That costs one flush per op — about 0.6 s on that stack
  for the first step of a chain. The opt-in real-scene test suite, which
  chains ops over both downloaded products, went from 372 s and a 6.85 GB peak
  to 181 s and 4.44 GB.
- This predates the block-wise work: it was measured on `main` as released in
  0.4.1, and was found only because a real-scene sweep of every method ran a
  chain on a scene large enough to outgrow the cache. The same change moves
  eight operations off `rasterio.io.MemoryFile` and onto the adapter, where
  backend-specific code is meant to live.
- `get_maximum_pixel` returned the **minimum** pixel of any unsigned band
  containing a zero. The maximum was found by negating the band and taking a
  minimum, and negating an unsigned integer wraps rather than changing sign —
  `0` wraps to `0`, the smallest possible score, so a zero always won. Found
  while giving the streamed version differential tests over several dtypes;
  the existing fixtures all started at 1000, so nothing had caught it. Scoring
  now happens in float64.
- `normalize_percentile` documented a `ValueError` for
  `lower_percentile >= upper_percentile` that NumPy never raised: an inverted
  range silently produced an inverted stretch, and an empty one divided by
  zero. Both are now refused with a `ValidationError`, as are percentiles
  outside `[0, 100]` — which NumPy did catch, and which the histogram path
  would otherwise have stopped catching.
- A documentation error introduced with the engine: the output driver was
  justified by "JP2 cannot be written", which is false — GDAL's `JP2OpenJPEG`
  supports creation in the build we test against, and the test suite writes JP2
  fixtures with it. Choosing the output driver rather than inheriting the
  source's is still correct, because a driver records how a raster was *read*
  and need not support creating one; the reasoning is now stated that way.

## [0.4.1] - 2026-09-08

### Added

- Coverage for the error paths Codecov flagged on the masking work: the
  confidence-field resolver's refusals, reading a bit field out of a plain
  Python list, three ways `mask_clouds` can be left unable to decide, and the
  precedence rule that a raster's own nodata tag wins over the product
  manifest — documented since the nodata fix but never exercised, because no
  real Sentinel-2 product writes one. `eeo/preprocessing/quality.py`,
  `eeo/preprocessing/masking.py` and `eeo/io/products.py` are now fully
  covered, branches included.
- A "Masking Clouds" user guide, written to be followed by someone who is not
  a remote-sensing specialist. It opens with why it matters rather than how it
  works — on a real Landsat 9 scene, average NDVI over the cloudy pixels is
  0.126 against 0.302 over the clear ones, so leaving them in makes the
  vegetation look less than half as healthy as it is, and nothing errors. It
  then gives the full Sentinel-2 class table and both Landsat bit tables with
  plain-language meanings, marks what is masked by default and explains why,
  shows how to disagree, and covers `clear_fraction` with the warning that a
  whole-scene figure is not cloudiness.
- The guide is honest about quality: it quotes ESA that snow "is most of the
  time identified as opaque clouds" and that the cirrus mask is only computed
  below 3,000 m, and the USGS CFMask known-issue list — trouble over bright
  targets such as building tops, beaches, snow and salt flats, and thin cloud
  liable to be omitted. It points to `s2cloudless` and Cloud Score+ for work
  that needs better, with the catch on each: s2cloudless needs band `B10`,
  which Level-2A products do not contain (checked against a real product), so
  it means downloading Level-1C; Cloud Score+ applies to Level-2A but lives in
  Google Earth Engine.
- A cross-cutting test sweep for masking (`tests/test_masking_contract.py`)
  covering the properties that span the decoders, the operation and the
  loaders and so belonged to none of them: the same quality values mask
  identically whether the scene came from a downloaded product, a catalog or a
  bare array; the nodata contract holds across every integer and float dtype
  rather than the uint16 both missions happen to use; and `clear_fraction`
  reports a proportion that was constructed rather than counted off a
  six-pixel row. Each `QA_PIXEL` bit is also now read in isolation — the USGS
  value table sets several bits per value, so a decoder that read two flags
  from one bit could satisfy every documented value if the errors cancelled.
- `eeo.clear_fraction()`, reporting the share of a raster that still holds a
  measurement — the number for deciding whether a scene is worth keeping.
  After `mask_clouds()` it is the clear fraction in the usual sense; on any
  other raster it is simply how much of it is not nodata. A raster declaring
  no nodata returns `1.0`, the same reading the rest of the library takes. By
  default a pixel counts as clear only where every band holds a measurement,
  since nodata is contagious; `band=` measures one band alone.
- It counts every absent pixel, scene-edge fill included, which is worth
  knowing before reading one as cloudiness: on a real Landsat 9 scene the
  whole-scene figure falls from 63.1% to 60.4% under masking, so only 2.7 of
  the 39.6 points lost are cloud and the rest is the north-up grid's own
  corners. The same scene clipped to its centre reads 100% before masking and
  96.8% after. Clip to the area you care about first if the question is "how
  cloudy was it".
- `eeo.mask_clouds()`, a chainable operation setting cloudy pixels to nodata
  across every band, from the scene's own quality layer. One operation serves
  both missions: which decoder runs is settled by the quality band's name, so
  `ds.mask_clouds().ndvi("red", nir="nir")` reads identically whether the scene
  came from Sentinel-2 or Landsat, from a STAC catalog or a folder on disk. The
  Landsat mission is read from the dataset's own `attrs` and required otherwise,
  since the same bit is not the same flag on every mission. A quality band the
  dataset does not carry, two quality bands, `classes=` on a Landsat band or
  `flags=` on a Sentinel-2 one are each refused by name rather than guessed at.
  A separately loaded mask is accepted with `mask=`, and must be on the same
  pixel grid — resampling it is the caller's business, because doing it here
  with an interpolating method would blend class numbers into classes nobody
  measured.
- Masked pixels take the raster's declared nodata, or NaN for a float raster
  that declares none. An **integer** raster declaring no nodata is refused with
  an actionable message rather than assigned a sentinel: an integer array cannot
  hold NaN, and picking a value could delete real measurements.
- An opt-in `realdata` test marker and `--run-realdata` flag, for checking
  decoders against whole downloaded products rather than only against
  hand-built arrays. Paths come from `EEO_TEST_SENTINEL2_SCENE` and
  `EEO_TEST_LANDSAT_SCENE` and have no default, so the rule that the default
  run reads nothing outside the repository still holds. The assertions are
  invariants rather than pixel counts, so any L2A and any Collection 2 Level-2
  product will do: on a real Landsat scene every single-bit flag is checked to
  equal its own confidence field reading High, which pins all eight bit
  positions and all four two-bit field offsets at once against data the agency
  produced. Both quality decoders are covered.
- `eeo.QAPixelFlag`, `eeo.QAConfidenceField` and `eeo.QAConfidence`, unpacking
  the Landsat Collection 2 `QA_PIXEL` band, with `eeo.qa_pixel_flag()`,
  `eeo.qa_pixel_confidence()` and `eeo.qa_pixel_mask()` reading it. Bit
  assignments are transcribed from the USGS product guides (LSDS-1619 Table 6-2
  for Landsat 8-9, LSDS-1618 Table 5-5 for Landsat 4-7) and the tests decode
  every pixel value in those guides' own value-interpretation tables, so the
  bit positions, the two-bit field offsets and the per-sensor differences are
  all checked against the mission's documentation rather than against our
  reading of it. A mission number is **required**, because the same bit is not
  the same flag on every Landsat: bit 2 is cirrus on Landsat 8-9 and Unused on
  4, 5 and 7, whose sensors have no cirrus band, as are bits 14-15. Asking for
  cirrus on a Landsat 7 scene raises rather than quietly reporting a constant
  `False` as though it were a measurement; the default mask drops it instead,
  so that one default works on every mission.
- `qa_pixel_mask()` masks from cloud *confidence* (bits 8-9) at Medium and
  above by default, not only from the single-bit cloud flag. The flag is set
  where confidence is High and nowhere else, so a flags-only mask passes
  medium-confidence cloud through untouched — USGS's own table lists pixel
  value 22080 as "Mid conf cloud" with no flag set at all. This is also what
  USGS means in advising that the confidence fields, rather than the
  clear/cloud bits, are the truer measure of cloud extent, and it makes a
  Landsat scene mask no more leniently than a Sentinel-2 one, where
  `SCL_CLOUDY` already includes medium-probability cloud. Pass
  `min_cloud_confidence=None` for the flags alone, or `QAConfidence.HIGH` to
  reproduce the flag's own threshold. Only cloud confidence is thresholded:
  for cloud shadow, snow/ice and cirrus the value 2 is Reserved rather than
  Medium, so `>= High` is the only threshold above Low that exists there, and
  their flag bits already report it.
- `eeo.SCLClass`, the twelve classes of the Sentinel-2 Level-2A scene
  classification as a named enumeration, with `eeo.scl_mask()` reporting which
  pixels of an `SCL` band fall in a given set of them. Reading the band is the
  only honest way to know which pixels of a scene are a view of the ground, and
  until now a user had to carry ESA's class table around in their head and
  compare raw numbers. What counts as cloud is a judgement rather than a fact,
  so the defaults are exported as named constants — `eeo.SCL_CLOUDY` is cloud
  shadow, both cloud probabilities and thin cirrus (3, 8, 9, 10), and
  `eeo.SCL_NODATA` is the two classes that hold no measurement at all (0, 1) —
  which lets a user disagree in their own code and a published analysis state
  exactly what it masked. Medium-probability cloud is masked by default because
  excluding it leaves a ring of half-cloud around every cloud edge, and cirrus
  because a contaminated measurement is still a wrong one; the set errs towards
  discarding some clear ground rather than admitting cloud. Classes may be named
  as enum members, as their numbers, or by name (`"cloud_shadows"`). Member
  names follow the Scene Classification table Copernicus publishes, so class 2
  is `CAST_SHADOWS` and class 5 is `NOT_VEGETATED` rather than the older
  `DARK_FEATURES` and the informal "bare soil"; both older spellings still
  resolve, because that is what existing scripts and tutorials say. Decoding is
  deliberately separate from the loaders: `SCL` means the same thing whether a
  scene arrived from a STAC catalog or from a folder on disk, so interpreting it
  must not be written once per load path.

### Fixed

- The documentation build no longer fails under `-W`. The enumerations added
  for cloud masking documented each member twice — once from the class
  docstring's `Attributes` section and once from autodoc's `:undoc-members:` —
  producing 28 duplicate-description warnings, which CI treats as errors.
  `napoleon_use_ivar` renders those sections as field lists instead, which
  removes the collision and keeps every description. The default class and
  flag sets (`SCL_CLOUDY` and friends) are also now in the API reference at
  all: autodoc could not see their `#:` comments through the package
  re-export, so they were silently absent despite the guide telling people to
  use them.

- A STAC-loaded scene now records `mission` (plus `platform` and `instruments`)
  in `attrs`, the way `load_sentinel2()` and `load_landsat()` already did.
  Without it a Landsat scene from a catalog carried a `qa_pixel` band that
  `mask_clouds()` could not decode — the same bit is not the same flag on every
  Landsat, so it refused rather than guess — while the identical scene opened
  from a downloaded `.tar` masked fine. The mission is derived from the item's
  own `platform` property, matched case-insensitively because the catalogs
  disagree on case: Planetary Computer writes `Sentinel-2B` where Earth Search
  writes `sentinel-2b`. `platform` is kept verbatim beside it, since `mission`
  deliberately drops the unit letter — 2A and 2B are one mission as far as band
  numbering and quality layers go. An item naming no platform, or one this does
  not recognise, records no mission rather than a guess: naming the wrong
  mission would decode the wrong bits and produce a plausible, wrong mask.

- `load_sentinel2()` now reports the fill value the product declares, instead
  of `nodata=None`. Sentinel-2's JP2 images carry no nodata tag — unlike
  Landsat's GeoTIFFs, which is why only one mission was affected — and ESA
  states the value once in the manifest, as `Special_Values / NODATA`, in a
  file the loader already parses for the quantification value and band
  offsets. Without it, by the nodata contract's rule 5, every fill pixel
  counted as a measurement: means and percentiles included it as reflectance
  −0.1, stretches began from a fabricated floor, and `mask_clouds()` refused
  outright because no value was available for a masked pixel. The value is
  parsed rather than assumed, and a product declaring none still reports
  `nodata=None` rather than being given a sentinel it never named.

  **This changes results.** Statistics, normalizations and indices over
  Sentinel-2 scenes containing fill will differ from 0.4.0, because fill no
  longer counts. Scenes without fill are unaffected.

  The gap was a `.SAFE`-only one: Earth Search and Planetary Computer both
  serve Level-2A COGs with `nodata=0` in the file header, so the same scene
  loaded through `stac_search` already behaved correctly — the two routes
  disagreed, which is exactly what the local loaders exist not to do. The
  test fixture had been hiding it by writing a nodata tag into its JP2s that
  real products do not have; it no longer does, and the loader tests now fail
  without this fix.

## [0.4.0] - 2026-08-29

### Added

- `eeo.load_sentinel2()` and `eeo.load_landsat()`, for reading a product you
  have already downloaded rather than one in a catalog. Both return the same
  `EEORasterDataset` a STAC load does — same grid, same band names, same
  values — so a workflow does not care which route the data took; they share
  one reader with `STACItem.load` for exactly that reason. `load_landsat` is
  one function for Landsat 4, 5, 7, 8 and 9, not one per satellite: the
  mission is read from the product's own metadata and only decides which band
  table the names resolve against, so `["red", "nir08"]` is the same request
  on a sensor where red is band 4 and one where it is band 3. Bands are always
  named explicitly — a full Sentinel-2 product is several gigabytes, so a
  default of "everything" would turn a two-band NDVI into a load that does not
  fit in memory. Only surface reflectance is read: Sentinel-2 Level-2A and
  Landsat `L2SP`/`L2SR`, with Level-1C and the Level-1 products refused by
  name and told which product to download instead. The level comes from the
  metadata rather than the filename, so a renamed product cannot mislead the
  reader.
- Either loader reads a product as it was downloaded — a `.SAFE` directory or
  a Copernicus `.zip`, a Landsat directory or a USGS `.tar` — without
  unpacking it first. A compressed archive is refused with the command to
  extract it: it holds no index, so reading one band would decompress every
  byte before it, once per band, and accepting that silently would make a load
  mysteriously slow rather than honestly refuse. Pointing at a folder that
  holds more than one product is also refused, by name, rather than loading
  whichever sorts first.
- A "Loading Downloaded Scenes" guide beside the STAC one, covering where the
  products come from, what a path can be, why bands are named rather than
  numbered, resolution selection, and the supported-level boundary. It carries
  a "What the values mean" section documenting something that was true before
  and undocumented: **a load returns the product's stored integers, not
  reflectance** — the values a GIS shows, since the assets declare a scale of
  1.0 and an offset of 0.0 to GDAL and nothing in the chain decodes them. That
  matters because a multiplicative scale cancels in a normalised difference
  while an additive offset does not, so an index over digital numbers is
  biased toward zero: measured on Sentinel-2 farmland, the same NDVI came out
  at 0.438 over DN against 0.668 over reflectance. The STAC guide's quickstart
  now says which convention its NDVI is in and links across.

- A DOI badge in the README, and the Zenodo DOI in `CITATION.cff`. Both use
  the concept DOI rather than the version DOI Zenodo offers by default, so
  they track the newest release instead of pinning to v0.3.1.
- A "Citing Easy-EO" documentation page covering the software DOI, the
  separate sample-data DOI, and the Copernicus attribution that citing the
  deposit does not replace.

### Fixed

- Pointing either loader at the folder a product was *downloaded* into now
  works. A folder holding an unpacked product was already accepted, but one
  holding the `.zip` or `.tar` itself was refused as holding nothing — which
  is the state a download is in before you touch it. Resolving to a single
  archive in the folder follows the same "exactly one, or refuse by name" rule
  the multiple-products case uses, and an unpacked product beside its own
  archive still wins, being the cheaper of the two to read.
- Composed transforms no longer use affine's `*` operator, which affine 3.0
  deprecates in favour of `@`. The library builds them directly instead, so it
  raises no deprecation warning on affine 3 while still working on affine 2.4,
  which has no `@` at all. Nothing about the values changes: the coefficients
  are identical, verified against real Sentinel-2 and Landsat scenes.

## [0.3.1] - 2026-08-16

### Added

- A `dpi` argument on every plotting function that writes a file —
  `plot_band_array`, `plot_raster`, `plot_histogram`,
  `plot_raster_with_histogram` and `plot_composite`. All five hardcoded
  `dpi=300` in their `savefig` call, so `save_path` could only ever write at
  print resolution: a 15x5 figure came out near 4500x1500 pixels and well over
  10 MB, and trimming one for a README or a paper meant re-encoding it
  afterwards. The default is unchanged at 300, so existing output is
  byte-identical. Note that raising it much past 200 enlarges the canvas
  without adding image detail, since bands are read at the on-screen display
  budget rather than at `dpi`; the visualization guide has a new "Saving
  Figures" section on choosing a value.
- `sentinel2_blue_cog`, `sentinel2_green_cog`, `sentinel2_red_cog` and
  `sentinel2_nir_cog` on `load_sample_dataset()`. The four per-band
  Cloud-Optimized GeoTIFFs were already published on the `sample-data-v1`
  release but were missing from the registry, so the single-band rasters were
  the only ones without a COG counterpart a user could ask for. Their pinned
  checksums and sizes were taken from the live release assets. A test now
  asserts every plain raster in the registry has a COG sibling, so a future
  upload cannot go unreachable the same way.
- A weekly link check covering the documentation, the Markdown files, and the
  tutorial notebooks. `sphinx-build -b linkcheck` handles the docs; `lychee`
  handles `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, and
  `examples/README.md`; and `scripts/check_notebook_links.py` handles the
  notebooks. The script resolves relative links against each notebook's real
  directory and extracts the Markdown prose for `lychee` to check the web
  links — a checker pointed at raw `.ipynb` JSON reports any URL ending a line
  as broken, with the `\n` escape attached. It is scheduled rather than run on
  pull requests: link rot arrives without a commit, and an external site being
  down should not block a merge.
- README badges for CodeQL and monthly PyPI downloads. The download badge is
  served by pepy.tech rather than Shields' `pypi/dm`, which proxies pypistats
  and renders "rate limited by upstream service" on the README when that
  service throttles. Badges were deliberately not added for the link check or
  the dependency audit: both are scheduled, so a badge reports a run up to a
  week old, and both can go red for reasons outside the project — an external
  site being down, a fresh advisory in a transitive dependency — which teaches
  readers to ignore every badge in the row.
- Every GitHub Action in the workflows is pinned to a full commit SHA with the
  version in a trailing comment, in place of the floating `@v7`-style major
  tags. A major tag is mutable: whoever controls the action repository can move
  it to different code at any time, and a workflow that uses one runs whatever
  it points to today. Dependabot keeps the SHAs and their version comments
  current, and its schedule for actions moved from monthly to weekly to match —
  a pinned action no longer picks up its own patches. Security advisories do
  not wait for that schedule.
- The README badges are now a table grouped under Install, Build & quality,
  Security, and Project, rather than a single row of thirteen. The URLs moved
  to reference-style definitions below the table, so adding or changing a badge
  is a one-line edit instead of an edit inside a very long line.
- The sdist and wheel attached to a GitHub Release are now signed with sigstore,
  which uses the workflow's own OIDC identity rather than a long-lived key, and
  the `.sigstore.json` bundles are attached alongside them. The copies on PyPI
  were already signed — `pypa/gh-action-pypi-publish` attaches PEP 740
  attestations by default under Trusted Publishing — so this closes the gap for
  anyone who downloads from the Releases page instead.
- `SECURITY.md`: supported versions, private vulnerability reporting through
  GitHub's advisory flow rather than public issues, realistic response targets
  for a single maintainer, and an explicit scope. It states plainly that
  Easy-EO implements no raster parsers — opening a file hands it to rasterio
  and GDAL, which is where the real trust boundary sits — so a parsing crash
  reached through Easy-EO is normally a GDAL issue.
- An OpenSSF Scorecard workflow and its README badge. Scorecard rates the
  repository's supply-chain posture — scoped workflow permissions, pinned
  dependencies, vulnerability scanning, release provenance — and publishes the
  result, so the badge reports a rating rather than another pass/fail tick.
- `scripts/check_colab_links.py`, checking all 51 "Open in Colab" links across
  the notebooks, the READMEs, and the tutorials page. These cannot be checked
  over HTTP: Colab is a single-page app that answers 200 for any URL, including
  a notebook path that does not exist and a repository that does not exist, so
  an ordinary link checker confirms nothing about them. Each link instead has
  its embedded `<owner>/<repo>`, branch, and notebook path verified against the
  repository, and a notebook's own badge must open that notebook — a badge
  copied to a new notebook and left unedited opens the wrong tutorial.
- Security scanning in CI. CodeQL analyses the Python source on every push and
  pull request and weekly (so new queries reach unchanged code), and a
  dependency audit runs `pip-audit` against the fully resolved `uv.lock` set —
  weekly, and on pull requests that touch `pyproject.toml` or `uv.lock`. The
  audit is deliberately not gating unrelated pull requests: a new advisory
  against an unchanged dependency is time-based news, not a regression in the
  branch that happens to be open.
- Tagging a release now creates a GitHub Release, with the tag's `CHANGELOG.md`
  section as its notes and the sdist and wheel attached. A tag previously
  published to PyPI and left no Release behind, which matters beyond
  presentation: Zenodo archives on Release events rather than tag pushes, so
  the planned DOI had nothing to hang off. Release notes are extracted by
  `scripts/extract_changelog.py`, which fails when the tag has no changelog
  entry — the notes and the changelog cannot drift apart. Pre-release tags
  (like `v0.1.0b1`) are marked as such, and re-running the workflow leaves an
  existing Release untouched.
- The release workflow now installs the built wheel and sdist into clean
  environments and smoke-tests them before publishing, via
  `scripts/smoke_test_wheel.py`. Previously the artifacts were built,
  metadata-checked, and uploaded without ever being installed, so a dropped
  `py.typed`, a missing `core.pyi`, or a tag disagreeing with `__version__`
  would have reached PyPI, where a release cannot be withdrawn. The script
  checks the packaged data files, the public API, the dynamically bound
  operations, an end-to-end NDVI chain, and the error raised when an optional
  extra is absent — importing Easy-EO as a user would, and refusing to run
  against a source checkout. The sdist is covered because conda-forge builds
  from it.
- The documentation now builds in CI on every pull request, with Sphinx
  warnings treated as errors. Previously it built only on Read the Docs, after
  a merge, so a broken cross-reference or an autodoc target that no longer
  resolved was discovered on the published site. The job installs the package
  without extras, exactly as Read the Docs does, so it fails on the same things
  the published build would.

### Fixed

- The xarray link in the interop guide now points at
  `https://docs.xarray.dev/en/stable/` rather than relying on the redirect from
  the bare domain — found by the new link check.

### Changed

- `docs/requirements.txt` now pins Sphinx, `sphinx-rtd-theme`, and
  `sphinx-copybutton` exactly, and the Read the Docs build moved from Python
  3.10 to 3.12. The two builds were resolving different Sphinx *majors* — 8.1.3
  on Read the Docs' 3.10, 9.1.0 on anything newer — so a docs check could pass
  on one toolchain while the published site rendered on another. Dependabot now
  watches the file and proposes bumps monthly. This affects the documentation
  toolchain only; the library still supports Python 3.10 and up.

## [0.3.0] - 2026-08-07

### Breaking

- `plot_raster_with_histogram` no longer takes `sharey`. Every histogram panel
  now has its own y-axis, so a quiet band is not flattened by a busy one, and
  no plotting function shares axes any more. A call passing `sharey=` raises
  `TypeError`; drop the argument, which was `False` by default.

### Added

- Easy-EO is now on conda-forge: `conda install -c conda-forge easy-eo`
  installs it and its dependencies without pip. The README, the getting-started
  guide, and the installation notebook document both package managers,
  including what to do about the optional extras — conda has no extras
  mechanism, so their dependencies are installed by name
  (`conda install -c conda-forge easy-eo pystac-client planetary-computer`),
  and Easy-EO should not be mixed across package managers within one
  environment.
- `nrows` and `ncols` on `plot_band_array`, `plot_raster`, and `plot_histogram`,
  so the subplot grid can be shaped. The layout was one row per band and one
  column per dataset with no way to reflow, which put a 4-band raster in a
  4x1 strip and four single-band datasets in a 1x4 one; `ncols=2` now gives a
  2x2 block. Giving one of the two derives the other, leftover cells are
  hidden, and a grid too small for every panel raises `ValidationError` instead
  of dropping bands. (The default layout changed too — see Changed above.)
- `colorbar` and `colorbar_label` on `plot_band_array`, `plot_raster`, and
  `plot_raster_with_histogram`. `colorbar=True` draws a scale beside each
  subplot in the band's own values, so
  `ndvi.plot_raster(cmap="RdYlGn", colorbar=True)` reads in index units rather
  than leaving the colours unexplained. The label defaults to the band's name —
  an index named at creation labels its own colorbar — and `colorbar_label`
  overrides it. Both default to off, so existing figures are unchanged.
  Arrowheads mark the ends where the stretch clips data, and each subplot gets
  its own bar because bands in a grid carry unrelated ranges. `plot_composite`
  is excluded: an RGB composite has no single scalar scale to label.

### Changed

- `plot_band_array`, `plot_raster`, and `plot_histogram` now lay their subplots
  out near-square by default rather than one per row. A 4-band raster renders
  2x2 instead of a 4x1 strip, and four single-band datasets 2x2 instead of 1x4;
  2 and 3 panels stay a single row, 6 become 2x3, 9 become 3x3. Several
  datasets *and* several bands still get the semantic grid — rows are bands,
  columns are datasets — because that is what puts band *i* of one dataset
  beside band *i* of the next. **Existing multi-panel figures will change
  shape**; pass `nrows`/`ncols` to pin a layout.
- `figsize` on those three functions now defaults to `None`, meaning derived:
  the previous default for a single row of panels, and for a taller grid the
  same width with the height set to keep the cells roughly square (so a 2x2 of
  square maps is not squeezed into a 10x5 letterbox). Passing a `figsize`
  disables the derivation, and single-panel figures are unchanged.
- `MissingDependencyError` now names an install command that can actually be
  run. It previously ended every message with `pip install 'easy-eo[<extra>]'`,
  which a conda user cannot follow: conda has no extras mechanism, and brackets
  already mean key-value constraints in its match syntax, so
  `conda install "easy-eo[stac]"` does not even parse. When conda manages the
  Easy-EO install the message now gives
  `conda install -c conda-forge pystac-client planetary-computer` instead, and
  says not to pip install an extra into a conda-managed environment — that
  combination works at first and breaks on a later `conda update`, and the old
  message was what recommended it. Detection reads conda's own record of the
  `easy-eo` package, so Easy-EO pip-installed into a conda environment still
  gets the pip command; if the environment cannot be inspected, both commands
  are shown. Existing pip installs see no change.
- The visualization notebook (`examples/01_fundamentals/07_visualization.ipynb`)
  and the spectral-indices guide now teach `colorbar=True`, including how a
  band's name becomes the label. The hand-rolled Matplotlib figures elsewhere
  in the docs are unchanged: they overlay two layers or give each panel its own
  colormap, neither of which the built-in plots do, so they still demonstrate
  the escape hatch rather than a gap.
- `plot_band_array`, `plot_raster`, and `plot_raster_with_histogram` now apply
  the percentile stretch as Matplotlib display limits (`vmin`/`vmax`) instead
  of rescaling the band to `[0, 1]`. The rendered figure is unchanged —
  verified pixel-for-pixel across float, integer, outlier-heavy, and partly-NaN
  bands — but the plotted array keeps its own units, which is what lets a
  colorbar report real values. Two consequences worth noting:
  - An explicit `vmin`/`vmax` passed through `**imshow_kwargs` / `**show_kwargs`
    now takes precedence over the stretch. Previously such a value was silently
    ineffective, the data having already been rescaled to `[0, 1]`.
  - `plot_raster_with_histogram(stretch=True)` previously binned the *stretched*
    values, putting the histogram on a 0-1 axis; it now always bins the band's
    raw values while the stretch scales the image panel alone.

- Added the `Programming Language :: Python :: 3.13` classifier. CI has tested
  3.13 since it was added to the matrix, but the metadata stopped at 3.12, so
  PyPI (and the README's version badge) under-reported supported versions.
- **Python 3.14 is now tested and supported**, on Linux, macOS and Windows.
  This was not a speculative addition: conda-forge already resolves 3.14 by
  default for `conda install easy-eo`, so users were running on it before the
  project claimed it. The CI matrix now covers 3.10–3.14 (15 jobs) and the
  classifier follows. Note that on 3.14 the full test suite intermittently
  prints a few empty `Error in sys.excepthook:` blocks on stderr after the
  summary; every test passes and the exit code is 0. It is a CPython 3.14
  interpreter-finalization artifact rather than an Easy-EO one — 3.13 with an
  identical dependency set is clean — and it affects the test suite, not the
  library.
- README rewritten for positioning: a feature matrix with per-topic guide
  links replaces the bullet list, and installation now covers the optional
  extras and the conda-forge status.
- Every public path parameter is now typed `str | os.PathLike` (`load_raster`,
  `save_raster`, `from_path`, `mosaic(save_path=...)`, and the `save_path` of
  each plotting function), exported as `eeo.core.types.StrPath`. These already
  accepted path-like values at runtime but were annotated `str`, so passing a
  `pathlib.Path` was a type error for users type-checking against the shipped
  stubs.

- README now opens with a runnable hero example — STAC search, NDVI, plot —
  so the library is visible working before any prose.
- README gained a "What's next" section covering block-wise execution, the lazy
  backend, time series, conda-forge, and citable releases, plus the xarray
  interop route for work that exceeds one machine's memory today.
- README gained a gallery of six figures - composites, an index map, a DEM,
  and histograms - each rendered by an Easy-EO call on the sample dataset and
  regenerable with `scripts/build_gallery.py`.
- README gained a "Before and after" section comparing the same
  clip-to-vector-AOI → NDVI → save task in raw Rasterio/GeoPandas/NumPy and in
  Easy-EO; both versions were executed and produce byte-identical output.

### Fixed

- The README's images no longer break on PyPI. The logo and all six gallery
  figures were referenced by repo-relative paths (`.github/assets/...`), which
  GitHub resolves against the repository but PyPI cannot, so the project page
  showed alt text where the images should be. They now use absolute
  `raw.githubusercontent.com` URLs, and the six repo-relative links
  (`CONTRIBUTING.md`, the tutorial notebooks) are absolute too — those were
  silently 404ing on PyPI for the same reason. The badges were always fine,
  being absolute already.
- `clip_raster_with_vector` now accepts any `os.PathLike` for `vector_file`,
  not just `str`. It previously raised `ValidationError` for a `pathlib.Path`
  or a `eeo.datasets` sample handle, so
  `ds.clip_raster_with_vector(sd.boundary)` failed.
- The getting-started guide told users to call
  `plot_histogram(..., sharey=True)`, which has no such parameter — the value
  fell through `**hist_kwargs` into `matplotlib.pyplot.hist` and raised
  `AttributeError: Rectangle.set() got an unexpected keyword argument
  'sharey'`. The example is corrected, and `plot_histogram` does not gain the
  parameter — see the Breaking note above, which removes the last one.
- `plot_raster` and `plot_raster_with_histogram` now pass `adjust=False` to
  `rasterio.plot.show`. rasterio 1.5 extended `adjust=` to 2D arrays, min-max
  rescaling the band to `[0, 1]` before drawing, which silently voided the
  display limits set from the percentile stretch: on Python 3.12+ (where the
  lockfile resolves rasterio 1.5) the image was drawn against limits its pixels
  no longer used, and the colorbar reported a range that was not there. Scaling
  is now left entirely to Matplotlib. A caller passing `adjust=True` through
  `**show_kwargs` still gets rasterio's behaviour.
- Plotting now excludes a declared nodata value, as the nodata contract
  ("Mask before compute" in `CODE_STYLE.md`) has always required: a `-9999`
  fill must not shift a percentile stretch. Every plotting function read
  unmasked, so a sentinel counted as an ordinary value — it widened the
  stretch, dragged a colorbar's end to the sentinel, and put a spike in every
  histogram. The sentinel is now masked before the percentiles are taken, and
  those pixels render blank instead of as a colour. In `plot_composite` a pixel
  that is nodata in any channel is transparent in the composite (the contract's
  contagion rule), on the stretched floating-point path where RGBA is
  available. Float rasters are unaffected: their nodata is already NaN, which
  the percentiles ignored. `plot_histogram`'s docstring, which documented the
  old behaviour ("nodata pixels are counted as ordinary values"), is corrected.
- Plotting a band whose percentile range is empty (a constant band, or one with
  a single valid pixel) no longer paints its nodata pixels as real values. The
  rescaling path mapped such a band to all zeros, turning every NaN into a 0
  that rendered as the colormap's low end; nodata now stays blank. An
  all-nodata band, whose percentiles are NaN, likewise falls back to
  Matplotlib's autoscaling rather than being handed NaN display limits.

## [0.2.0] - 2026-07-29

### Breaking

- The spectral indices and `normalized_difference` no longer accept
  `return_as_ndarray`; they always return an `EEORasterDataset` now. Use
  `.get_band(1)` or `.to_array()` to get the raw values instead.
- `normalize_percentile` now defaults to `(2, 98)` percentiles (was
  `(0.0, 1.0)`).
- `resample` now defaults to `resampling_method="nearest"` (was `"bilinear"`).
- `plot_band_array`, `plot_raster`, and `plot_composite` now default to
  `stretch=True` (was `False`); pass `stretch=False` for the old behaviour.
- A handful of invalid-input and backend-mismatch errors now raise
  `ValidationError`/`BackendError` instead of `TypeError` (still catchable as
  `ValueError`/`RuntimeError`).
- Algebra ops (`add`, `subtract`, `multiply`, `divide`, `power`, `sqrt`,
  `log`, `absolute`) now mask nodata (contagious, NaN on float output) and no
  longer truncate fractional results to the input's integer dtype;
  `divide`/`sqrt`/`log` always output float32. See the "Nodata & Dtype
  Contract" guide.
- `normalized_difference` now masks nodata as NaN instead of letting the
  sentinel flow through the ratio.
- `extract_value_at_coordinate` returns `float('nan')` at nodata pixels
  instead of the raw sentinel.
- `normalize_min_max`, `normalize_percentile`, and `standardize` now output
  float32, compute statistics over valid pixels only, and mark nodata as NaN
  (previously computed over all pixels and truncated to the input dtype).

### Added

- Sixteen tutorial notebooks in `examples/` (up from five), each runnable
  top-to-bottom in Colab with committed outputs; indexed in
  `examples/README.md` and the new docs "Tutorials" page.
- Notebooks now run in CI via `nbmake`; the four that need network are
  auto-excluded by their own metadata.
- New docs: "Tutorials" index and "Loading satellite data" guide.
- `stac_search(intersects=...)` — filter scenes by a real area of interest
  (GeoDataFrame, geometry, GeoJSON, or vector file) instead of a bounding box.
- `STACItem.load()` — turns a search result into an `EEORasterDataset`,
  reading only the AOI via HTTP range requests against the COG.
- `eeo.stac_search()` — query a STAC API (Microsoft Planetary Computer by
  default) by collection, bbox, date range, and cloud cover.
- New optional `stac` extra (`pip install "easy-eo[stac]"`).
- New docs: "Working with the xarray ecosystem" guide.
- `eeo.from_xarray(da)` — wraps a georeferenced `xarray.DataArray` as an
  `EEORasterDataset`.
- `EEORasterDataset.to_xarray()` — converts a dataset to a georeferenced
  `xarray.DataArray` laid out like `rioxarray.open_rasterio`.
- New optional `xarray` extra (`pip install "easy-eo[xarray]"`).
- `eeo.datasets.load_sample_dataset()` — a cached, checksum-verified
  Sentinel-2 + Copernicus DEM sample bundle, opened by attribute name.
- Named bands: assign at load or via `band_names`/`set_band_name`, resolve by
  name anywhere a band index is accepted, propagate through operations by
  rule, and round-trip through a saved GeoTIFF. New "Naming Bands" guide.
- Spectral index library (`eeo.analysis.indices`): `ndvi`, `ndwi`, `ndmi`,
  `ndbi`, `evi`, `savi` — chainable, nodata-safe, float32 output. New
  "Spectral Indices" guide.
- Custom exception hierarchy (`EEOError`, `ValidationError`,
  `CRSMismatchError`, `AlignmentError`, `BackendError`), exported from the
  top level.
- `eeo.show_versions()` for bug reports.
- PEP 561 typing support (`py.typed` + generated `core.pyi`).
- New `dev` extra (`pip install easy-eo[dev]`) bundling test/lint tooling.
- New "Nodata & Dtype Contract" documentation.
- Datasets now carry an optional `timestamp` and `attrs` dict, preserved
  through every chainable operation.
- `EEORasterDataset.describe()` and a richer `__repr__`.

### Changed

- Plot functions now read rasterio-backed rasters decimated to display
  resolution instead of full resolution, making large scenes fast and
  memory-safe to plot.
- Loosened runtime dependency bounds to library-appropriate ranges
  (`rasterio`, `geopandas`, `numpy`, `matplotlib`).
- Public docstrings unified to a single NumPy-style template, enforced in CI.
- Consolidated per-module `.pyi` stubs into inline annotations plus the
  single generated `core.pyi`.

### Removed

- Per-module `.pyi` stub files, superseded by inline type annotations and
  `py.typed`.

### Fixed

- `plot_composite(stretch=True)` no longer renders integer rasters black.
- `normalized_difference` no longer leaves `inf` at some zero-denominator
  pixels.
- `reproject_raster` now passes nodata through the warp instead of filling
  exposed border pixels with 0.
- `get_maximum_pixel`, `get_minimum_pixel`, `get_mean_pixel`, and
  `get_percentile_pixel` no longer crash on multi-band rasters.
- `reproject_raster` no longer swaps width/height when computing the
  destination grid for non-square rasters.
- `to_rasterio()`, `normalized_difference`, and `extract_value_at_coordinate`
  no longer needlessly re-read/copy datasets that are already rasterio-backed.
- Chained operations (e.g. `ds.add(1).clip_raster_with_bbox(...)`) no longer
  misdetect the backend and reject a valid rasterio-backed dataset.
- `mosaic(..., save_path=...)` now returns `None` as documented.
- `mosaic(..., auto_reproject=True)` across different CRSs now works
  (previously always raised `TypeError`).
- `resample` surfaces an invalid `resampling_method` as `ValidationError`.
- `clip_raster_with_bbox` gives a clear error when the bbox doesn't
  intersect the raster.
- rasterio 1.4 compatibility fix in `extract_value_at_coordinate`.
- Docs: fixed invalid code examples, normalized branding, resolved
  signature/docs mismatches.

## [0.1.0b1] - 2025-12-24

Initial public beta release.

### Added

- `EEORasterDataset`, the chainable raster dataset, with rasterio- and
  NumPy-backed adapters behind a common backend interface.
- Loaders: `load_raster` (from a file) and `load_array` (from a NumPy array).
- Raster algebra: `add`, `subtract`, `multiply`, `divide`, `power`, `sqrt`,
  `log`, `absolute`, with operator overloading.
- Analysis: `normalized_difference` (NDVI/NDWI-family) and per-pixel
  statistics (`get_maximum_pixel`, `get_minimum_pixel`, `get_mean_pixel`,
  `get_percentile_pixel`, `extract_value_at_coordinate`).
- Preprocessing: `clip_raster_with_bbox`, `clip_raster_with_vector`,
  `resample`, `reproject_raster`, `normalize_min_max`, `normalize_percentile`,
  `standardize`.
- Merging: `mosaic` and `stack`.
- Visualization: `plot_raster`, `plot_composite`,
  `plot_raster_with_histogram`, `plot_band_array`.

[Unreleased]: https://github.com/Tommy-Burns/easy-eo/compare/v0.4.2...HEAD
[0.4.2]: https://github.com/Tommy-Burns/easy-eo/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/Tommy-Burns/easy-eo/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/Tommy-Burns/easy-eo/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/Tommy-Burns/easy-eo/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/Tommy-Burns/easy-eo/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Tommy-Burns/easy-eo/compare/v0.1.0b1...v0.2.0
[0.1.0b1]: https://github.com/Tommy-Burns/easy-eo/releases/tag/v0.1.0b1
