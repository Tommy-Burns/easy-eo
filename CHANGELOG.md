# Changelog

All notable changes to Easy-EO are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is pre-1.0, breaking changes may occur in minor releases and
are called out under a **Breaking** heading.

## [Unreleased]

### Breaking

- **`return_as_ndarray` removed from the spectral indices and
  `normalized_difference`.** `ndvi`, `ndwi`, `ndmi`, `ndbi`, `evi`, `savi`, and
  `normalized_difference` now always return an `EEORasterDataset`. The flag made
  their return type `numpy.ndarray | EEORasterDataset`, which meant chaining off
  a result — `ds.ndvi(...).save_raster(...)`, exactly what the docs show — failed
  static type checking for anyone running mypy or pyright against the package,
  since easy-eo ships type information. Read the values off the result instead:
  `.get_band(1)` for the 2D band (what `return_as_ndarray=True` used to give the
  indices) or `.to_array()` for the `(bands, height, width)` array (what it gave
  `normalized_difference`). Runtime behaviour is otherwise unchanged.
- **Default values corrected to conventional choices.**
  - `normalize_percentile` now defaults to `lower_percentile=2`,
    `upper_percentile=98` (previously `0.0` / `1.0`). Callers relying on the
    old defaults will get different output; pass the values explicitly to
    reproduce the previous behaviour.
  - `resample` now defaults to `resampling_method="nearest"` (previously
    `"bilinear"`), matching `reproject_raster` and avoiding blending of
    categorical values and nodata edges. Pass `resampling_method="bilinear"`
    to restore the old default.
  - `plot_band_array`, `plot_raster` and `plot_composite` now default to
    `stretch=True` (previously `False`), so a 2-98 percentile contrast stretch
    is applied out of the box — the setting that renders most EO rasters best
    (and stops integer rasters such as Sentinel-2 reflectance from displaying as
    black). Pass `stretch=False` to display raw values. `pmin`/`pmax` are
    unchanged (2/98). `plot_raster_with_histogram` deliberately keeps
    `stretch=False` so its histogram reflects the raw value distribution.
- **Operations now raise the Easy-EO exception hierarchy** instead of bare
  built-ins. Most raises keep a built-in base for backward compatibility, so
  `except ValueError` (validation, CRS, alignment) and `except RuntimeError`
  (`BackendError`) keep working. The exceptions are a few conditions that
  previously raised `TypeError` and now raise a hierarchy member that is *not*
  a `TypeError`:
  - Backend guards (a rasterio-only op on a NumPy-backed dataset) now raise
    `BackendError` (a `RuntimeError`) instead of `TypeError`.
  - Invalid-type inputs — a non-array to `load_array`, an invalid `target_crs`
    type to `reproject_raster`, an invalid `vector_file` to
    `clip_raster_with_vector`, an invalid resampling method — now raise
    `ValidationError` (a `ValueError`) instead of `TypeError`.
- **Algebra operations now honour the nodata & dtype contract.** The
  arithmetic and transform ops (`add`, `subtract`, `multiply`, `divide`,
  `power`, `sqrt`, `log`, `absolute`) previously let nodata sentinels take
  part in the computation and wrote the result back in the input's dtype,
  silently truncating fractional results. They now:
  - **Mask nodata before computing.** A pixel that is nodata in *any* operand
    is nodata in the output (nodata is contagious). Floating outputs mark it
    with `NaN` and set `nodata=nan`; integer outputs keep the input's integer
    sentinel. A raster with `nodata=None` is unchanged (all pixels valid).
  - **Stop truncating fractional results.** `divide`, `sqrt`, and `log` now
    always output float32. `add`, `subtract`, `multiply`, `power`, and
    `absolute` follow NumPy type promotion with floating results narrowed to
    float32, so `uint16 + 0.5` yields float32 instead of truncating to
    `uint16`; integer-only arithmetic still stays integer.

  Chains that relied on the old integer truncation or on nodata sentinels
  flowing through arithmetic will see different output dtypes and NaN-masked
  gaps. See the "Nodata & Dtype Contract" guide for the full policy.
- **`normalized_difference` now masks nodata.** A pixel that is nodata in
  either input band is NaN in the output (with `nodata=nan`); previously the
  nodata sentinel took part in the ratio and the output carried the input's
  nodata value unchanged. Output stays float32; the zero-denominator guard
  (`ds + other == 0` → 0) is unchanged.
- **`extract_value_at_coordinate` returns `float('nan')` at nodata pixels**
  instead of the raw nodata sentinel, so a fill value sitting near real
  measurements can no longer be mistaken for one. Valid pixels are unchanged
  (returned in the band's dtype). A pixel counts as nodata when it equals the
  raster's declared nodata value or is already NaN.
- **Normalization ops now output float32 and exclude nodata.**
  `normalize_min_max`, `normalize_percentile`, and `standardize` previously
  wrote the result back in the input dtype — truncating an integer raster's
  normalized values to `0`/`1` — and computed their statistics (min/max,
  percentiles, mean/std) over every pixel including nodata sentinels. They now
  output **float32**, compute statistics over valid pixels only, and mark
  nodata pixels as NaN (`nodata=nan`); a raster with no declared nodata is
  unchanged apart from the float32 output.

### Added

- **"Loading satellite data" guide** (`docs/source/user_guide/loading_satellite_data.rst`).
  Quickstart from a STAC archive to a plotted NDVI in under ten lines, then the
  detail: what each search filter means, how to read a result, which asset keys
  the common catalogs use for each Sentinel-2 and Landsat 8/9 band, how AOI
  cropping and
  multi-asset stacking behave, and the caveats worth knowing (signed URLs
  expire, the cloud filter is the catalog's own, catalogs return reprocessed
  duplicates).
- **Search by shape (`stac_search(intersects=...)`).** Filter scenes by a real
  area of interest instead of its bounding rectangle: a GeoDataFrame, GeoSeries,
  shapely geometry, GeoJSON mapping (geometry, Feature, or FeatureCollection),
  or a path to any vector file GeoPandas can read. Anything carrying a CRS is
  reprojected to WGS 84 lon/lat automatically — a projected geometry submitted
  as-is would match nothing, silently, so one whose coordinates cannot be
  degrees is rejected with an actionable error instead. `bbox` and `intersects`
  are mutually exclusive, as the STAC spec requires. The geometry is retained on
  the result (`STACSearchResult.intersects`, `STACItem.search_intersects`), so
  `load()` crops to its bounds by default and `load(..., mask=True)` sets the
  pixels outside its outline to nodata — the asset's own nodata value where it
  declares one, otherwise NaN for floating dtypes and 0 for integer ones,
  recorded as the result's `nodata` either way so a masked area is never
  mistaken for data by a later operation.
- **STAC asset loading (`STACItem.load`).** Turns a search result into an
  `EEORasterDataset`: `results[0].load(["B04", "B08"])`. **Only the area of
  interest is read** — the assets are opened remotely and just the pixels
  covering the AOI are fetched as HTTP range requests against the
  cloud-optimized GeoTIFF, so a small AOI over a Sentinel-2 tile transfers a
  fraction of the band and nothing is downloaded whole. The AOI defaults to the
  `bbox` of the search that produced the item (exposed as
  `STACItem.search_bbox`); pass `bbox=` to override it or `crop=False` to read
  the entire scene. Several assets stack into one multi-band dataset with each
  band named after its asset, ready for `ds.ndvi(red="B04", nir="B08")`; assets
  that do not share the first one's grid — a 20 m band beside a 10 m one, or a
  different UTM zone — are resampled onto it (`resampling=`, nearest by
  default). The result carries the item's acquisition time as `timestamp` and
  its id, collection, and asset list in `attrs`.
- **STAC search (`eeo.stac_search`).** Query any STAC API — Microsoft Planetary
  Computer by default — by collection, bounding box (WGS 84 lon/lat), date or
  date range, maximum cloud cover, and result limit:
  `eeo.stac_search("sentinel-2-l2a", bbox=..., datetime="2023-06-01/2023-08-31",
  cloud_cover=20, limit=5)`. Returns a `STACSearchResult`: a sequence of
  `STACItem`s ordered oldest-first, each carrying its acquisition `timestamp`,
  `assets`, `cloud_cover`, and STAC `properties`, with the raw `pystac.Item`
  still reachable at `.item`. The result keeps the search `bbox` as its area of
  interest. Planetary Computer asset URLs are signed automatically (override
  with `sign=`). Searching is metadata-only — no pixel data is read and nothing
  is downloaded. Needs the `stac` extra.
- **Optional `stac` extra (`pip install "easy-eo[stac]"`).** Installs
  `pystac-client>=0.8,<1` and `planetary-computer>=1.0,<2`, the dependencies of
  the forthcoming STAC data access. Features behind an extra import their
  packages through a shared helper that raises the new
  `MissingDependencyError` — an `EEOError` that is also an `ImportError`, so
  `except ImportError` still catches it — whose message names the feature, the
  missing package, and the exact `pip install 'easy-eo[stac]'` command. A
  missing *transitive* dependency of an installed extra still surfaces as its
  own `ModuleNotFoundError` rather than being reported as a missing extra.
- **"Working with the xarray ecosystem" guide**
  (`docs/source/user_guide/xarray_interop.rst`). The round trip in six lines,
  then the detail: a table of what travels and where it lives on the DataArray,
  the layout `to_xarray()` produces (and why a rotated grid gets 2-D
  coordinates), which layouts `from_xarray()` accepts, why the coordinate axes
  rather than the stored affine decide the geotransform, what is rejected and
  what to do instead, the round-trip guarantees and the three normalisations
  applied on the way, the memory behaviour of each direction, and an honest
  "when to use rioxarray instead" section.
- **`eeo.from_xarray(da)`.** The reverse of `to_xarray()`: wraps a georeferenced
  `xarray.DataArray` as an `EEORasterDataset`, reading the CRS, geotransform, and
  nodata value from rioxarray's `.rio` accessor, plus band names from
  `long_name`, a `timestamp` from a scalar `time` coordinate, and the remaining
  `attrs`. Dimensions may be `(band, y, x)` in any order or `(y, x)` for a single
  band, spare length-1 dimensions are collapsed, and the spatial dimensions are
  whichever rioxarray identifies (so `da.rio.set_spatial_dims()` is honoured).
  The geotransform is taken from the coordinate axes where they can give it, so a
  sliced, sorted, or reversed DataArray is placed where its coordinates actually
  are rather than where its stored geotransform used to be. Values and dtype are
  passed through untouched, the nodata value is recorded as a plain Python scalar
  (rioxarray reports it in the array's own dtype), and the array is wrapped
  without an extra copy.
  Raises `ValidationError` for a Dataset, an unidentifiable spatial dimension, an
  unevenly spaced axis, more than one band-like dimension, or a `time` dimension
  longer than one step (a time series, not a band stack). Needs the `xarray`
  extra.
- **`EEORasterDataset.to_xarray()`.** Converts a dataset into a georeferenced
  `xarray.DataArray` laid out exactly like one `rioxarray.open_rasterio` would
  return: dimensions `("band", "y", "x")` (always three, even for a single
  band), a 1-based `band` coordinate, pixel-centre `y`/`x` coordinates, and the
  CRS, geotransform, and nodata value readable from `da.rio.crs`,
  `da.rio.transform()`, and `da.rio.nodata` — so the result can be handed to the
  xarray ecosystem and written back with `da.rio.to_raster()`. Provenance
  travels too: band names become rioxarray's `long_name` attribute (which
  `to_raster` writes back as GDAL band descriptions), a `timestamp` becomes a
  scalar `time` coordinate in UTC, and `attrs` are copied onto the DataArray.
  Values, dtype, and nodata pixels are carried through unchanged, and the
  returned array never shares memory with the dataset. Reads the whole raster
  into memory. Needs the `xarray` extra.
- **Optional `xarray` extra (`pip install "easy-eo[xarray]"`).** Installs
  `xarray>=2024.7` and `rioxarray>=0.17,<1`, the dependencies of the
  forthcoming xarray interop (`EEORasterDataset.to_xarray()` /
  `eeo.from_xarray()`). Nothing in the base install changes: `import eeo` still
  needs neither package, and a feature that does raises
  `MissingDependencyError` naming the exact install command. Extras compose —
  `pip install "easy-eo[stac,xarray]"`.
- **Sample-data helper (`eeo.datasets.load_sample_dataset`).** Returns a
  `SampleDataset` namespace whose attributes are the individual bundled files, so
  a curated Sentinel-2 / Copernicus-DEM sample is opened by readable,
  autocompletable name — `load_raster(sd.copernicus_dem)`,
  `load_raster(sd.sentinel2_blue)`, `load_raster(sd.sentinel2_cog_stacked)` —
  never by a hard-coded string key. Each attribute is a lazy `SamplePath` (an
  `os.PathLike`): constructing the namespace touches no network, and a file is
  downloaded to `~/.cache/easy-eo` (override with `EEO_DATA_DIR`, or
  `XDG_CACHE_HOME`) and checksum-verified only when it is actually opened, so a
  fetch is instant after the first call and never returns corrupt data. Pass
  `prefetch=True` to warm the whole cache up front. Provenance travels with each
  handle: `sd.<name>.info()` and `sd.<name>.attribution` carry the required
  Copernicus attribution; `sd.<name>.path` gives the raw cached path; the
  namespace is iterable. Downloading uses only the standard library — no new
  dependency. Exposed files: `sentinel2_stacked`, `sentinel2_cog_stacked`,
  `sentinel2_blue`/`green`/`red`/`nir`, `copernicus_dem`, `copernicus_dem_cog`,
  and `boundary` (a vector, for GeoPandas).
- **Band names on `EEORasterDataset`.** Datasets now carry an optional
  per-band name list (one entry per band, `None` for an unnamed band), seeded
  from the raster's GDAL band descriptions at load time. Read or replace them
  via the settable `band_names` property, rename a single band with
  `set_band_name(band, new_name)`, or set them at load time via
  `load_raster(..., band_names=[...])` / `load_array(..., band_names=[...])`
  (an explicit list overrides the file's own descriptions). Names are held in
  memory (so a read-only file handle is never mutated) and normalized on
  assignment (whitespace stripped, blanks become `None`). Band names resolve
  case-insensitively, a string is always a name and an int always a 1-based
  index (so `"4"` never means band 4), and an unknown or ambiguous name raises
  `ValidationError` naming the available bands.
- **Bands can be addressed by name anywhere an index is accepted.**
  `get_band`, the pixel-stats ops and `extract_value_at_coordinate`
  (`band_idx=`), every spectral-index band argument, and the plotting
  functions (`bands=`, including mixed lists such as `["red", 2, "blue"]` and
  `plot_composite(bands=["red", "green", "blue"])`) all take a band name in
  place of a 1-based index. Plot subplot titles show the name beside the band
  number when the band has one.
- **Band names propagate through operations by rule, not by blanket copy.**
  Identity-preserving ops (scalar algebra, `clip_*`, `resample`,
  `reproject_raster`, the normalizations) carry their input's names onto the
  result. Index ops synthesize a new band that maps to no input band, so they
  never auto-name their output after the operation — the result is unnamed
  unless you pass `name=` (`scene.ndvi(red="red", name="ndvi_2024")`), which
  `normalized_difference` also accepts for single-band results. `stack`
  concatenates its inputs' names in band order and `mosaic` keeps the
  primary's, both overridable with `names=`. `to_rasterio()` carries names
  onto the promoted dataset. Names can always be corrected afterwards via
  `band_names` / `set_band_name`.
- **Band names round-trip through a GeoTIFF.** `save_raster` writes them to
  the output's GDAL band descriptions and `load_raster` reads them back, so
  no sidecar file is needed. `describe()` gained a `band names` row and
  labels each per-band statistics row `band 4 (red)`; `repr()` lists the
  names (elided past four bands). A new "Naming Bands" user-guide page covers
  assignment, resolution rules, the propagation table, and the round trip.
- **Spectral index library** (`eeo.analysis.indices`): six chainable,
  nodata-safe, float32-output indices bound onto `EEORasterDataset` —
  `ndvi`, `ndwi` (McFeeters water), `ndmi`, `ndbi`, `evi`, and `savi`. Each
  band argument accepts **either** a separate single-band `EEORasterDataset`
  **or** a 1-based `int` band index into the receiver, so the same method
  serves per-band rasters (`nir.ndvi(red)`) and one stacked scene
  (`scene.ndvi(red=3, nir=4)`); the primary band defaults to index 1.
  Mismatched-grid bands auto-align by default, a zero denominator is guarded
  to 0, and nodata is contagious across all input bands (output nodata is
  `NaN`). A new "Spectral Indices" user-guide page documents the formulas,
  per-sensor band conventions (Sentinel-2, Landsat 8/9), and a comparison
  figure.
- **Custom exception hierarchy** in `eeo.core.exceptions`, exported from the
  top-level package: `EEOError` (base), `ValidationError`, `CRSMismatchError`,
  `AlignmentError`, and `BackendError`. Catch `EEOError` to handle any
  Easy-EO-specific failure. Actionable error messages now state the expected
  versus received value.
- `eeo.show_versions()` — prints an environment report (Easy-EO, Python,
  rasterio, GDAL, numpy, geopandas, matplotlib, and installed extras) for bug
  reports.
- PEP 561 typing support: a `py.typed` marker and a generated
  `eeo/core/core.pyi` that exposes the dynamically-bound operation methods to
  type checkers, both shipped in the wheel.
- A `dev` optional-dependencies extra (`pip install easy-eo[dev]`) bundling
  pytest, pytest-cov, ruff, mypy, and pre-commit.
- **"Nodata & Dtype Contract" documentation** — a user guide page plus a
  normative section in `CODE_STYLE.md` defining how operations mask nodata
  (contagious; NaN for float, sentinel for int) and what dtype they return.
- **Provenance metadata on `EEORasterDataset`.** Datasets now carry an
  optional `timestamp` (acquisition time) and a free-form `attrs` tags dict,
  settable via `load_raster`/`load_array` or directly on the dataset. Both are
  preserved through every chainable operation (each operation copies them onto
  its result; `attrs` is copied, not shared), laying the groundwork for the
  planned time-series API.
- **`EEORasterDataset.describe()` and `__repr__`.** `describe()` prints a
  human-readable summary — structural metadata (source, driver, size, dtype,
  CRS, pixel size, extent, nodata, timestamp, attrs) with **no pixel reads** by
  default. `describe(stats="approx")` (or `stats=True`) adds nodata-aware
  per-band statistics from a fast decimated read (marked `~` as approximate);
  `describe(stats="exact")` reads every pixel for exact statistics. `repr(ds)`
  gives a one-line summary (`<EEORasterDataset 4×1200×1200 uint16 EPSG:32633>`).

### Changed

- **Plot functions read at display resolution.** `plot_raster`,
  `plot_band_array`, `plot_raster_with_histogram`, and `plot_composite` now
  read rasterio-backed rasters decimated to the figure's display budget
  (via `out_shape`, served from GDAL overviews when present) instead of at
  full resolution, making plotting of large scenes fast and memory-safe.
  Small rasters and NumPy-backed datasets are still read in full;
  `plot_raster_with_histogram`'s histogram is computed from the decimated
  pixels for large rasters. `plot_histogram` is unchanged (exact counts).
- **Loosened runtime dependency bounds** to library-appropriate ranges:
  `rasterio>=1.4,<2`, `geopandas>=1.1,<2`, `numpy>=1.26,<3`, `matplotlib>=3.8`
  (previously over-tight caps such as a single matplotlib minor).
- Public docstrings rewritten to a single NumPy-style template, enforced in CI
  via ruff pydocstyle rules and numpydoc validation.
- Consolidated per-module `.pyi` stubs into inline annotations plus the single
  generated `core.pyi`.

### Removed

- Per-module `.pyi` stub files, superseded by inline type annotations and the
  `py.typed` marker.

### Fixed

- **`plot_composite(stretch=True)` no longer renders integer rasters black.**
  The percentile-stretched channels (floats in ``[0, 1]``) were written back
  into the composite's integer band dtype, truncating every value below 1 to 0
  — so a true-colour composite of any integer raster (e.g. Sentinel-2
  reflectance) came out black. The composite now stays floating when stretched.
- `normalized_difference` no longer leaves `inf` where the denominator is zero
  but the numerator is not (``ds + other == 0`` with ``ds != other``); the
  zero-denominator guard now sets both the ``0/0`` and ``x/0`` cases to 0.
- `reproject_raster` now passes the raster's nodata value to the warp
  (`src_nodata`/`dst_nodata`). Previously source nodata pixels were warped as
  ordinary values and border pixels exposed by the reprojection were filled
  with 0; they are now filled with the nodata value (or 0 only when the raster
  declares no nodata).
- **Statistics pixel locators work on multi-band rasters.**
  `get_maximum_pixel`, `get_minimum_pixel`, `get_mean_pixel`, and
  `get_percentile_pixel` previously crashed with `ValueError` on any raster
  with more than one band. All four now analyse the band selected by
  `band_idx` (default 1 — a single-band raster's only band) and raise
  `IndexError` for an out-of-range band. Single-band results are unchanged.
- `reproject_raster` computed the destination grid with the source raster's
  **width passed as its height**, distorting the output resolution and shape
  for non-square rasters (square rasters were unaffected). The destination
  grid now matches rasterio's `calculate_default_transform` result.
- `to_rasterio()` no longer re-reads and copies datasets that are already
  rasterio-backed but were produced in memory by a previous operation
  (`DatasetWriter`-backed); it now returns the same dataset unchanged. The
  same needless re-promotion is fixed inside `normalized_difference` and
  `extract_value_at_coordinate`.
- **Backend detection for chained operations.** `clip_raster_with_bbox`,
  `clip_raster_with_vector`, `mosaic`, `stack`, and `reproject_raster` no
  longer reject genuinely rasterio-backed datasets produced by a previous
  operation (e.g. `ds.add(1).clip_raster_with_bbox(...)`).
- `mosaic(..., save_path=...)` via the bound method now returns `None` as
  documented when writing to disk, instead of returning the source dataset.
- `mosaic(..., auto_reproject=True)` across rasters in different CRSs now
  works; it previously raised `TypeError` and the feature never functioned.
- `resample` now surfaces an invalid `resampling_method` as `ValidationError`
  rather than masking it inside a backend failure.
- `clip_raster_with_bbox` raises a clear `ValidationError` when the bounding
  box does not intersect the raster, instead of a cryptic rasterio
  "0x0 dataset" error.
- rasterio 1.4 compatibility: `extract_value_at_coordinate` coerces the
  `index()` row/col to `int` before indexing.
- Documentation: fixed invalid multi-line chained examples (missing
  parentheses), normalized branding to Easy-EO / easy-eo / eeo, and resolved
  signature/stub/docs mismatches across the API.

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

[Unreleased]: https://github.com/Tommy-Burns/easy-eo/compare/v0.1.0b1...HEAD
[0.1.0b1]: https://github.com/Tommy-Burns/easy-eo/releases/tag/v0.1.0b1
