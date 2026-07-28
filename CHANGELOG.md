# Changelog

All notable changes to Easy-EO are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is pre-1.0, breaking changes may occur in minor releases and
are called out under a **Breaking** heading.

## [Unreleased]

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

[Unreleased]: https://github.com/Tommy-Burns/easy-eo/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Tommy-Burns/easy-eo/compare/v0.1.0b1...v0.2.0
[0.1.0b1]: https://github.com/Tommy-Burns/easy-eo/releases/tag/v0.1.0b1
