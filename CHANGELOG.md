# Changelog

All notable changes to Easy-EO are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is pre-1.0, breaking changes may occur in minor releases and
are called out under a **Breaking** heading.

## [Unreleased]

### Breaking

- **Default values corrected to conventional choices.**
  - `normalize_percentile` now defaults to `lower_percentile=2`,
    `upper_percentile=98` (previously `0.0` / `1.0`). Callers relying on the
    old defaults will get different output; pass the values explicitly to
    reproduce the previous behaviour.
  - `resample` now defaults to `resampling_method="nearest"` (previously
    `"bilinear"`), matching `reproject_raster` and avoiding blending of
    categorical values and nodata edges. Pass `resampling_method="bilinear"`
    to restore the old default.
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

### Added

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

- **Statistics pixel locators work on multi-band rasters.**
  `get_maximum_pixel`, `get_minimum_pixel`, `get_mean_pixel`, and
  `get_percentile_pixel` previously crashed with `ValueError` on any raster
  with more than one band. All four now analyse the band selected by
  `band_idx` (default 1 — a single-band raster's only band) and raise
  `IndexError` for an out-of-range band. Single-band results are unchanged.
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
