# Changelog

All notable changes to Easy-EO are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is pre-1.0, breaking changes may occur in minor releases and
are called out under a **Breaking** heading.

## [Unreleased]

### Added

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

### Fixed

- `clip_raster_with_vector` now accepts any `os.PathLike` for `vector_file`,
  not just `str`. It previously raised `ValidationError` for a `pathlib.Path`
  or a `eeo.datasets` sample handle, so
  `ds.clip_raster_with_vector(sd.boundary)` failed.
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

### Changed

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
