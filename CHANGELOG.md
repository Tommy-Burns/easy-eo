# Changelog

All notable changes to Easy-EO are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is pre-1.0, breaking changes may occur in minor releases and
are called out under a **Breaking** heading.

## [Unreleased]

### Added

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

### Added

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

[Unreleased]: https://github.com/Tommy-Burns/easy-eo/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Tommy-Burns/easy-eo/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Tommy-Burns/easy-eo/compare/v0.1.0b1...v0.2.0
[0.1.0b1]: https://github.com/Tommy-Burns/easy-eo/releases/tag/v0.1.0b1
