# Easy-EO

<p align="center">
  <img src="https://raw.githubusercontent.com/Tommy-Burns/easy-eo/main/.github/assets/eeo_logo.png" alt="Easy-EO logo" width="200">
</p>

|  |  |
| :-- | :-- |
| **Install** | [![PyPI][pypi-badge]][pypi] [![conda-forge][conda-badge]][conda] [![Python versions][pyversions-badge]][pypi] [![Platforms][platforms-badge]][ci] |
| **Build & quality** | [![CI][ci-badge]][ci] [![Latest deps][latest-deps-badge]][latest-deps] [![Coverage][codecov-badge]][codecov] [![Ruff][ruff-badge]][ruff] [![Checked with mypy][mypy-badge]][mypy] |
| **Security** | [![CodeQL][codeql-badge]][codeql] [![OpenSSF Scorecard][scorecard-badge]][scorecard] |
| **Project** | [![Documentation][docs-badge]][docs] [![License: MIT][license-badge]][license] [![DOI][doi-badge]][doi] |

[pypi]: https://pypi.org/project/easy-eo/
[pypi-badge]: https://img.shields.io/pypi/v/easy-eo.svg
[conda]: https://anaconda.org/conda-forge/easy-eo
[conda-badge]: https://img.shields.io/conda/vn/conda-forge/easy-eo.svg
[pyversions-badge]: https://img.shields.io/pypi/pyversions/easy-eo.svg
[platforms-badge]: https://img.shields.io/badge/platform-linux%20%7C%20macOS%20%7C%20windows-lightgrey.svg
[ci]: https://github.com/Tommy-Burns/easy-eo/actions/workflows/ci.yml
[ci-badge]: https://github.com/Tommy-Burns/easy-eo/actions/workflows/ci.yml/badge.svg
[latest-deps]: https://github.com/Tommy-Burns/easy-eo/actions/workflows/latest-deps.yml
[latest-deps-badge]: https://github.com/Tommy-Burns/easy-eo/actions/workflows/latest-deps.yml/badge.svg
[codecov]: https://codecov.io/gh/Tommy-Burns/easy-eo
[codecov-badge]: https://codecov.io/gh/Tommy-Burns/easy-eo/branch/main/graph/badge.svg
[ruff]: https://github.com/astral-sh/ruff
[ruff-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[mypy]: https://mypy-lang.org/
[mypy-badge]: https://img.shields.io/badge/mypy-checked-2a6db2.svg
[codeql]: https://github.com/Tommy-Burns/easy-eo/actions/workflows/codeql.yml
[codeql-badge]: https://github.com/Tommy-Burns/easy-eo/actions/workflows/codeql.yml/badge.svg
[scorecard]: https://scorecard.dev/viewer/?uri=github.com/Tommy-Burns/easy-eo
[scorecard-badge]: https://api.scorecard.dev/projects/github.com/Tommy-Burns/easy-eo/badge
[docs]: https://easy-eo.readthedocs.io/en/latest/?badge=latest
[docs-badge]: https://readthedocs.org/projects/easy-eo/badge/?version=latest
[license]: https://github.com/Tommy-Burns/easy-eo/blob/main/LICENSE
[license-badge]: https://img.shields.io/github/license/Tommy-Burns/easy-eo
[doi]: https://doi.org/10.5281/zenodo.21967655
[doi-badge]: https://zenodo.org/badge/DOI/10.5281/zenodo.21967655.svg


Easy-EO is a lightweight, extensible Python library for raster-based Earth
Observation (EO) analysis. It covers searching and loading imagery, cloud
masking, band algebra and spectral indices, time series and visualization, and
each step is a few readable lines of Python.

## From satellite archive to NDVI map

```python
import eeo

results = eeo.stac_search(
    collection="sentinel-2-l2a",
    bbox=(11.0, 46.5, 11.2, 46.7),        # area of interest, WGS 84 lon/lat
    datetime="2023-06-01/2023-08-31",
    cloud_cover=20,
    limit=1,
)
scene = results[0].load(assets=["B04", "B08"])   # reads only the area of interest
ndvi = scene.ndvi(red="B04", nir="B08")
ndvi.plot_raster()
```

That is the whole workflow: no scene downloads and no GDAL wrangling. The search
queries [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/)
by default, and any other STAC catalog works through `catalog="<catalog URL>"`.
The load streams only the window covering your bounding box over HTTP, which
here is 14 MB out of a 240 MB Sentinel-2 tile and takes a few seconds.

`stac_search` needs the STAC extra: `pip install "easy-eo[stac]"`, or the
[conda equivalent](#installation). To start offline instead, jump to the
[hosted sample dataset](#quick-example), which needs no network after the first
call.

---

## Features

| | What you get | Guide |
| --- | --- | --- |
| **Catalog search** | `stac_search()` over any STAC catalog, by bounding box or by a polygon; loading reads only your area of interest over HTTP; `deduplicate()` drops reprocessed copies of the same acquisition | [Satellite data](https://easy-eo.readthedocs.io/en/latest/user_guide/loading_satellite_data.html) |
| **Downloaded products** | `load_sentinel2()` and `load_landsat()` read a Sentinel-2 L2A `.SAFE` or a Landsat Collection 2 product as you downloaded it, still zipped or tarred, with bands chosen by name and the acquisition time recorded | [Downloaded scenes](https://easy-eo.readthedocs.io/en/latest/user_guide/loading_downloaded_scenes.html) |
| **Cloud masking** | `mask_clouds()` sets cloud, cloud shadow and cirrus to nodata from the Sentinel-2 scene classification or the Landsat `QA_PIXEL` band, including medium-confidence cloud; `clear_fraction()` reports how much of a scene is left | [Masking clouds](https://easy-eo.readthedocs.io/en/latest/user_guide/masking_clouds.html) |
| **Time series** | `time_series()` builds a series from a search, a folder of GeoTIFFs or scenes you loaded; `map()` applies any operation to every date; median/mean/min/max, cloud-free `composite()`, monthly or weekly `resample_time()`; `extract_at()` returns a pixel's trajectory as a pandas DataFrame | [Time series](https://easy-eo.readthedocs.io/en/latest/user_guide/time_series.html) |
| **Spectral indices** | `ndvi`, `ndwi`, `ndmi`, `ndbi`, `evi`, `savi`, plus `normalized_difference` for anything else, all chainable and float32 | [Spectral indices](https://easy-eo.readthedocs.io/en/latest/user_guide/spectral_indices.html) |
| **Band algebra** | `add`, `subtract`, `multiply`, `divide`, `power`, `sqrt`, `log`, `absolute`, and the matching operators | [Operations](https://easy-eo.readthedocs.io/en/latest/user_guide/ops.html) |
| **Preprocessing** | Clip to a bounding box or a vector, resample, reproject, mosaic, stack, normalize (min-max, percentile, z-score) | [Preprocessing](https://easy-eo.readthedocs.io/en/latest/user_guide/preprocessing.html) |
| **Large rasters** | Algebra, indices and normalization run a block at a time; `apply_blockwise()` streams a computation of your own to disk; `load_raster(path, chunks=...)` opens a raster lazily with dask, including a COG over HTTP | [Large rasters](https://easy-eo.readthedocs.io/en/latest/user_guide/large_rasters.html) |
| **Named bands and metadata** | Address any band as `"red"` or `"nir"` wherever a 1-based index works; names survive a GeoTIFF round-trip; a timestamp and free-form `attrs` follow the data through every operation; `describe()` summarises a raster without reading its pixels | [Naming bands](https://easy-eo.readthedocs.io/en/latest/user_guide/band_names.html) · [The dataset](https://easy-eo.readthedocs.io/en/latest/user_guide/core_dataset.html) |
| **Statistics** | Per-band min/max/mean/percentile with their pixel locations, and value extraction at a coordinate | [Statistical locations](https://easy-eo.readthedocs.io/en/latest/user_guide/statistical_locations.html) |
| **Visualization** | Single bands, RGB composites, histograms, map-plus-histogram views, and for a series a trajectory plot and a filmstrip of dates, all read at display resolution | [Visualization](https://easy-eo.readthedocs.io/en/latest/user_guide/visualization.html) |
| **Predictable nodata and dtype** | One written-down contract every operation follows: mask before compute, nodata stays contagious, fractional results are float32 | [Nodata and dtype](https://easy-eo.readthedocs.io/en/latest/user_guide/nodata_and_dtype.html) |
| **Ecosystem interop** | `to_xarray()` / `from_xarray()` in both directions, and `to_xarray()` on a whole series; NumPy, Rasterio and xarray backends behind one interface | [xarray interop](https://easy-eo.readthedocs.io/en/latest/user_guide/xarray_interop.html) · [Backends](https://easy-eo.readthedocs.io/en/latest/backends.html) |
| **Typed and tested** | Ships `py.typed`, 2,800+ tests, 99% coverage, checked on Python 3.10-3.14 across Linux, macOS and Windows | [Contributing](https://github.com/Tommy-Burns/easy-eo/blob/main/CONTRIBUTING.md) |

---

## Before and after

One ordinary task: clip a 4-band scene to an area of interest held in a vector
file, compute NDVI, and save it as a GeoTIFF. Both versions below run as
written, against the same
[hosted sample dataset](https://easy-eo.readthedocs.io/en/latest/user_guide/sample_data.html)
(a 1024x1024 Sentinel-2 subset and a boundary polygon), so you can paste either
one and watch it work.

Here it is in raw Rasterio, GeoPandas and NumPy, with Easy-EO not installed at
all:

```python
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask

BASE = "https://github.com/Tommy-Burns/easy-eo/releases/download/sample-data-v1/"

with rasterio.open(BASE + "sentinel2_small_cog.tif") as src:
    aoi = gpd.read_file(BASE + "roi.gpkg").to_crs(src.crs)
    clipped, transform = mask(src, aoi.geometry.values, crop=True)
    bands = {name: i for i, name in enumerate(src.descriptions)}
    nodata = src.nodata
    profile = src.profile

red = clipped[bands["red"]].astype("float32")
nir = clipped[bands["nir"]].astype("float32")

valid = (red != nodata) & (nir != nodata)
total = nir + red
ndvi = np.where(valid & (total != 0), (nir - red) / np.where(total == 0, 1, total), 0.0)
ndvi = np.where(valid, ndvi, np.nan).astype("float32")

profile.update(
    count=1, dtype="float32", nodata=np.nan,
    height=ndvi.shape[0], width=ndvi.shape[1], transform=transform,
)
with rasterio.open("ndvi.tif", "w", **profile) as dst:
    dst.write(ndvi, 1)
```

And in Easy-EO, where `load_sample_dataset()` fetches the same two files and
caches them:

```python
import eeo
from eeo.datasets import load_sample_dataset

sd = load_sample_dataset()

(
    eeo.load_raster(path=sd.sentinel2_cog_stacked)
    .clip_raster_with_vector(vector_file=sd.boundary)
    .ndvi(red="red", nir="nir")
    .save_raster(path="ndvi.tif")
)
```

Both blocks produce **byte-identical output**: the same shape, CRS, transform
and nodata, and every pixel value, including the roughly one quarter of the
image that the clip masks away.

So the point is not the line count. Rasterio makes you take four decisions by
hand, and each one is a chance to be quietly wrong: reprojecting the AOI into
the raster's CRS (the sample boundary is lon/lat, the scene is UTM), mapping
band names to indices, masking nodata before the arithmetic, and rebuilding the
output profile. Drop just the mask and NDVI comes out as `0.0` across the
clipped-away quarter of the image, a value that looks like bare ground in your
statistics and your plot rather than like missing data.

Easy-EO applies those same rules for you, consistently, on every operation.
They are written down in the
[nodata and dtype contract](https://easy-eo.readthedocs.io/en/latest/user_guide/nodata_and_dtype.html),
and each one is backed by tests.

---

## A season, not a scene

One cloud-free NDVI map per month over a growing season, and how one field
changed through it:

```python
import eeo

results = eeo.stac_search(
    collection="sentinel-2-l2a",
    bbox=(5.60, 52.05, 5.65, 52.085),
    datetime="2023-04-01/2023-09-30",
    cloud_cover=50,
)
ts = eeo.time_series(source=results.deduplicate(), assets=["B04", "B08", "SCL"])

monthly = ts.resample_time(freq="MS").composite()      # one cloud-free raster per month
ndvi = monthly.map(eeo.ndvi, red="B04", nir="B08", name="ndvi")

ndvi.plot_filmstrip(cmap="RdYlGn")                     # a small map per month
ndvi.extract_at(coordinates=(5.625, 52.0675), crs="EPSG:4326")   # a pandas DataFrame
```

`composite()` masks every date with its own scene classification band before
taking the median, so a pixel is built only from the dates it was clear, and
the `SCL` band is dropped from the result. A month that was overcast
throughout comes back as nodata rather than as a cloud-coloured value.

`deduplicate()` keeps one copy of each acquisition. Catalogs publish
reprocessed scenes alongside the originals, and without it the repeated dates
would count twice in the median.

---

## Products as you downloaded them

A Sentinel-2 product still zipped, and a Landsat product still tarred, read by
band name:

```python
import eeo

s2 = eeo.load_sentinel2(
    path="S2A_MSIL2A_20260814T100041_N0512_R122_T33TVM_20260814T151116.SAFE.zip",
    bands=["red", "nir", "scl"],
)
l9 = eeo.load_landsat(
    path="LC09_L2SP_192027_20260815_20260816_02_T1.tar",
    bands=["red", "nir08", "qa_pixel"],
)

clear = l9.mask_clouds()      # cloud, cloud shadow and cirrus set to nodata
clear.clear_fraction()        # share of the scene still usable
ndvi = clear.ndvi(red="red", nir="nir08")
```

Each loader resolves the band names to the right files for that mission and
product, and records the acquisition time and product metadata. Pass
`bbox=(minx, miny, maxx, maxy)` in lon/lat to read only part of the scene.

---

## Installation

Python 3.10 or newer, from either package manager:

```bash
pip install easy-eo
```

```bash
conda install -c conda-forge easy-eo
```

That is everything you need for the core: raster I/O, the product loaders,
cloud masking, algebra, indices, preprocessing, time series and plotting.
Three heavier integrations are kept separate, so you only install them if you
use them:

| Adds | pip | conda |
| --- | --- | --- |
| `stac_search()` and loading scenes from STAC catalogs | `pip install "easy-eo[stac]"` | `conda install -c conda-forge easy-eo pystac-client planetary-computer` |
| `to_xarray()` / `from_xarray()` | `pip install "easy-eo[xarray]"` | `conda install -c conda-forge easy-eo xarray rioxarray` |
| Opening a raster lazily (`load_raster(path, chunks=...)`), including over HTTP | `pip install "easy-eo[lazy]"` | `conda install -c conda-forge easy-eo xarray rioxarray dask-core` |

pip extras compose: `pip install "easy-eo[stac,xarray]"` installs both. conda
has no concept of extras, so `conda install "easy-eo[stac]"` is not a valid
command; the same packages are installed by name instead, as above.

**Use one package manager, not both.** If Easy-EO came from conda, install the
extras from conda too. conda's solver knows nothing about pip-installed files,
so a later `conda install` or `conda update` can overwrite them or leave a
second copy of a shared dependency in the environment. Every extra dependency
is on conda-forge, so there is no reason to mix.

Without an extra installed, the features that need it raise a
`MissingDependencyError` that tells you exactly what to install. Nothing fails
silently at import time.

## Quick example

Offline after the first call, using the hosted sample dataset:

```python
import eeo
from eeo.datasets import load_sample_dataset

sd = load_sample_dataset()
scene = eeo.load_raster(path=sd.sentinel2_cog_stacked)   # red, green, blue, nir bands
scene.describe()                                          # metadata only, no pixels read

ndvi = scene.ndvi(red="red", nir="nir", name="NDVI")
ndvi.plot_raster(cmap="RdYlGn", colorbar=True)
```

Operations chain, and a multi-line chain is wrapped in parentheses:

```python
result = (
    scene.clip_raster_with_vector(vector_file=sd.boundary)
    .ndvi(red="red", nir="nir")
    .normalize_percentile(lower_percentile=2, upper_percentile=98)
)
```

## Tutorials

Seventeen runnable notebooks live in
[`examples/`](https://github.com/Tommy-Burns/easy-eo/blob/main/examples/README.md),
from first install through to complete analyses (flood mapping, drought stress,
land cover, terrain, and a season of cloud-free composites). Each one opens in
Colab with no local setup, because the first cell installs Easy-EO when it
detects Colab:

| | |
| --- | --- |
| [Quickstart: NDVI](https://github.com/Tommy-Burns/easy-eo/blob/main/examples/00_getting_started/02_quickstart_ndvi.ipynb): open a scene, compute an index, plot it | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/00_getting_started/02_quickstart_ndvi.ipynb) |
| [Search and load from STAC](https://github.com/Tommy-Burns/easy-eo/blob/main/examples/02_data_access/02_stac_search_and_load.ipynb): find real scenes, read them over HTTP | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/02_data_access/02_stac_search_and_load.ipynb) |
| [Cloud-free composites and trends](https://github.com/Tommy-Burns/easy-eo/blob/main/examples/04_timeseries/01_cloud_free_composite_and_trends.ipynb): a season of Sentinel-2, masked, composited and followed through time | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/04_timeseries/01_cloud_free_composite_and_trends.ipynb) |
| [Flood mapping with NDWI](https://github.com/Tommy-Burns/easy-eo/blob/main/examples/03_real_world/01_flood_mapping_ndwi.ipynb): Pakistan 2022, before/after, area affected | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/03_real_world/01_flood_mapping_ndwi.ipynb) |

The full index, including what each notebook covers, is in
[`examples/README.md`](https://github.com/Tommy-Burns/easy-eo/blob/main/examples/README.md)
and on the [tutorials page](https://easy-eo.readthedocs.io/en/latest/tutorials.html)
of the documentation.

---

## Gallery

Every image below comes straight out of an Easy-EO plotting call on the sample
dataset, with the library's own defaults and no touch-ups. Regenerate them all
with `python scripts/build_gallery.py`.

| | |
| --- | --- |
| <img src="https://raw.githubusercontent.com/Tommy-Burns/easy-eo/main/.github/assets/gallery/composite_true_colour.jpg" alt="True colour composite of a Sentinel-2 scene"> | <img src="https://raw.githubusercontent.com/Tommy-Burns/easy-eo/main/.github/assets/gallery/composite_false_colour.jpg" alt="False colour composite, vegetation in red"> |
| `scene.plot_composite(bands=["red", "green", "blue"])` | `scene.plot_composite(bands=["nir", "red", "green"])` |
| <img src="https://raw.githubusercontent.com/Tommy-Burns/easy-eo/main/.github/assets/gallery/index_ndvi.jpg" alt="NDVI map on a red-yellow-green colour scale"> | <img src="https://raw.githubusercontent.com/Tommy-Burns/easy-eo/main/.github/assets/gallery/dem_terrain.jpg" alt="Copernicus DEM elevation map"> |
| `scene.ndvi(red="red", nir="nir", name="NDVI").plot_raster(cmap="RdYlGn")` | `dem.plot_raster(cmap="Spectral_r")`, the same call on a DEM |
| <img src="https://raw.githubusercontent.com/Tommy-Burns/easy-eo/main/.github/assets/gallery/histogram_bands.png" alt="Value distribution of each of the four bands"> | <img src="https://raw.githubusercontent.com/Tommy-Burns/easy-eo/main/.github/assets/gallery/clip_ndvi_histogram.png" alt="NDVI clipped to a hexagonal boundary beside its histogram"> |
| `scene.plot_histogram()`, every band at once | `clipped.plot_raster_with_histogram(cmap="RdYlGn")` |

Bands are addressed by name throughout (`"red"`, `"nir"`) because the sample
carries band descriptions; a 1-based index works anywhere a name does.

## Backends

One dataset type, with three backends underneath it:

| Backend | What it is for |
| --- | --- |
| Rasterio | Files and COGs GDAL can open, local or over HTTP, from `load_raster()` |
| NumPy | In-memory arrays with a CRS and transform attached, from `load_array()` and the product loaders |
| xarray + dask | Lazy, chunked reading of rasters larger than memory, from `load_raster(path, chunks=...)` (the `lazy` extra) |

Every operation, statistic and plot works on the Rasterio and lazy backends and
returns the same answer on both. On the NumPy backend, clipping, reprojecting,
mosaicking and stacking need a `.to_rasterio()` first, and the error says so.

See [Backends](https://easy-eo.readthedocs.io/en/latest/backends.html) for how
they fit together.

## Documentation

Full documentation, with user guides and the API reference:
[easy-eo.readthedocs.io](https://easy-eo.readthedocs.io/en/latest/index.html)

## Citing Easy-EO

If Easy-EO contributes to published work, please cite it. The DOI
[10.5281/zenodo.21967655](https://doi.org/10.5281/zenodo.21967655) always
resolves to the latest release, and
[`CITATION.cff`](https://github.com/Tommy-Burns/easy-eo/blob/main/CITATION.cff)
drives GitHub's "Cite this repository" button. The
[citation page](https://easy-eo.readthedocs.io/en/latest/citation.html) has a
BibTeX entry, and a separate one for the sample dataset.

## Project status

Active development. The API is stabilising but may change before v1.0, and
every breaking change is listed under a **Breaking** heading in the
[changelog](https://github.com/Tommy-Burns/easy-eo/blob/main/CHANGELOG.md).

## Contributing

Contributions are welcome: bug reports, feature requests and documentation
improvements as much as code. Please open an issue or pull request on GitHub,
and see [`CONTRIBUTING.md`](https://github.com/Tommy-Burns/easy-eo/blob/main/CONTRIBUTING.md)
for how to set up a development environment and run the checks. Report
security issues privately, as described in
[`SECURITY.md`](https://github.com/Tommy-Burns/easy-eo/blob/main/SECURITY.md).

## License

MIT License © 2025 Thomas Burns Botchwey
