# Easy-EO tutorial notebooks

Runnable tutorials, grouped from first install through to complete analyses.

Every notebook is standalone - start wherever the topic you need is. All of them
get their data from `eeo.datasets.load_sample_dataset()` or from a STAC catalog,
so none depends on files in this repository or on a particular working
directory.

## Getting started

| notebook | what it covers | run it |
| --- | --- | --- |
| [00_getting_started/01_installation_and_setup](00_getting_started/01_installation_and_setup.ipynb) | install, check the stack, optional extras, warm the sample cache | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/00_getting_started/01_installation_and_setup.ipynb) |
| [00_getting_started/02_quickstart_ndvi](00_getting_started/02_quickstart_ndvi.ipynb) | load a scene, compute NDVI, plot it, save it | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/00_getting_started/02_quickstart_ndvi.ipynb) |

## Fundamentals

| notebook | what it covers | run it |
| --- | --- | --- |
| [01_fundamentals/01_reading_and_inspecting](01_fundamentals/01_reading_and_inspecting.ipynb) | `load_raster`, deferred reads, `describe`, metadata, band names, `load_array` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/01_fundamentals/01_reading_and_inspecting.ipynb) |
| [01_fundamentals/02_clip_and_mosaic](01_fundamentals/02_clip_and_mosaic.ipynb) | clip to a vector or a bbox, `crop`/`invert`/`all_touched`, mosaic tiles | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/01_fundamentals/02_clip_and_mosaic.ipynb) |
| [01_fundamentals/03_reproject_and_resample](01_fundamentals/03_reproject_and_resample.ipynb) | change CRS, change pixel size, choosing a resampling method | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/01_fundamentals/03_reproject_and_resample.ipynb) |
| [01_fundamentals/04_band_algebra_and_indices](01_fundamentals/04_band_algebra_and_indices.ipynb) | arithmetic, integer wraparound, safe division, the six indices, normalization | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/01_fundamentals/04_band_algebra_and_indices.ipynb) |
| [01_fundamentals/05_stacking_bands](01_fundamentals/05_stacking_bands.ipynb) | build a multi-band scene from per-band files; stack vs mosaic | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/01_fundamentals/05_stacking_bands.ipynb) |
| [01_fundamentals/06_statistics_and_extraction](01_fundamentals/06_statistics_and_extraction.ipynb) | min/max/mean/percentile pixels, sampling at coordinates, area statistics | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/01_fundamentals/06_statistics_and_extraction.ipynb) |
| [01_fundamentals/07_visualization](01_fundamentals/07_visualization.ipynb) | every plot function, stretching, composites, dropping to Matplotlib | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/01_fundamentals/07_visualization.ipynb) |

## Data access

| notebook | what it covers | run it |
| --- | --- | --- |
| [02_data_access/01_sample_data](02_data_access/01_sample_data.ipynb) | the bundled sample, lazy handles, caching, COGs, attribution | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/02_data_access/01_sample_data.ipynb) |
| [02_data_access/02_stac_search_and_load](02_data_access/02_stac_search_and_load.ipynb) | search a STAC catalog, load assets over HTTP, crop to an AOI ⚠️ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/02_data_access/02_stac_search_and_load.ipynb) |
| [02_data_access/03_xarray_interop](02_data_access/03_xarray_interop.ipynb) | `to_xarray()` / `from_xarray()`, and when to use xarray instead | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/02_data_access/03_xarray_interop.ipynb) |

## Real-world analyses

| notebook | what it covers | run it |
| --- | --- | --- |
| [03_real_world/01_flood_mapping_ndwi](03_real_world/01_flood_mapping_ndwi.ipynb) | Pakistan 2022 floods: pre/post NDWI, flood extent, area affected ⚠️ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/03_real_world/01_flood_mapping_ndwi.ipynb) |
| [03_real_world/02_vegetation_health_and_drought](03_real_world/02_vegetation_health_and_drought.ipynb) | California Central Valley: NDVI, NDMI, SAVI, EVI, stress detection ⚠️ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/03_real_world/02_vegetation_health_and_drought.ipynb) |
| [03_real_world/03_urban_footprint_land_cover](03_real_world/03_urban_footprint_land_cover.ipynb) | Cairo: why NDBI alone fails, and a four-class land-cover map ⚠️ | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/03_real_world/03_urban_footprint_land_cover.ipynb) |
| [03_real_world/04_terrain_analysis_with_dem](03_real_world/04_terrain_analysis_with_dem.ipynb) | slope, aspect, hillshade, hypsometry, terrain against land cover | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommy-Burns/easy-eo/blob/main/examples/03_real_world/04_terrain_analysis_with_dem.ipynb) |

⚠️ **needs network and `pip install "easy-eo[stac]"`.** These notebooks query a
live STAC catalog. The rest run offline once the sample data is cached. Every
notebook is committed with its outputs, so you can read the results on GitHub
without running anything.

## Running them

### In Colab, with nothing installed

Click the badge on any row above, or the one at the top of the notebook itself.
Its first cell detects Colab and installs Easy-EO with the extras that notebook
needs; outside Colab the same cell does nothing. Colab's filesystem is
discarded when the session ends, so the sample data is re-downloaded on each
new session.

### Locally

```bash
pip install "easy-eo[stac,xarray]" jupyterlab
jupyter lab
```

The sample data downloads on first use (a few tens of megabytes) and is cached
under `~/.cache/easy-eo`, so subsequent runs are offline. To fetch everything up
front:

```python
from eeo.datasets import load_sample_dataset

load_sample_dataset(prefetch=True)
```

### Executing the whole set

CI runs every offline notebook top to bottom on each pull request, and you can
run the same check locally:

```bash
pip install nbmake
pytest --nbmake --no-cov examples/
```

The ⚠️ notebooks are excluded automatically — each one declares
`"easy_eo": {"requires_network": true}` in its notebook metadata, which
[`conftest.py`](conftest.py) reads. Flag a new notebook the same way and it is
skipped with no CI change.

## Data attribution

The sample contains modified Copernicus Sentinel-2 data (2023) and Copernicus
GLO-30 DEM data, accessed via Microsoft Planetary Computer. The STAC notebooks
use Sentinel-2 L2A from the same source. Each sample handle carries its full
attribution - see `sd.<name>.info()`.
