# Easy-EO

[![CI](https://github.com/Tommy-Burns/easy-eo/actions/workflows/ci.yml/badge.svg)](https://github.com/Tommy-Burns/easy-eo/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Tommy-Burns/easy-eo/branch/main/graph/badge.svg)](https://codecov.io/gh/Tommy-Burns/easy-eo)
[![PyPI](https://img.shields.io/pypi/v/easy-eo.svg)](https://pypi.org/project/easy-eo/)
[![Python versions](https://img.shields.io/pypi/pyversions/easy-eo.svg)](https://pypi.org/project/easy-eo/)
[![Documentation Status](https://readthedocs.org/projects/easy-eo/badge/?version=latest)](https://easy-eo.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/github/license/Tommy-Burns/easy-eo)](https://github.com/Tommy-Burns/easy-eo/blob/main/LICENSE)

Easy-EO is a lightweight, extensible Python library for raster-based Earth Observation (EO) analysis which allows for chainable raster processing, algebra, and visualization.
It provides high-level abstractions over libraries such as Rasterio, NumPy, and Matplotlib, enabling users to perform common earth-observation analyses and visualization tasks efficiently, without dealing with the underlying complexity.

---

## Features

- Raster operations with spatial awareness
- Algebraic operations (`add`, `subtract`, `multiply`, `divide`)
- Resampling, reprojection, and alignment
- Clip rasters using vectors or bounding boxes
- Normalization (min-max, percentile, z-score)
- Visualization helpers (bands, composites, histograms)
- Backend-aware design (NumPy ↔ Rasterio)

---

## Installation

```bash
conda create -n env_name python=3.10
conda activate env_name
pip install easy-eo
```

## Quick Example
```python
from eeo import load_raster

ds_nir = load_raster("path/to/nir.tif")
ds_red = load_raster("path/to/red.tif")

# Chainable example: clip -> resample -> compute NDVI -> multiply
result = (
    ds_nir.clip_raster_with_bbox((0, 0, 1000, 1000))
    .resample(scale_factor=2)
    .normalized_difference(ds_red)
    .multiply(100)
)
```

## Supported Backends
| Backend  | Description                                          |
|----------|------------------------------------------------------|
| NumPy    | Fast, in-memory arrays without I/O                   |
| Rasterio | Full geospatial support (CRS, transform, resampling) |


## Documentation

📚 Full documentation is available at:

👉 [Easy-EO Documentation](https://easy-eo.readthedocs.io/en/latest/index.html)

## Project Status
🚧 Active development
The API is stabilizing but may change before v1.0.

## Contributing
Contributions are welcome!
 - Bug reports
 - Feature requests
 - Documentation improvements

Please open an issue or pull request on GitHub.

## License
MIT License © 2025 Thomas Burns Botchwey
