import os
import numpy as np
import pytest
import rasterio as rio
from affine import Affine
from rasterio.crs import CRS

from eeo import load_raster, load_array
from eeo.core.core import EEORasterDataset


# File does not exist
def test_load_raster_file_not_found(tmp_path):
    missing = tmp_path / "missing.tif"

    with pytest.raises(FileNotFoundError):
        load_raster(str(missing))

# Invalid raster file (exists but not a raster)
def test_load_raster_invalid_file(tmp_path):
    bad_file = tmp_path / "not_a_raster.txt"
    bad_file.write_text("this is not a raster")

    with pytest.raises(RuntimeError, match="could not be opened"):
        load_raster(str(bad_file))

# Valid raster
def test_load_raster_success(tmp_path):
    path = tmp_path / "test.tif"

    data = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    transform = Affine.identity()
    crs = CRS.from_epsg(4326)

    with rio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    ds = load_raster(str(path))

    assert isinstance(ds, EEORasterDataset)
    assert ds.get_shape() == (2, 2)
    assert ds.get_count() == 1
    assert ds.get_crs() == crs

# Load array non-empty input
def test_load_array_rejects_non_numpy():
    with pytest.raises(TypeError):
        load_array([[1, 2], [3, 4]])


# invalid array dimensions
@pytest.mark.parametrize("shape", [(10,), (2, 2, 2, 2)])
def test_invalid_array_dimensions(shape):
    array = np.zeros(shape)

    with pytest.raises(ValueError):
        load_array(array)

# Load 2D array successfully
def test_load_array_2d_success():
    array = np.array([[1, 2], [3, 4]], dtype=np.float32)
    transform = Affine.identity()
    crs = CRS.from_epsg(4326)

    ds = load_array(array, transform=transform, crs=crs)

    assert isinstance(ds, EEORasterDataset)
    assert ds.get_shape() == (2, 2)
    assert ds.get_count() == 1
    np.testing.assert_array_equal(ds.read()[0], array)


# Load 3D array successfully (multi-band)
def test_load_array_3d_success():
    array = np.random.rand(3, 4, 5)
    transform = Affine.identity()
    crs = CRS.from_epsg(3857)

    ds = load_array(array, transform=transform, crs=crs)

    assert ds.get_count() == 3
    assert ds.get_shape() == (4, 5)

    band2 = ds.get_band(2)
    np.testing.assert_array_equal(band2, array[1])

# Nodata propagation
def test_load_array_nodata_propagation():
    array = np.array([[1, -999], [3, 4]], dtype=np.float32)
    nodata = -999

    ds = load_array(array, nodata=nodata)

    meta = ds.get_metadata()
    assert meta["nodata"] == nodata
