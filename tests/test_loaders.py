import numpy as np
import pytest
import rasterio as rio
from affine import Affine
from rasterio.crs import CRS

from eeo import load_array, load_raster
from eeo.core.adapters import RasterioAdapter
from eeo.core.core import EEORasterDataset
from eeo.core.exceptions import BackendError, ValidationError


def _write_tiny_tif(path):
    data = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    transform = Affine.translation(0, 2) * Affine.scale(1, -1)

    with rio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype=data.dtype,
        crs=CRS.from_epsg(4326),
        transform=transform,
    ) as dst:
        dst.write(data, 1)


# File does not exist
def test_load_raster_file_not_found(tmp_path):
    missing = tmp_path / "missing.tif"

    with pytest.raises(FileNotFoundError):
        load_raster(str(missing))


# Invalid raster file (exists but not a raster)
def test_load_raster_invalid_file(tmp_path):
    bad_file = tmp_path / "not_a_raster.txt"
    bad_file.write_text("this is not a raster")

    with pytest.raises(BackendError, match="could not be opened"):
        load_raster(str(bad_file))


# Valid raster
def test_load_raster_success(tmp_path):
    path = tmp_path / "test.tif"

    data = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    transform = Affine.translation(0, 2) * Affine.scale(1, -1)
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


# Memory model: opening a raster and touching metadata must not read pixels
def test_load_raster_performs_no_pixel_read(tmp_path, monkeypatch):
    path = tmp_path / "lazy.tif"
    _write_tiny_tif(path)

    read_calls = []
    real_read = RasterioAdapter.read
    real_read_band = RasterioAdapter.read_band

    def spy_read(self, *args, **kwargs):
        read_calls.append(("read", args, kwargs))
        return real_read(self, *args, **kwargs)

    def spy_read_band(self, idx):
        read_calls.append(("read_band", idx))
        return real_read_band(self, idx)

    monkeypatch.setattr(RasterioAdapter, "read", spy_read)
    monkeypatch.setattr(RasterioAdapter, "read_band", spy_read_band)

    ds = load_raster(str(path))

    # metadata access must also stay read-free
    ds.get_crs()
    ds.get_transform()
    ds.get_shape()
    ds.get_bounds()
    ds.get_metadata()
    ds.get_width()
    ds.get_height()
    ds.get_count()

    assert read_calls == []

    # the spy is live: an actual pixel read goes through it and still works
    band = ds.get_band(1)
    np.testing.assert_array_equal(band, np.array([[1, 2], [3, 4]], dtype=np.uint8))
    assert read_calls == [("read_band", 1)]

    ds.close()


# Load array non-empty input
def test_load_array_rejects_non_numpy():
    with pytest.raises(ValidationError):
        load_array([[1, 2], [3, 4]])


# invalid array dimensions
@pytest.mark.parametrize("shape", [(10,), (2, 2, 2, 2)])
def test_invalid_array_dimensions(shape):
    array = np.zeros(shape)

    with pytest.raises(ValidationError):
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
    array = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
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
