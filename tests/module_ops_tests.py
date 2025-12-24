import numpy as np
import pytest
import rasterio as rio
from affine import Affine
from rasterio.crs import CRS

from eeo.core.core import EEORasterDataset
from eeo import load_array
from eeo.ops.algebra import (
    add, subtract, multiply, divide, power,
    sqrt, log, absolute,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def raster_a():
    array = np.array([[1, 2], [3, 4]], dtype=np.float32)
    return load_array(array, transform=Affine.identity(), crs=CRS.from_epsg(4326))


@pytest.fixture
def raster_b():
    array = np.array([[4, 3], [2, 1]], dtype=np.float32)
    return load_array(array, transform=Affine.identity(), crs=CRS.from_epsg(4326))


# ---------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------

def test_add_raster_raster(raster_a, raster_b):
    result = raster_a.add(raster_b)
    np.testing.assert_array_equal(result.read()[0], [[5, 5], [5, 5]])


def test_add_raster_scalar(raster_a):
    result = raster_a.add(10)
    np.testing.assert_array_equal(result.read()[0], [[11, 12], [13, 14]])


def test_subtract(raster_a, raster_b):
    result = raster_a.subtract(raster_b)
    np.testing.assert_array_equal(result.read()[0], [[-3, -1], [1, 3]])


def test_multiply(raster_a, raster_b):
    result = raster_a.multiply(raster_b)
    np.testing.assert_array_equal(result.read()[0], [[4, 6], [6, 4]])


def test_divide_safe(raster_a):
    result = raster_a.divide(0)
    assert np.all(result.read() == 0)


def test_power(raster_a):
    result = raster_a.power(2)
    np.testing.assert_array_equal(result.read()[0], [[1, 4], [9, 16]])


# ---------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------

def test_sqrt(raster_a):
    result = raster_a.sqrt()
    np.testing.assert_allclose(
        result.read()[0],
        np.sqrt([[1, 2], [3, 4]]),
        rtol=1e-6,
    )


def test_log_default_base(raster_a):
    result = raster_a.log()
    assert np.all(np.isfinite(result.read()))


def test_absolute():
    array = np.array([[-1, -2], [3, -4]], dtype=np.float32)
    ds = load_array(array, transform=Affine.identity(), crs=CRS.from_epsg(4326))

    result = ds.absolute()
    np.testing.assert_array_equal(result.read()[0], [[1, 2], [3, 4]])
