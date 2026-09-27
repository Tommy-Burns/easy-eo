"""Contract tests for the shared fixtures in ``tests/conftest.py``.

Each test pins the properties (backend, shape, dtype, CRS, nodata, values)
that other test modules rely on, so a fixture change that would silently
weaken the suite fails here first.
"""

import numpy as np
from rasterio.crs import CRS

from eeo.core.adapters import NumpyRasterioAdapter, RasterioAdapter


def test_single_band_float32_contract(single_band_float32):
    ds = single_band_float32
    assert isinstance(ds._adapter, RasterioAdapter)
    assert ds.get_count() == 1
    assert ds.get_shape() == (6, 6)
    data = ds.read()
    assert data.dtype == np.float32
    np.testing.assert_array_equal(data[0], np.arange(36, dtype=np.float32).reshape(6, 6))
    assert ds.get_crs() == CRS.from_epsg(32633)


def test_multiband_uint16_contract(multiband_uint16):
    ds = multiband_uint16
    assert isinstance(ds._adapter, RasterioAdapter)
    assert ds.get_count() == 4
    data = ds.read()
    assert data.dtype == np.uint16
    np.testing.assert_array_equal(data[2], 3000 + np.arange(36, dtype=np.uint16).reshape(6, 6))


def test_raster_with_nodata_contract(raster_with_nodata):
    ds = raster_with_nodata
    assert ds._adapter.get_nodata() == -9999.0
    data = ds.read()[0]
    assert np.all(data[:2, :2] == -9999.0)
    assert not np.any(data[2:, :] == -9999.0)


def test_crs_mismatch_pair_contract(crs_mismatch_pair):
    utm, geo = crs_mismatch_pair
    assert utm.get_crs() != geo.get_crs()
    np.testing.assert_array_equal(utm.read(), geo.read())


def test_shape_mismatch_pair_contract(shape_mismatch_pair):
    fine, coarse = shape_mismatch_pair
    assert fine.get_shape() == (6, 6)
    assert coarse.get_shape() == (3, 3)
    assert fine.get_crs() == coarse.get_crs()
    assert tuple(fine.get_bounds()) == tuple(coarse.get_bounds())


def test_numpy_backed_dataset_contract(numpy_backed_dataset):
    ds = numpy_backed_dataset
    assert isinstance(ds._adapter, NumpyRasterioAdapter)
    assert isinstance(ds.ds, np.ndarray)
    assert ds.read().dtype == np.float32


def test_raster_3x3_contract(raster_3x3):
    ds = raster_3x3
    assert isinstance(ds._adapter, NumpyRasterioAdapter)
    np.testing.assert_array_equal(ds.read()[0], np.arange(1, 10, dtype=np.float32).reshape(3, 3))
    assert ds.get_crs() == CRS.from_epsg(4326)


def test_season_stack_contract(season_stack):
    assert len(season_stack) == 5
    for index, ds in enumerate(season_stack):
        assert isinstance(ds._adapter, RasterioAdapter)
        assert ds.get_count() == 2
        assert ds.get_shape() == (4, 4)
        assert ds.band_names == ["red", "nir"]
        assert ds.get_crs() == CRS.from_epsg(32633)
        assert ds.get_metadata()["nodata"] == 0
        assert ds.timestamp.month == (3, 4, 5, 6, 7)[index]
        assert ds.timestamp.tzinfo is not None
        assert ds.read().dtype == np.uint16

    # Uniform within a scene, so every reduction across the stack is exact.
    reds = [int(ds.read()[0, 1, 1]) for ds in season_stack]
    nirs = [int(ds.read()[1, 1, 1]) for ds in season_stack]
    assert reds == [1000, 900, 800, 900, 1000]
    assert nirs == [2000, 3000, 4000, 3000, 2000]

    # Pixel (0, 0) is the nodata gap, at the third and fourth timesteps only.
    gaps = [int(ds.read()[0, 0, 0]) for ds in season_stack]
    assert gaps == [1000, 900, 0, 0, 1000]


def test_season_series_contract(season_series):
    assert len(season_series) == 5
    assert season_series.band_names == ["red", "nir"]
    assert season_series.shape == (4, 4)
    assert [stamp.month for stamp in season_series.timestamps] == [3, 4, 5, 6, 7]


def test_season_reference_contract(season_reference, season_stack):
    assert isinstance(season_reference._adapter, RasterioAdapter)
    assert season_reference.get_count() == 1
    # On the stack's grid, so a two-raster op needs no alignment.
    assert season_reference.get_shape() == season_stack[0].get_shape()
    assert season_reference.get_transform() == season_stack[0].get_transform()
    np.testing.assert_array_equal(
        season_reference.read()[0], np.full((4, 4), 1000, dtype=np.uint16)
    )
