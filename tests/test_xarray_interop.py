"""Tests for EEORasterDataset.to_xarray (the eeo.io.xarray interop module)."""

import datetime as dt
import importlib

import numpy as np
import pytest
import rasterio as rio
from rasterio.transform import Affine

import eeo
from eeo import load_array

xr = pytest.importorskip("xarray", reason="needs the optional xarray extra")
rioxarray = pytest.importorskip("rioxarray", reason="needs the optional xarray extra")

UTM_CRS = "EPSG:32633"


def _write_geotiff(path, array, transform, crs=UTM_CRS, nodata=None, descriptions=None):
    """Write a GeoTIFF so rioxarray can open the very same raster."""
    profile = {
        "driver": "GTiff",
        "height": array.shape[1],
        "width": array.shape[2],
        "count": array.shape[0],
        "dtype": array.dtype.name,
        "crs": crs,
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rio.open(path, "w", **profile) as dst:
        dst.write(array)
        for idx, name in enumerate(descriptions or [], start=1):
            if name:
                dst.set_band_description(idx, name)
    return path


# ---------------------------------------------------------------- structure


def test_dims_shape_and_dtype(multiband_uint16):
    da = multiband_uint16.to_xarray()

    assert isinstance(da, xr.DataArray)
    assert da.dims == ("band", "y", "x")
    assert da.shape == (4, 6, 6)
    assert da.dtype == np.uint16


def test_single_band_still_has_a_band_dimension(single_band_float32):
    da = single_band_float32.to_xarray()

    assert da.dims == ("band", "y", "x")
    assert da.shape == (1, 6, 6)
    assert da.band.values.tolist() == [1]


def test_band_coordinate_is_one_based(multiband_uint16):
    da = multiband_uint16.to_xarray()

    assert da.band.values.tolist() == [1, 2, 3, 4]


def test_values_match_the_raster(multiband_uint16):
    da = multiband_uint16.to_xarray()

    np.testing.assert_array_equal(da.values, multiband_uint16.read())


def test_nodata_pixels_are_carried_through_as_stored(raster_with_nodata):
    da = raster_with_nodata.to_xarray()

    np.testing.assert_array_equal(da.values, raster_with_nodata.read())
    assert da.rio.nodata == -9999.0
    # The sentinel is preserved rather than converted, matching read().
    assert (da.values == -9999.0).sum() == 4


# ---------------------------------------------------------- georeferencing


def test_crs_transform_and_nodata_are_carried(raster_with_nodata):
    da = raster_with_nodata.to_xarray()

    assert da.rio.crs == raster_with_nodata.get_crs()
    assert da.rio.transform() == raster_with_nodata.get_transform()
    assert da.rio.nodata == raster_with_nodata.get_metadata()["nodata"]


def test_nan_nodata_is_carried(single_band_float32):
    nan_nodata = single_band_float32.divide(2)  # fractional op -> nodata=nan
    array = nan_nodata.read()
    array[0, 0, 0] = np.nan
    ds = load_array(array, transform=nan_nodata.get_transform(), crs=UTM_CRS, nodata=np.nan)

    da = ds.to_xarray()

    assert np.isnan(da.rio.nodata)
    assert np.isnan(da.values[0, 0, 0])


def test_no_declared_nodata_leaves_nodata_unset(single_band_float32):
    assert single_band_float32.get_metadata()["nodata"] is None

    da = single_band_float32.to_xarray()

    assert da.rio.nodata is None
    assert "_FillValue" not in da.attrs


def test_coordinates_are_pixel_centres(nonsquare_float32):
    da = nonsquare_float32.to_xarray()
    transform = nonsquare_float32.get_transform()

    # First pixel centre is half a pixel in from the raster's origin.
    assert da.x.values[0] == pytest.approx(transform.c + transform.a / 2)
    assert da.y.values[0] == pytest.approx(transform.f + transform.e / 2)
    # y descends for a north-up raster.
    assert da.y.values[0] > da.y.values[-1]
    assert da.x.size == 8
    assert da.y.size == 4


def test_layout_matches_rioxarray_open_rasterio(tmp_path, multiband_uint16):
    """The conversion must be indistinguishable from rioxarray's own read."""
    path = _write_geotiff(
        tmp_path / "scene.tif",
        multiband_uint16.read(),
        multiband_uint16.get_transform(),
    )

    ours = eeo.load_raster(str(path)).to_xarray()
    theirs = rioxarray.open_rasterio(path)

    assert ours.dims == theirs.dims
    np.testing.assert_array_equal(ours.band.values, theirs.band.values)
    np.testing.assert_allclose(ours.x.values, theirs.x.values)
    np.testing.assert_allclose(ours.y.values, theirs.y.values)
    np.testing.assert_array_equal(ours.values, theirs.values)
    assert ours.rio.crs == theirs.rio.crs
    assert ours.rio.transform() == theirs.rio.transform()


def test_rotated_transform_gets_two_dimensional_coordinates(tmp_path):
    """A rotated grid cannot use 1-D axes; follow rioxarray and emit xc/yc."""
    rotated = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0) * Affine.rotation(30)
    array = np.arange(12, dtype="float32").reshape(1, 3, 4)
    path = _write_geotiff(tmp_path / "rotated.tif", array, rotated)

    ours = eeo.load_raster(str(path)).to_xarray()
    theirs = rioxarray.open_rasterio(path)

    assert "x" not in ours.coords and "y" not in ours.coords
    np.testing.assert_allclose(ours.xc.values, theirs.xc.values)
    np.testing.assert_allclose(ours.yc.values, theirs.yc.values)
    assert ours.xc.dims == ("y", "x")
    # The full affine survives even though the coordinates cannot express it.
    assert ours.rio.transform() == rotated


def test_missing_crs_produces_no_crs_rather_than_failing():
    ds = load_array(
        np.zeros((2, 2), dtype="float32"),
        transform=Affine.identity(),
        crs=None,
    )

    da = ds.to_xarray()

    assert da.rio.crs is None
    assert da.rio.transform() == Affine.identity()


# ------------------------------------------------------------- provenance


def test_band_names_become_long_name(multiband_uint16):
    multiband_uint16.band_names = ["blue", "green", "red", "nir"]

    da = multiband_uint16.to_xarray()

    assert da.attrs["long_name"] == ("blue", "green", "red", "nir")


def test_single_band_long_name_is_a_plain_string(single_band_float32):
    """rioxarray uses a bare string for a one-band raster, not a 1-tuple."""
    single_band_float32.band_names = ["ndvi"]

    assert single_band_float32.to_xarray().attrs["long_name"] == "ndvi"


def test_partially_named_bands_keep_none_placeholders(multiband_uint16):
    multiband_uint16.set_band_name(2, "green")

    assert multiband_uint16.to_xarray().attrs["long_name"] == (None, "green", None, None)


def test_unnamed_bands_set_no_long_name(multiband_uint16):
    assert "long_name" not in multiband_uint16.to_xarray().attrs


def test_band_names_survive_a_write_through_rioxarray(tmp_path, multiband_uint16):
    """long_name is the key rioxarray writes back to GDAL band descriptions."""
    multiband_uint16.band_names = ["blue", "green", "red", "nir"]

    out = tmp_path / "named.tif"
    multiband_uint16.to_xarray().rio.to_raster(out)

    assert eeo.load_raster(str(out)).band_names == ["blue", "green", "red", "nir"]


def test_timestamp_becomes_a_scalar_time_coordinate(single_band_float32):
    single_band_float32.timestamp = dt.datetime(2023, 9, 7, 10, 0, 31)

    da = single_band_float32.to_xarray()

    assert da.time.shape == ()
    assert da.time.values == np.datetime64("2023-09-07T10:00:31")
    # A scalar coordinate adds no dimension.
    assert da.dims == ("band", "y", "x")


def test_aware_timestamp_is_converted_to_naive_utc(single_band_float32):
    tz = dt.timezone(dt.timedelta(hours=2))
    single_band_float32.timestamp = dt.datetime(2023, 9, 7, 12, 0, 31, tzinfo=tz)

    da = single_band_float32.to_xarray()

    assert da.time.values == np.datetime64("2023-09-07T10:00:31")


def test_timestamp_uses_nanosecond_precision(single_band_float32):
    """Coarser units make xarray convert and warn about it; do it up front."""
    single_band_float32.timestamp = dt.datetime(2023, 9, 7, 10, 0, 31)

    assert single_band_float32.to_xarray().time.dtype == np.dtype("datetime64[ns]")


def test_timestamp_outside_the_nanosecond_range_is_not_silently_wrapped():
    """``datetime64[ns]`` only spans 1678-2262; narrowing past it wraps years."""
    from eeo.io.xarray import _as_datetime64

    assert _as_datetime64(dt.datetime(1500, 1, 1)) == np.datetime64("1500-01-01", "us")
    assert _as_datetime64(dt.datetime(2500, 1, 1)) == np.datetime64("2500-01-01", "us")
    # Sub-second detail survives inside the range.
    assert _as_datetime64(dt.datetime(2023, 9, 7, 10, 0, 31, 123456)) == np.datetime64(
        "2023-09-07T10:00:31.123456", "ns"
    )


def test_no_timestamp_means_no_time_coordinate(single_band_float32):
    assert single_band_float32.timestamp is None

    assert "time" not in single_band_float32.to_xarray().coords


def test_attrs_are_copied(single_band_float32):
    single_band_float32.attrs = {"stac_item_id": "S2A_123", "collection": "sentinel-2-l2a"}

    da = single_band_float32.to_xarray()

    assert da.attrs["stac_item_id"] == "S2A_123"
    assert da.attrs["collection"] == "sentinel-2-l2a"


def test_attrs_cannot_shadow_the_georeferencing_metadata(raster_with_nodata):
    raster_with_nodata.band_names = ["gradient"]
    raster_with_nodata.attrs = {"long_name": "junk", "_FillValue": 1.0, "grid_mapping": "junk"}

    da = raster_with_nodata.to_xarray()

    assert da.attrs["long_name"] == "gradient"
    assert da.rio.nodata == -9999.0
    assert da.rio.crs == raster_with_nodata.get_crs()


# ------------------------------------------------------------- backends


def test_numpy_backed_dataset_converts(numpy_backed_dataset):
    da = numpy_backed_dataset.to_xarray()

    assert da.dims == ("band", "y", "x")
    assert da.shape == (1, 6, 6)
    assert da.rio.crs == numpy_backed_dataset.get_crs()
    assert da.rio.transform() == numpy_backed_dataset.get_transform()
    np.testing.assert_array_equal(da.values, numpy_backed_dataset.read())


def test_conversion_does_not_mutate_the_dataset(multiband_uint16):
    before = multiband_uint16.read().copy()
    metadata = dict(multiband_uint16.get_metadata())

    multiband_uint16.to_xarray()

    np.testing.assert_array_equal(multiband_uint16.read(), before)
    assert multiband_uint16.get_metadata() == metadata
    assert multiband_uint16.band_names == [None] * 4


@pytest.mark.parametrize("fixture", ["multiband_uint16", "numpy_backed_dataset"])
def test_conversion_returns_an_independent_array(request, fixture):
    """Writing into the DataArray must never reach back into the dataset.

    The NumPy backend's ``read()`` hands out the array it holds, so this only
    holds because the conversion copies it — and it must not silently start
    depending on rioxarray copying internally.
    """
    ds = request.getfixturevalue(fixture)
    da = ds.to_xarray()
    original = ds.read()[0, 0, 0]

    da[0, 0, 0] = 42

    assert ds.read()[0, 0, 0] == original
    assert not np.shares_memory(da.values, ds.read())


# ------------------------------------------------------- missing extra


def test_missing_xarray_raises_missing_dependency_error(monkeypatch, single_band_float32):
    """The absent-extra path is exercised even where xarray is installed."""

    def fake_import_module(name):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(eeo.MissingDependencyError) as excinfo:
        single_band_float32.to_xarray()

    message = str(excinfo.value)
    assert "xarray interop" in message
    assert "pip install 'easy-eo[xarray]'" in message


def test_missing_dependency_error_is_an_import_error(monkeypatch, single_band_float32):
    def fake_import_module(name):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError):
        single_band_float32.to_xarray()
