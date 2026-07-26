"""Round-trip tests for the xarray interop, in both directions.

``to_xarray`` and ``from_xarray`` are each covered in detail in
``test_xarray_interop.py``. What is checked here is that composing them loses
nothing: CRS, transform, nodata, dtype, band count, values, and the Easy-EO
provenance fields survive a dataset -> DataArray -> dataset trip, and a
DataArray -> dataset -> DataArray trip returns an equivalent DataArray.
"""

import datetime as dt

import numpy as np
import pytest
import rasterio as rio
from rasterio.crs import CRS
from rasterio.transform import Affine

import eeo
from eeo import load_array

xr = pytest.importorskip("xarray", reason="needs the optional xarray extra")
rioxarray = pytest.importorskip("rioxarray", reason="needs the optional xarray extra")

UTM = CRS.from_epsg(32633)
GEO = CRS.from_epsg(4326)
UTM_TRANSFORM = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0)
# Deliberately awkward: a resolution whose half cannot be represented exactly,
# so a transform rebuilt from pixel centres would drift if it were rebuilt.
GEO_TRANSFORM = Affine(1 / 3600, 0.0, 11.123456789, 0.0, -1 / 3600, 46.987654321)


def _values(shape, dtype):
    """Return a deterministic gradient of the given shape and dtype."""
    return np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)


# Each case is a no-argument builder, so a test gets a fresh dataset and the
# matrix stays readable as a list of raster characteristics.
def _single_band_float32():
    return load_array(_values((1, 6, 8), "float32"), transform=UTM_TRANSFORM, crs=UTM)


def _multiband_named():
    return load_array(
        _values((4, 6, 8), "uint16"),
        transform=UTM_TRANSFORM,
        crs=UTM,
        band_names=["blue", "green", "red", "nir"],
    )


def _float_sentinel_nodata():
    array = _values((1, 6, 8), "float32")
    array[0, :2, :2] = -9999.0
    return load_array(array, transform=UTM_TRANSFORM, crs=UTM, nodata=-9999.0)


def _nan_nodata():
    array = _values((1, 6, 8), "float32")
    array[0, :2, :2] = np.nan
    return load_array(array, transform=UTM_TRANSFORM, crs=UTM, nodata=np.nan)


def _integer_nodata():
    array = _values((2, 6, 8), "int16")
    array[:, 0, 0] = -1
    return load_array(array, transform=UTM_TRANSFORM, crs=UTM, nodata=-1)


def _geographic_fine_resolution():
    return load_array(_values((1, 5, 7), "float64"), transform=GEO_TRANSFORM, crs=GEO)


def _unreferenced():
    return load_array(_values((1, 4, 4), "float32"), transform=Affine.identity(), crs=None)


def _rotated():
    return load_array(
        _values((1, 3, 4), "float32"),
        transform=UTM_TRANSFORM * Affine.rotation(30),
        crs=UTM,
    )


def _single_pixel():
    return load_array(_values((1, 1, 1), "float32"), transform=UTM_TRANSFORM, crs=UTM)


def _single_row():
    return load_array(_values((1, 1, 6), "float32"), transform=UTM_TRANSFORM, crs=UTM)


def _with_provenance():
    return load_array(
        _values((2, 4, 5), "float32"),
        transform=UTM_TRANSFORM,
        crs=UTM,
        band_names=["red", "nir"],
        timestamp=dt.datetime(2023, 9, 7, 10, 0, 31),
        attrs={"mission": "sentinel-2", "processing_level": "L2A"},
    )


def _rasterio_backed():
    return _multiband_named().to_rasterio()


CASES = {
    "single band float32": _single_band_float32,
    "multiband named uint16": _multiband_named,
    "float sentinel nodata": _float_sentinel_nodata,
    "nan nodata": _nan_nodata,
    "integer nodata": _integer_nodata,
    "geographic, fine resolution, float64": _geographic_fine_resolution,
    "no crs": _unreferenced,
    "rotated transform": _rotated,
    "single pixel": _single_pixel,
    "single row": _single_row,
    "timestamp and attrs": _with_provenance,
    "rasterio-backed": _rasterio_backed,
}


def _same_nodata(left, right) -> bool:
    """Compare two nodata values, treating NaN as equal to NaN.

    Works on NumPy scalars as well as Python ones: ``np.float32("nan")`` is not
    an instance of ``float``, so a plain isinstance check would miss it.
    """
    if left is None or right is None:
        return left is None and right is None
    left, right = float(left), float(right)
    if np.isnan(left) or np.isnan(right):
        return np.isnan(left) and np.isnan(right)
    return left == right


def assert_same_raster(original, restored):
    """Assert two datasets are the same raster, field by field."""
    assert restored.get_count() == original.get_count()
    assert restored.get_shape() == original.get_shape()
    assert restored.get_transform() == original.get_transform()
    assert restored.get_crs() == original.get_crs()
    assert np.dtype(restored.get_metadata()["dtype"]) == np.dtype(original.get_metadata()["dtype"])
    assert _same_nodata(restored.get_metadata()["nodata"], original.get_metadata()["nodata"]), (
        f"nodata {original.get_metadata()['nodata']!r} -> {restored.get_metadata()['nodata']!r}"
    )
    # assert_array_equal counts NaN as equal to NaN in the same position.
    np.testing.assert_array_equal(restored.read(), original.read())
    assert restored.band_names == original.band_names
    assert restored.timestamp == original.timestamp
    assert restored.attrs == original.attrs


# =======================================================  dataset first


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_dataset_survives_a_round_trip(case):
    ds = CASES[case]()

    assert_same_raster(ds, eeo.from_xarray(ds.to_xarray()))


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_round_trip_is_stable_when_repeated(case):
    """A second trip must change nothing a first trip did not."""
    once = eeo.from_xarray(CASES[case]().to_xarray())

    assert_same_raster(once, eeo.from_xarray(once.to_xarray()))


@pytest.mark.parametrize(
    "dtype", ["uint8", "int16", "uint16", "int32", "uint32", "float32", "float64"]
)
def test_dtype_survives_a_round_trip(dtype):
    ds = load_array(_values((2, 3, 4), dtype), transform=UTM_TRANSFORM, crs=UTM)

    restored = eeo.from_xarray(ds.to_xarray())

    assert np.dtype(restored.get_metadata()["dtype"]) == np.dtype(dtype)
    np.testing.assert_array_equal(restored.read(), ds.read())


@pytest.mark.parametrize(
    ("case", "expected_type"),
    [("float sentinel nodata", float), ("integer nodata", int)],
)
def test_nodata_comes_back_as_a_python_scalar(case, expected_type):
    """rioxarray reports nodata in the array's dtype; Easy-EO records a scalar."""
    restored = eeo.from_xarray(CASES[case]().to_xarray())

    nodata = restored.get_metadata()["nodata"]
    assert type(nodata) is expected_type
    assert nodata == CASES[case]().get_metadata()["nodata"]


def test_round_trip_through_a_geotiff_on_disk(tmp_path):
    """The whole point of the interop: hand data over, get it back, write it out."""
    ds = _with_provenance()

    ds.to_xarray().rio.to_raster(tmp_path / "out.tif")
    restored = eeo.load_raster(str(tmp_path / "out.tif"))

    assert restored.get_crs() == ds.get_crs()
    assert restored.get_transform() == ds.get_transform()
    assert restored.band_names == ds.band_names
    np.testing.assert_array_equal(restored.read(), ds.read())


def test_round_trip_keeps_the_dataset_chainable():
    ds = _multiband_named()

    ndvi = eeo.from_xarray(ds.to_xarray()).ndvi(red="red", nir="nir")

    np.testing.assert_allclose(ndvi.read(), ds.ndvi(red="red", nir="nir").read())


def test_a_string_crs_round_trips_to_the_same_crs():
    """load_array keeps a CRS as given; the trip back normalises it to a CRS object."""
    ds = load_array(_values((1, 3, 3), "float32"), transform=UTM_TRANSFORM, crs="EPSG:32633")

    restored = eeo.from_xarray(ds.to_xarray())

    assert restored.get_crs() == CRS.from_user_input(ds.get_crs())
    assert restored.get_crs().to_epsg() == 32633


def test_an_aware_timestamp_comes_back_as_the_same_instant_in_utc():
    """xarray datetimes carry no timezone, so the offset is normalised away."""
    aware = dt.datetime(2023, 9, 7, 12, 0, 31, tzinfo=dt.timezone(dt.timedelta(hours=2)))
    ds = load_array(
        _values((1, 3, 3), "float32"), transform=UTM_TRANSFORM, crs=UTM, timestamp=aware
    )

    restored = eeo.from_xarray(ds.to_xarray())

    assert restored.timestamp == dt.datetime(2023, 9, 7, 10, 0, 31)
    assert restored.timestamp.replace(tzinfo=dt.timezone.utc) == aware


# ====================================================  DataArray first


def _write_scene(path, *, count=2, nodata=-1.0, descriptions=("red", "nir")):
    """Write a small GeoTIFF so rioxarray can open a realistic DataArray."""
    array = _values((count, 6, 8), "float32")
    array[0, 0, 0] = nodata if nodata is not None else 0.0
    profile = {
        "driver": "GTiff",
        "height": 6,
        "width": 8,
        "count": count,
        "dtype": "float32",
        "crs": UTM,
        "transform": UTM_TRANSFORM,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rio.open(path, "w", **profile) as dst:
        dst.write(array)
        for idx, name in enumerate(descriptions[:count], start=1):
            dst.set_band_description(idx, name)
    return path


def test_dataarray_survives_a_round_trip(tmp_path):
    original = rioxarray.open_rasterio(_write_scene(tmp_path / "scene.tif"))

    restored = eeo.from_xarray(original).to_xarray()

    # Compares values, dims, and coordinates.
    xr.testing.assert_allclose(restored, original)
    assert restored.rio.crs == original.rio.crs
    assert restored.rio.transform() == original.rio.transform()
    assert _same_nodata(restored.rio.nodata, original.rio.nodata)
    assert restored.dtype == original.dtype
    assert restored.attrs["long_name"] == original.attrs["long_name"]


def test_dataarray_round_trip_keeps_the_reader_supplied_attributes(tmp_path):
    original = rioxarray.open_rasterio(_write_scene(tmp_path / "scene.tif"))
    assert "AREA_OR_POINT" in original.attrs

    restored = eeo.from_xarray(original).to_xarray()

    for key, value in original.attrs.items():
        assert restored.attrs[key] == value


def test_nan_masked_dataarray_survives_a_round_trip(tmp_path):
    """A mask_and_scale read carries its nodata as NaN in the values themselves."""
    original = rioxarray.open_rasterio(_write_scene(tmp_path / "masked.tif"), mask_and_scale=True)
    assert np.isnan(original.values).any()

    restored = eeo.from_xarray(original).to_xarray()

    xr.testing.assert_allclose(restored, original)
    assert np.isnan(restored.rio.nodata)


def test_a_two_dimensional_dataarray_comes_back_with_a_band_dimension(tmp_path):
    """A raster always has bands, so the 2-D form is normalised on the way back."""
    original = rioxarray.open_rasterio(_write_scene(tmp_path / "scene.tif")).sel(band=1)

    restored = eeo.from_xarray(original).to_xarray()

    assert original.dims == ("y", "x")
    assert restored.dims == ("band", "y", "x")
    assert restored.shape == (1, *original.shape)
    np.testing.assert_array_equal(restored.values[0], original.values)


def test_a_transposed_dataarray_comes_back_in_band_y_x_order(tmp_path):
    original = rioxarray.open_rasterio(_write_scene(tmp_path / "scene.tif"))

    restored = eeo.from_xarray(original.transpose("y", "x", "band")).to_xarray()

    assert restored.dims == ("band", "y", "x")
    xr.testing.assert_allclose(restored, original)


def test_a_labelled_band_coordinate_is_normalised_to_indices(tmp_path):
    """Band identity travels in long_name, so a labelled band axis is renumbered."""
    original = rioxarray.open_rasterio(_write_scene(tmp_path / "scene.tif")).assign_coords(
        band=["red", "nir"]
    )

    restored = eeo.from_xarray(original).to_xarray()

    assert restored.band.values.tolist() == [1, 2]
    # The names themselves are not lost: they travel in long_name.
    assert restored.attrs["long_name"] == ("red", "nir")
    np.testing.assert_array_equal(restored.values, original.values)


def test_a_sliced_dataarray_round_trips_to_its_own_extent(tmp_path):
    """The slice's coordinates, not the parent's geotransform, define the result."""
    original = rioxarray.open_rasterio(_write_scene(tmp_path / "scene.tif")).isel(
        y=slice(2, 5), x=slice(3, 8)
    )

    restored = eeo.from_xarray(original).to_xarray()

    xr.testing.assert_allclose(restored, original)
    assert restored.rio.transform() == original.rio.transform()
