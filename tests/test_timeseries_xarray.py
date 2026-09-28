"""Stacking a series into a DataArray (EEOTimeSeries.to_xarray).

The single-raster conversion is covered in test_xarray_interop.py and
test_xarray_roundtrip.py. What is checked here is what the time dimension adds:
that ``time`` is a real indexed dimension in the right order, that the axes are
oriented the way the dims claim, and that a stack does not inherit one
timestep's provenance as though it described all of them.

The orientation check is the load-bearing one — a transposed or mis-ordered
stack would still have the right shape, so it is asserted against the series'
own reducer rather than against a shape.
"""

import datetime as dt

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin

import eeo

xr = pytest.importorskip("xarray", reason="needs the optional xarray extra")
pytest.importorskip("rioxarray", reason="needs the optional xarray extra")

UTC = dt.timezone.utc
UTM = CRS.from_epsg(32633)
TRANSFORM = from_origin(500_000.0, 4_200_000.0, 10.0, 10.0)
MONTHS = (3, 4, 5)


def scene(month, *, value=None, attrs=None, names=("red", "nir"), nodata=0):
    fill = month if value is None else value
    return eeo.load_array(
        np.full((len(names), 4, 4), fill, dtype="uint16"),
        transform=TRANSFORM,
        crs=UTM,
        nodata=nodata,
        band_names=list(names),
        timestamp=dt.datetime(2023, month, 1, tzinfo=UTC),
        attrs=attrs,
    )


@pytest.fixture
def series():
    ts = eeo.time_series([scene(month) for month in MONTHS])
    yield ts
    ts.close()


# --------------------------------------------------------------------------
# The shape of the result
# --------------------------------------------------------------------------
def test_time_is_the_leading_dimension(series):
    da = series.to_xarray()

    assert da.dims == ("time", "band", "y", "x")
    assert da.shape == (3, 2, 4, 4)


def test_the_time_coordinate_holds_the_acquisition_times_oldest_first(series):
    da = series.to_xarray()

    assert [str(value)[:10] for value in da.coords["time"].values] == [
        "2023-03-01",
        "2023-04-01",
        "2023-05-01",
    ]


def test_time_is_indexed_so_xarray_can_select_on_it(series):
    da = series.to_xarray()

    # The point of a real dimension rather than a bare axis.
    assert da.sel(time="2023-04-01").shape == (2, 4, 4)
    assert float(da.sel(time="2023-04-01")[0, 0, 0]) == 4.0


def test_the_spatial_and_band_coordinates_survive(series):
    da = series.to_xarray()

    assert list(da.coords["band"].values) == [1, 2]
    assert da.coords["x"].values[0] == pytest.approx(500_005.0)
    assert da.coords["y"].values[0] == pytest.approx(4_199_995.0)


def test_the_result_is_georeferenced_not_merely_shaped_like_a_raster(series):
    da = series.to_xarray()

    assert da.rio.crs == UTM
    assert da.rio.transform() == TRANSFORM
    assert da.rio.nodata == 0
    assert "spatial_ref" in da.coords


def test_band_names_are_carried_as_long_name(series):
    assert series.to_xarray().attrs["long_name"] == ("red", "nir")


def test_a_single_timestep_series_still_gets_a_time_dimension():
    ts = eeo.time_series([scene(6)])

    da = ts.to_xarray()

    assert da.dims == ("time", "band", "y", "x")
    assert da.shape == (1, 2, 4, 4)
    ts.close()


# --------------------------------------------------------------------------
# The axes mean what the dims say
# --------------------------------------------------------------------------
def test_every_timestep_keeps_its_own_values(series):
    da = series.to_xarray()

    assert [float(da[index, 0, 0, 0]) for index in range(3)] == [3.0, 4.0, 5.0]


def test_reducing_over_time_in_xarray_matches_the_series_own_reducer(series):
    # The orientation check: a stack with time and band transposed would have a
    # plausible shape and the wrong answer here.
    da = series.to_xarray()

    np.testing.assert_allclose(da.mean(dim="time").values, series.mean().to_array())
    np.testing.assert_allclose(da.max(dim="time").values, series.max().to_array())


def test_a_timestep_converts_back_to_an_equivalent_dataset(series):
    da = series.to_xarray()

    restored = eeo.from_xarray(da.isel(time=1))

    assert restored.get_crs() == UTM
    assert restored.get_transform() == TRANSFORM
    np.testing.assert_array_equal(restored.to_array(), series[1].to_array())


def test_the_spatial_axes_are_not_transposed():
    # A non-square raster is the only way to catch y and x swapped.
    tall = eeo.load_array(
        np.arange(2 * 6 * 3, dtype="uint16").reshape(2, 6, 3),
        transform=TRANSFORM,
        crs=UTM,
        band_names=["red", "nir"],
        timestamp=dt.datetime(2023, 3, 1, tzinfo=UTC),
    )
    ts = eeo.time_series([tall])

    da = ts.to_xarray()

    assert da.shape == (1, 2, 6, 3)
    assert da.sizes["y"] == 6
    assert da.sizes["x"] == 3
    ts.close()


# --------------------------------------------------------------------------
# Whose timestamps, and whose attrs
# --------------------------------------------------------------------------
def test_the_series_timestamps_win_over_the_datasets_own():
    # A series can be handed timestamps its datasets do not carry, and those are
    # the authoritative ones — a plain concat of the timesteps would produce a
    # time dimension with no coordinate at all here.
    undated = [
        eeo.load_array(np.full((1, 4, 4), value, dtype="uint16"), transform=TRANSFORM, crs=UTM)
        for value in (1, 2)
    ]
    ts = eeo.time_series(undated, timestamps=[dt.datetime(2019, 7, 4), dt.datetime(2020, 7, 4)])

    da = ts.to_xarray()

    assert [str(value)[:10] for value in da.coords["time"].values] == [
        "2019-07-04",
        "2020-07-04",
    ]
    ts.close()


def test_timezones_become_naive_utc():
    # datetime64 holds no timezone, so an aware timestamp is converted rather
    # than truncated: 09:00 in UTC+2 is 07:00 UTC.
    berlin = dt.timezone(dt.timedelta(hours=2))
    ts = eeo.time_series(
        [
            eeo.load_array(
                np.ones((1, 4, 4), dtype="uint16"),
                transform=TRANSFORM,
                crs=UTM,
                timestamp=dt.datetime(2023, 6, 1, 9, 0, tzinfo=berlin),
            )
        ]
    )

    assert str(ts.to_xarray().coords["time"].values[0]).startswith("2023-06-01T07:00")
    ts.close()


def test_an_attr_every_timestep_agrees_on_is_kept():
    ts = eeo.time_series([scene(month, attrs={"mission": "Sentinel-2"}) for month in MONTHS])

    assert ts.to_xarray().attrs["mission"] == "Sentinel-2"
    ts.close()


def test_an_attr_the_timesteps_disagree_on_is_dropped():
    # xarray's own rule would keep the first timestep's, labelling a three-scene
    # stack with one scene's item id.
    ts = eeo.time_series([scene(month, attrs={"stac_item_id": f"S2A_{month}"}) for month in MONTHS])

    assert "stac_item_id" not in ts.to_xarray().attrs
    ts.close()


def test_an_attr_only_one_timestep_carries_is_dropped():
    ts = eeo.time_series(
        [
            scene(3, attrs={"mission": "Sentinel-2", "processed_at": "2023-03-02"}),
            scene(4, attrs={"mission": "Sentinel-2"}),
        ]
    )

    attrs = ts.to_xarray().attrs
    assert attrs["mission"] == "Sentinel-2"
    assert "processed_at" not in attrs
    ts.close()


def test_the_georeferencing_attrs_are_kept_from_the_first_timestep():
    # Dropping _FillValue because two timesteps declare different nodata would
    # lose the declaration altogether, which is worse than taking the first's.
    ts = eeo.time_series([scene(3, nodata=0), scene(4, nodata=65535)])

    assert ts.to_xarray().attrs["_FillValue"] == 0
    ts.close()


def test_a_band_name_missing_at_one_timestep_does_not_lose_the_names():
    # Band names may be absent at a timestep without conflicting, and long_name
    # is georeferencing rather than provenance, so the reference timestep's wins
    # — the same rule as EEOTimeSeries.band_names.
    ts = eeo.time_series([scene(3, names=("red", "nir")), scene(4, names=("red", "nir"))])

    assert ts.to_xarray().attrs["long_name"] == ("red", "nir")
    ts.close()


# --------------------------------------------------------------------------
# What it composes with
# --------------------------------------------------------------------------
def test_a_mapped_series_converts_too(series):
    ndvi = series.map(eeo.ndvi, red="red", nir="nir", name="ndvi")

    da = ndvi.to_xarray()

    assert da.dims == ("time", "band", "y", "x")
    assert da.sizes["band"] == 1
    assert da.attrs["long_name"] == "ndvi"
    ndvi.close()


def test_bands_can_become_named_variables(series):
    # The shape xarray users usually want, and the reason band names are carried.
    dataset = series.to_xarray().to_dataset(dim="band")

    assert sorted(dataset.data_vars) == [1, 2]
    assert dataset[1].dims == ("time", "y", "x")


def test_a_binned_series_converts_to_a_monthly_stack(series):
    monthly = series.resample_time("MS").median()

    da = monthly.to_xarray()

    assert da.sizes["time"] == 3
    assert [str(value)[:10] for value in da.coords["time"].values] == [
        "2023-03-01",
        "2023-04-01",
        "2023-05-01",
    ]
    monthly.close()
