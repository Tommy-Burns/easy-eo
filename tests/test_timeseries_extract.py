"""Sampling a series at a location (eeo/timeseries/extract.py).

The season fixtures give a stack whose values are known per timestep, so a
trajectory can be asserted value by value — including the timestep where the
sampled pixel is nodata, which must read as a gap rather than as a fill value.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform

import eeo
from eeo.core.exceptions import ValidationError
from eeo.timeseries import extract

UTC = dt.timezone.utc

# Centre of pixel (1, 1) on the conftest season grid: origin (500000, 4200000),
# 10 m pixels, so 1.5 pixels in from the top-left corner.
INSIDE = (500_015.0, 4_199_985.0)
# Centre of pixel (0, 0), the one the season stack leaves nodata twice.
GAP_PIXEL = (500_005.0, 4_199_995.0)


def test_a_trajectory_is_a_dataframe_indexed_by_time(season_series):
    table = season_series.extract_at(INSIDE)

    assert isinstance(table, pd.DataFrame)
    assert isinstance(table.index, pd.DatetimeIndex)
    assert table.index.name == "time"
    assert list(table.index) == season_series.timestamps
    assert list(table.columns) == ["red", "nir"]


def test_the_trajectory_holds_each_timestep_value(season_series):
    table = season_series.extract_at(INSIDE)

    assert list(table["red"]) == [1000.0, 900.0, 800.0, 900.0, 1000.0]
    assert list(table["nir"]) == [2000.0, 3000.0, 4000.0, 3000.0, 2000.0]


def test_values_are_floats_so_a_gap_can_be_nan(season_series):
    table = season_series.extract_at(GAP_PIXEL)

    assert table["red"].dtype == np.float64
    # Nodata at the third and fourth timesteps: a gap, not a zero.
    assert table["red"].isna().tolist() == [False, False, True, True, False]
    assert list(table["red"].dropna()) == [1000.0, 900.0, 1000.0]


def test_bands_can_be_selected(season_series):
    table = season_series.extract_at(INSIDE, bands=["nir"])

    assert list(table.columns) == ["nir"]
    assert list(table["nir"]) == [2000.0, 3000.0, 4000.0, 3000.0, 2000.0]


def test_an_unnamed_band_is_labelled_by_its_position():
    plain = eeo.load_array(
        np.full((1, 4, 4), 7, dtype="uint16"),
        transform=from_origin(500_000.0, 4_200_000.0, 10.0, 10.0),
        crs="EPSG:32633",
        timestamp=dt.datetime(2023, 5, 1, tzinfo=UTC),
    ).to_rasterio()

    table = eeo.time_series([plain]).extract_at(INSIDE)

    assert list(table.columns) == ["band_1"]
    assert list(table["band_1"]) == [7.0]


def test_the_sampled_point_is_recorded(season_series):
    table = season_series.extract_at(INSIDE)

    assert table.attrs["x"] == INSIDE[0]
    assert table.attrs["y"] == INSIDE[1]
    assert "32633" in table.attrs["crs"]


def test_coordinates_can_be_given_in_another_crs(season_series):
    lon, lat = (
        value[0]
        for value in warp_transform(season_series.crs, "EPSG:4326", [INSIDE[0]], [INSIDE[1]])
    )

    table = season_series.extract_at((lon, lat), crs="EPSG:4326")

    assert list(table["red"]) == [1000.0, 900.0, 800.0, 900.0, 1000.0]
    # The point is recorded in the series' CRS, whichever way it was given.
    assert table.attrs["x"] == pytest.approx(INSIDE[0], abs=0.5)


def test_a_point_outside_the_extent_says_where_the_extent_is(season_series):
    with pytest.raises(ValidationError, match="falls outside the series' extent"):
        season_series.extract_at((0.0, 0.0))


def test_a_point_just_past_the_last_pixel_is_outside(season_series):
    # The 4x4 grid ends at x = 500040; that easting is the first column beyond
    # it, which a bounds check on the corner coordinate alone would admit.
    with pytest.raises(ValidationError, match="falls outside"):
        season_series.extract_at((500_040.0, 4_199_985.0))


def test_coordinates_must_be_a_pair(season_series):
    with pytest.raises(ValidationError, match="exactly 2 values"):
        season_series.extract_at((500_015.0, 4_199_985.0, 12.0))


def test_an_unknown_band_is_refused(season_series):
    with pytest.raises(ValidationError, match="swir"):
        season_series.extract_at(INSIDE, bands=["swir"])


def test_an_empty_band_list_is_refused(season_series):
    with pytest.raises(ValidationError, match="bands is empty"):
        season_series.extract_at(INSIDE, bands=[])


def test_crs_is_refused_for_a_series_that_declares_none():
    unplaced = eeo.load_array(
        np.full((1, 4, 4), 7, dtype="uint16"),
        transform=from_origin(500_000.0, 4_200_000.0, 10.0, 10.0),
        crs=None,
        timestamp=dt.datetime(2023, 5, 1, tzinfo=UTC),
    )
    ts = eeo.time_series([unplaced])

    # There is nothing to transform the point onto.
    with pytest.raises(ValidationError, match="declares none"):
        ts.extract_at((11.1, 46.6), crs="EPSG:4326")


def test_an_index_trajectory_reads_as_a_season(season_series):
    ndvi = season_series.map(eeo.ndvi, red="red", nir="nir", name="ndvi")

    trajectory = ndvi.extract_at(INSIDE)

    assert list(trajectory.columns) == ["ndvi"]
    assert trajectory["ndvi"].tolist() == pytest.approx(
        [1 / 3, 0.538462, 2 / 3, 0.538462, 1 / 3], abs=1e-6
    )
    # The shape pandas understands: a resample, a rolling mean or a plot all
    # work from here without further conversion.
    assert trajectory["ndvi"].idxmax() == season_series.timestamps[2]


def test_extraction_is_also_a_plain_function(season_series):
    table = extract.extract_at(season_series, INSIDE, bands=["red"])

    assert list(table.columns) == ["red"]
