"""Grouping a series in time (eeo/timeseries/binning.py).

Two halves: the grouping is arithmetic on timestamps and is asserted by labels
and sizes, and the reduction within each period is asserted against values
chosen so every monthly statistic is a round number.

A note on periods used here. pandas 2.2 renamed the end-of-period aliases
("M" to "ME", "Q" to "QE", "Y" to "YE"), and Easy-EO supports pandas on both
sides of that change — so a test naming either spelling would pass on one
installation and fail on the other. Every test here uses the start-of-period
aliases, which were not renamed, and the rename guidance is asserted against the
message builder directly instead.
"""

import datetime as dt

import numpy as np
import pytest
from rasterio.transform import from_origin

import eeo
from eeo.core.adapters import RasterioAdapter
from eeo.core.exceptions import ValidationError
from eeo.timeseries.binning import TemporalBins, _explain
from eeo.timeseries.core import EEOTimeSeries

UTC = dt.timezone.utc
CRS = "EPSG:32633"
TRANSFORM = from_origin(500_000.0, 4_200_000.0, 10.0, 10.0)

CLEAR = 4  # SCLClass.VEGETATION
CLOUD = 9  # SCLClass.CLOUD_HIGH_PROBABILITY

# Two acquisitions in March, one in April, none in May, two in June: enough for
# a monthly grouping to have an empty period to drop and periods of two
# different sizes.
SEASON = ((3, 5, 10), (3, 20, 30), (4, 2, 50), (6, 11, 70), (6, 25, 90))


def scene(month, day, value, *, bands=None, names=None):
    array = (
        np.full((1, 4, 4), value, dtype="uint16")
        if bands is None
        else np.stack([np.asarray(band, dtype="uint16") for band in bands])
    )
    return eeo.load_array(
        array,
        transform=TRANSFORM,
        crs=CRS,
        # Declared so mask_clouds has a value to write into a masked pixel; 0 is
        # what Sentinel-2 L2A and Landsat Collection 2 both use.
        nodata=0,
        timestamp=dt.datetime(2023, month, day, tzinfo=UTC),
        band_names=names or ["red"],
    )


@pytest.fixture
def season():
    ts = eeo.time_series([scene(month, day, value) for month, day, value in SEASON])
    yield ts
    ts.close()


def s2_scene(month, day, red, nir, *, clouded=()):
    """A Sentinel-2-like scene whose SCL band flags the given pixels."""
    scl = np.full((4, 4), CLEAR, dtype="uint16")
    for row, col in clouded:
        scl[row, col] = CLOUD
    return scene(
        month,
        day,
        0,
        bands=[np.full((4, 4), red), np.full((4, 4), nir), scl],
        names=["red", "nir", "scl"],
    )


@pytest.fixture
def clouded():
    """Two months of two acquisitions each, the cloud moving between them."""
    ts = eeo.time_series(
        [
            s2_scene(3, 5, 1000, 2000, clouded=[(0, 0), (3, 3)]),
            s2_scene(3, 20, 1200, 2200, clouded=[(1, 1), (3, 3)]),
            s2_scene(4, 2, 800, 4000, clouded=[(0, 0), (3, 3)]),
            s2_scene(4, 25, 600, 4400, clouded=[(1, 1), (3, 3)]),
        ]
    )
    yield ts
    ts.close()


# --------------------------------------------------------------------------
# The grouping
# --------------------------------------------------------------------------
def test_monthly_periods_group_by_calendar_month(season):
    periods = season.resample_time("MS")

    assert len(periods) == 3
    assert [label.date().isoformat() for label in periods.labels] == [
        "2023-03-01",
        "2023-04-01",
        "2023-06-01",
    ]
    assert [len(period) for period in periods] == [2, 1, 2]


def test_a_period_holding_no_acquisition_is_dropped(season):
    # May is empty, and a series cannot hold a timestep with no raster behind it.
    assert dt.date(2023, 5, 1) not in [label.date() for label in season.resample_time("MS").labels]


def test_a_quarterly_grouping_splits_the_same_series_differently(season):
    periods = season.resample_time("QS")

    assert [label.date().isoformat() for label in periods.labels] == ["2023-01-01", "2023-04-01"]
    assert [len(period) for period in periods] == [2, 3]


def test_a_weekly_grouping_puts_every_acquisition_in_its_own_period(season):
    # The five acquisitions are all more than a week apart.
    assert [len(period) for period in season.resample_time("7D")] == [1, 1, 1, 1, 1]


def test_an_anchored_alias_reaches_pandas(season):
    # Nothing here reimplements pandas' vocabulary, so an alias Easy-EO has
    # never heard of still works.
    assert len(season.resample_time("W-MON")) == 5


def test_a_period_is_a_series_of_its_own_timesteps(season):
    march = next(iter(season.resample_time("MS")))

    assert isinstance(march, EEOTimeSeries)
    assert [stamp.day for stamp in march.timestamps] == [5, 20]
    assert march.band_names == ["red"]
    assert march.shape == season.shape


def test_the_grouping_leaves_the_series_alone(season):
    season.resample_time("MS").median()

    assert len(season) == 5


def test_the_repr_says_how_many_periods_and_how_full(season):
    text = repr(season.resample_time("MS"))

    assert "3 periods of 'MS'" in text
    assert "2023-03-01 to 2023-06-01" in text
    assert "2, 1, 2 timesteps each" in text


def test_the_grouping_is_also_a_plain_class(season):
    assert len(TemporalBins(season, "MS")) == 3


# --------------------------------------------------------------------------
# Reducing within each period
# --------------------------------------------------------------------------
def test_the_monthly_median_is_the_median_within_each_month(season):
    monthly = season.resample_time("MS").median()

    # March (10, 30), April (50), June (70, 90).
    assert [float(ds.to_array()[0, 0, 0]) for ds in monthly] == pytest.approx([20.0, 50.0, 80.0])


def test_the_monthly_mean_averages_within_each_month(season):
    monthly = season.resample_time("MS").mean()

    assert [float(ds.to_array()[0, 0, 0]) for ds in monthly] == pytest.approx([20.0, 50.0, 80.0])


def test_the_monthly_extremes_are_taken_within_each_month(season):
    lowest = season.resample_time("MS").min()
    highest = season.resample_time("MS").max()

    assert [int(ds.to_array()[0, 0, 0]) for ds in lowest] == [10, 50, 70]
    assert [int(ds.to_array()[0, 0, 0]) for ds in highest] == [30, 50, 90]


def test_each_result_is_stamped_with_its_period(season):
    monthly = season.resample_time("MS").median()

    assert [stamp.date().isoformat() for stamp in monthly.timestamps] == [
        "2023-03-01",
        "2023-04-01",
        "2023-06-01",
    ]
    # On the datasets too, not only on the series that holds them.
    assert [ds.timestamp for ds in monthly] == monthly.timestamps


def test_a_result_records_the_span_it_actually_covers(season):
    march = season.resample_time("MS").median()[0]

    # The label places it in time; attrs say what really went into it, which is
    # the honest answer to "when was this composite acquired".
    assert march.attrs["timesteps"] == 2
    assert march.attrs["time_start"] == dt.datetime(2023, 3, 5, tzinfo=UTC)
    assert march.attrs["time_end"] == dt.datetime(2023, 3, 20, tzinfo=UTC)
    assert march.attrs["temporal_reduction"] == "median"
    assert march.attrs["temporal_bin"] == "MS"


def test_the_dtype_policy_carries_over_to_a_binned_reduction(season):
    periods = season.resample_time("MS")

    assert periods.median()[0].get_metadata()["dtype"] == "float32"
    assert periods.mean()[0].get_metadata()["dtype"] == "float32"
    assert periods.min()[0].get_metadata()["dtype"] == "uint16"
    assert periods.max()[0].get_metadata()["dtype"] == "uint16"


def test_a_binned_reduction_is_a_series_like_any_other(season):
    monthly = season.resample_time("MS").median()

    # Reducible again — the reason the reducers return a plain dataset and the
    # series stays immutable.
    assert float(monthly.max().to_array()[0, 0, 0]) == pytest.approx(80.0)
    # Mappable, sliceable, samplable. Each intermediate is bound rather than
    # chained: a derived series closes the datasets it shares when it is
    # collected, so a discarded temporary takes the parent's rasters with it.
    doubled = monthly.map(eeo.multiply, other=2)
    assert len(doubled) == 3
    later = monthly[1:]
    assert len(later) == 2
    trajectory = monthly.extract_at((500_005.0, 4_199_995.0))
    assert list(trajectory["red"]) == pytest.approx([20.0, 50.0, 80.0])
    assert monthly.band_names == ["red"]
    assert monthly.transform == season.transform


def test_a_series_that_fits_in_one_period_reduces_to_one_timestep(season):
    yearly = season.resample_time("YS").median()

    assert len(yearly) == 1
    assert yearly.timestamps[0].date() == dt.date(2023, 1, 1)
    assert float(yearly[0].to_array()[0, 0, 0]) == pytest.approx(50.0)


# --------------------------------------------------------------------------
# Streaming the results to disk
# --------------------------------------------------------------------------
def test_save_dir_writes_one_raster_per_period(season, tmp_path):
    monthly = season.resample_time("MS").median(save_dir=tmp_path / "monthly")

    written = sorted((tmp_path / "monthly").glob("*.tif"))
    assert [path.name for path in written] == [
        "0000_20230301T000000.tif",
        "0001_20230401T000000.tif",
        "0002_20230601T000000.tif",
    ]
    # And the series reads those files rather than holding the results.
    assert all(isinstance(ds._adapter, RasterioAdapter) for ds in monthly)
    assert [float(ds.to_array()[0, 0, 0]) for ds in monthly] == pytest.approx([20.0, 50.0, 80.0])
    monthly.close()


def test_save_dir_is_created_if_absent(season, tmp_path):
    target = tmp_path / "does" / "not" / "exist"

    monthly = season.resample_time("MS").median(save_dir=target)

    assert target.is_dir()
    assert len(monthly) == 3
    monthly.close()


def test_a_saved_result_keeps_its_period_as_its_timestamp(season, tmp_path):
    monthly = season.resample_time("MS").max(save_dir=tmp_path / "peaks")

    assert [ds.timestamp for ds in monthly] == monthly.timestamps
    assert [int(ds.to_array()[0, 0, 0]) for ds in monthly] == [30, 50, 90]
    monthly.close()


# --------------------------------------------------------------------------
# A composite per period
# --------------------------------------------------------------------------
def test_a_monthly_composite_leaves_the_quality_band_out(clouded):
    monthly = clouded.resample_time("MS").composite()

    assert len(monthly) == 2
    assert monthly.band_names == ["red", "nir"]


def test_a_monthly_composite_masks_within_its_own_month(clouded):
    monthly = clouded.resample_time("MS").composite()

    march, april = (ds.to_array() for ds in monthly)
    # (0, 0) is clouded at the first acquisition of each month, so each month's
    # value comes from its second — not from the other month's.
    assert march[0, 0, 0] == pytest.approx(1200.0)
    assert april[0, 0, 0] == pytest.approx(600.0)
    # A pixel nothing in the month saw stays a hole.
    assert np.isnan(march[0, 3, 3])
    assert np.isnan(april[0, 3, 3])


def test_a_clear_pixel_is_the_months_median(clouded):
    monthly = clouded.resample_time("MS").composite()

    march = monthly[0].to_array()
    assert march[0, 0, 1] == pytest.approx(1100.0)
    assert march[1, 0, 1] == pytest.approx(2100.0)


def test_composite_arguments_reach_mask_clouds(clouded):
    # classes= is mask_clouds' own argument; naming a class that flags nothing
    # leaves every pixel in, which is how we can see it arrived — the clouded
    # pixel (0, 0) is then averaged in rather than masked away.
    monthly = clouded.resample_time("MS").composite(classes=[11])

    assert monthly[0].to_array()[0, 0, 0] == pytest.approx(1100.0)


def test_a_binned_composite_can_be_streamed_to_disk(clouded, tmp_path):
    monthly = clouded.resample_time("MS").composite(save_dir=tmp_path / "clear")

    assert len(sorted((tmp_path / "clear").glob("*.tif"))) == 2
    assert monthly.band_names == ["red", "nir"]
    monthly.close()


def test_a_binned_composite_refuses_save_path(clouded):
    with pytest.raises(ValidationError, match="save_dir= instead"):
        clouded.resample_time("MS").composite(save_path="composite.tif")


def test_a_series_with_no_quality_band_is_refused_before_anything_is_read(season):
    with pytest.raises(ValidationError, match="no quality band"):
        season.resample_time("MS").composite()


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------
def test_an_unrecognised_period_is_refused(season):
    with pytest.raises(ValidationError, match="is not a period pandas recognises"):
        season.resample_time("bogus")


def test_the_refusal_names_the_aliases_that_work(season):
    with pytest.raises(ValidationError, match="'MS' a calendar month"):
        season.resample_time("bogus")


@pytest.mark.parametrize("freq", ["", "   ", None, 7])
def test_a_period_must_be_a_period_string(season, freq):
    with pytest.raises(ValidationError, match="not a period"):
        season.resample_time(freq)


def test_the_refusal_explains_an_old_alias_on_a_new_pandas():
    # Asserted against the message builder, not through pandas: whether "M" is
    # accepted depends on the installed pandas, and both are supported.
    message = _explain("1M", "Invalid frequency: 1M")

    assert "renamed 'M' to 'ME'" in message
    assert "were not renamed" in message


def test_the_refusal_explains_a_new_alias_on_an_old_pandas():
    message = _explain("ME", "Invalid frequency: ME")

    assert "needs pandas 2.2 or newer" in message
    assert "spelled 'M'" in message


@pytest.mark.parametrize(("freq", "expected"), [("Q", "QE"), ("2Y", "YE"), ("A", "YE")])
def test_every_renamed_alias_is_explained(freq, expected):
    assert f"to {expected!r}" in _explain(freq, "Invalid frequency")


def test_an_alias_that_was_never_renamed_gets_no_rename_hint():
    assert "renamed" not in _explain("bogus", "Invalid frequency: bogus")
