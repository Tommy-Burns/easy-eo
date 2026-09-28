"""Pairing two series timestep by timestep (EEOTimeSeries.map_with).

Change detection: two series of one place, zipped off by position and combined.
The claim that matters is that it is a *zip* and not a broadcast, so the values
are chosen to tell the two apart — a broadcast of the first partner would give
the same answer at every timestep, and a zip gives a different one at each.

The other half is the contrast with ``map``, which broadcasts a single operand.
Both spellings are exercised side by side so the difference is pinned down rather
than described.
"""

import datetime as dt

import numpy as np
import pytest
from rasterio.transform import from_origin

import eeo
from eeo.core.adapters import RasterioAdapter
from eeo.core.exceptions import ValidationError

UTC = dt.timezone.utc
CRS = "EPSG:32633"
TRANSFORM = from_origin(500_000.0, 4_200_000.0, 10.0, 10.0)

# Three timesteps a year apart in each series, so a pairing that used timestamps
# rather than positions would find nothing to pair at all.
BEFORE = ((2020, 100), (2021, 200), (2022, 300))
AFTER = ((2023, 110), (2024, 230), (2025, 360))


def scene(year, value, *, month=6, bands=1, names=None):
    array = np.full((bands, 4, 4), value, dtype="int32")
    return eeo.load_array(
        array,
        transform=TRANSFORM,
        crs=CRS,
        timestamp=dt.datetime(year, month, 1, tzinfo=UTC),
        band_names=names or [f"b{index + 1}" for index in range(bands)],
    )


@pytest.fixture
def before():
    ts = eeo.time_series([scene(year, value) for year, value in BEFORE])
    yield ts
    ts.close()


@pytest.fixture
def after():
    ts = eeo.time_series([scene(year, value) for year, value in AFTER])
    yield ts
    ts.close()


def first_pixels(series):
    return [float(ds.to_array()[0, 0, 0]) for ds in series]


# --------------------------------------------------------------------------
# It is a zip, not a broadcast
# --------------------------------------------------------------------------
def test_timesteps_are_paired_off_by_position(before, after):
    change = after.map_with(before, eeo.subtract)

    # 110 - 100, 230 - 200, 360 - 300: a different answer at each timestep,
    # which is only possible if each pair was taken in turn.
    assert first_pixels(change) == pytest.approx([10.0, 30.0, 60.0])


def test_the_receiver_is_the_operations_first_operand(before, after):
    # Which way round a difference comes out, spelled out: the series the method
    # is called on goes in first, so "after minus before" is after.map_with(...).
    assert first_pixels(after.map_with(before, eeo.subtract)) == pytest.approx([10.0, 30.0, 60.0])
    assert first_pixels(before.map_with(after, eeo.subtract)) == pytest.approx(
        [-10.0, -30.0, -60.0]
    )


def test_mapping_a_single_raster_broadcasts_it_instead(before, after):
    # The contrast the method exists for, asserted rather than described.
    broadcast = before.map(eeo.subtract, other=after[0])

    # Every timestep less the *same* raster: 100 - 110, 200 - 110, 300 - 110.
    assert first_pixels(broadcast) == pytest.approx([-10.0, 90.0, 190.0])


def test_pairing_ignores_the_timestamps(before, after):
    # The two series share no dates at all, which is the normal case for change
    # detection between epochs.
    assert set(before.timestamps).isdisjoint(after.timestamps)

    assert len(after.map_with(before, eeo.subtract)) == 3


def test_the_result_carries_this_series_timestamps(before, after):
    change = before.map_with(after, eeo.subtract)

    assert change.timestamps == before.timestamps


def test_each_result_records_the_date_it_was_paired_with(before, after):
    change = before.map_with(after, eeo.subtract)

    # Without this a 2020-vs-2023 difference would claim to be a 2020 raster and
    # say nothing about the other end of the comparison.
    assert [ds.attrs["paired_timestamp"] for ds in change] == after.timestamps


def test_neither_input_series_is_touched(before, after):
    before.map_with(after, eeo.subtract)

    assert first_pixels(before) == pytest.approx([100.0, 200.0, 300.0])
    assert first_pixels(after) == pytest.approx([110.0, 230.0, 360.0])
    assert "paired_timestamp" not in before[0].attrs


# --------------------------------------------------------------------------
# Which operations work
# --------------------------------------------------------------------------
def test_an_index_between_two_series(before, after):
    difference = after.map_with(before, eeo.normalized_difference, name="change")

    # (110 - 100) / (110 + 100), and so on.
    assert first_pixels(difference) == pytest.approx([10 / 210, 30 / 430, 60 / 660], abs=1e-6)
    assert difference.band_names == ["change"]


def test_the_operation_keeps_the_provenance_its_bound_call_would(before, after):
    # The 19.3 finding, still true with a second operand: a registered op must be
    # invoked through its bound method or band names are silently dropped.
    named = eeo.time_series([scene(year, value, names=["red"]) for year, value in BEFORE])
    partner = eeo.time_series([scene(year, value, names=["red"]) for year, value in AFTER])

    change = named.map_with(partner, eeo.subtract)

    assert change.band_names == ["red"]
    assert change[0].timestamp == named.timestamps[0]
    named.close()
    partner.close()


def test_a_function_of_your_own_takes_the_pair_in_order(before, after):
    ratio = before.map_with(after, lambda a, b: a.divide(b))

    assert first_pixels(ratio) == pytest.approx([100 / 110, 200 / 230, 300 / 360], abs=1e-6)


def test_keyword_arguments_reach_the_operation(before, after):
    difference = before.map_with(after, eeo.normalized_difference, name="ndc")

    assert difference.band_names == ["ndc"]


def test_a_paired_result_is_a_series_like_any_other(before, after):
    change = after.map_with(before, eeo.subtract)

    # Reducible, mappable, samplable — which is what makes "the typical change
    # over these three pairs" one more call.
    assert float(change.median().to_array()[0, 0, 0]) == pytest.approx(30.0)
    doubled = change.map(eeo.multiply, other=2)
    assert first_pixels(doubled) == pytest.approx([20.0, 60.0, 120.0])
    trajectory = change.extract_at((500_005.0, 4_199_995.0))
    assert list(trajectory["b1"]) == pytest.approx([10.0, 30.0, 60.0])


def test_pairing_works_across_multiple_bands(before, after):
    left = eeo.time_series([scene(year, value, bands=2) for year, value in BEFORE])
    right = eeo.time_series([scene(year, value, bands=2) for year, value in AFTER])

    change = right.map_with(left, eeo.subtract)

    assert change.band_count == 2
    assert [float(ds.to_array()[1, 0, 0]) for ds in change] == pytest.approx([10.0, 30.0, 60.0])
    left.close()
    right.close()


# --------------------------------------------------------------------------
# Streaming to disk
# --------------------------------------------------------------------------
def test_save_dir_writes_one_raster_per_pair(before, after, tmp_path):
    change = after.map_with(before, eeo.subtract, save_dir=tmp_path / "change")

    # Named by the receiving series' own acquisition times.
    written = sorted((tmp_path / "change").glob("*.tif"))
    assert [path.name for path in written] == [
        "0000_20230601T000000.tif",
        "0001_20240601T000000.tif",
        "0002_20250601T000000.tif",
    ]
    assert all(isinstance(ds._adapter, RasterioAdapter) for ds in change)
    assert first_pixels(change) == pytest.approx([10.0, 30.0, 60.0])
    change.close()


def test_save_dir_is_created_if_absent(before, after, tmp_path):
    target = tmp_path / "not" / "yet"

    change = before.map_with(after, eeo.subtract, save_dir=target)

    assert target.is_dir()
    assert len(change) == 3
    change.close()


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------
def test_a_single_dataset_names_the_broadcasting_call_instead(before, after):
    # The likeliest mistake, and the one worth answering with the right spelling
    # rather than a type name.
    with pytest.raises(ValidationError, match=r"ts\.map\(subtract, other=that_raster\)"):
        before.map_with(after[0], eeo.subtract)


def test_something_that_is_not_a_series_is_refused(before):
    with pytest.raises(ValidationError, match="another EEOTimeSeries; got list"):
        before.map_with([1, 2, 3], eeo.subtract)


def test_two_series_of_different_lengths_are_refused(before, after):
    with pytest.raises(ValidationError, match="this one has 3 and the other has 2"):
        before.map_with(after[:2], eeo.subtract)


def test_the_length_refusal_names_the_fix(before, after):
    with pytest.raises(ValidationError, match="Slice them to a common length"):
        before.map_with(after[1:], eeo.subtract)


def test_the_name_of_an_operation_is_refused(before, after):
    with pytest.raises(ValidationError, match="pass eeo.subtract rather than 'subtract'"):
        before.map_with(after, "subtract")


def test_something_uncallable_is_refused(before, after):
    with pytest.raises(ValidationError, match="taking two datasets"):
        before.map_with(after, 7)


def test_an_operation_returning_a_value_is_refused(before, after):
    with pytest.raises(ValidationError, match="must be an EEORasterDataset"):
        before.map_with(after, lambda a, b: float(a.to_array().mean()))


def test_a_failure_part_way_closes_what_it_opened(before, after):
    calls = []

    def fails_on_the_third(a, b):
        calls.append(len(calls))
        if len(calls) == 3:
            raise RuntimeError("no")
        return a.subtract(b)

    with pytest.raises(RuntimeError):
        before.map_with(after, fails_on_the_third)

    # The two results built before the failure are released, not leaked.
    assert len(calls) == 3
