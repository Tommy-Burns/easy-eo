"""Construction and ordering of EEOTimeSeries (eeo/timeseries/core.py).

Covers construction: the three paths (datasets, a STAC
search result, and a folder of rasters), chronological ordering, the timestamp
requirement, the sequence protocol, and the scene cache.

The STAC metadata is faked and its assets are local GeoTIFFs, following
tests/test_stac_load.py, so the catalog path runs end to end without the
network.
"""

import datetime as dt
from pathlib import Path

import numpy as np
import pytest
import rasterio as rio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

import eeo
from eeo.core.adapters import RasterioAdapter, XarrayAdapter
from eeo.core.exceptions import AlignmentError, CRSMismatchError, ValidationError
from eeo.timeseries.core import EEOTimeSeries

UTC = dt.timezone.utc

SCENE_CRS = "EPSG:32633"
SCENE_ORIGIN = (500000.0, 5000000.0)
SCENE_SIZE = 60
SCENE_RES = 10.0


def transform():
    return from_origin(SCENE_ORIGIN[0], SCENE_ORIGIN[1], SCENE_RES, SCENE_RES)


def scene(month, *, value=None, size=4):
    """An in-memory single-band scene dated the first of ``month`` in 2023."""
    fill = month if value is None else value
    return eeo.load_array(
        np.full((size, size), fill, dtype="uint16"),
        transform=transform(),
        crs=SCENE_CRS,
        timestamp=dt.datetime(2023, month, 1, tzinfo=UTC),
    )


# -------------
# From datasets
# -------------
def test_datasets_are_sorted_oldest_first():
    ts = eeo.time_series([scene(7), scene(2), scene(4)])

    assert [stamp.month for stamp in ts.timestamps] == [2, 4, 7]
    assert [int(ds.to_array().flat[0]) for ds in ts] == [2, 4, 7]


def test_len_indexing_and_iteration():
    ts = eeo.time_series([scene(1), scene(2), scene(3)])

    assert len(ts) == 3
    assert ts[0].timestamp.month == 1
    assert ts[-1].timestamp.month == 3
    assert [ds.timestamp.month for ds in ts] == [1, 2, 3]


def test_slicing_returns_a_series_sharing_its_datasets():
    ts = eeo.time_series([scene(1), scene(2), scene(3)])
    sliced = ts[1:]

    assert isinstance(sliced, EEOTimeSeries)
    assert [stamp.month for stamp in sliced.timestamps] == [2, 3]
    assert sliced[0] is ts[1]


def test_an_empty_slice_is_refused_rather_than_returned():
    ts = eeo.time_series([scene(1), scene(2)])

    with pytest.raises(ValidationError, match="at least one dataset"):
        ts[2:]


def test_explicit_timestamps_override_without_mutating_the_datasets():
    scenes = [scene(1), scene(2)]
    stamps = [dt.datetime(2020, 5, 4, tzinfo=UTC), dt.datetime(2020, 5, 5, tzinfo=UTC)]

    ts = eeo.time_series(scenes, timestamps=stamps)

    assert ts.timestamps == stamps
    # The caller's datasets keep their own timestamps.
    assert [ds.timestamp.year for ds in scenes] == [2023, 2023]


def test_naive_timestamps_are_read_as_utc():
    naive = eeo.load_array(
        np.zeros((4, 4), dtype="uint16"),
        transform=transform(),
        crs=SCENE_CRS,
        timestamp=dt.datetime(2023, 3, 1),
    )
    # Mixing a naive timestamp with an aware one must not raise on sorting.
    ts = eeo.time_series([naive, scene(1)])

    assert [stamp.tzinfo for stamp in ts.timestamps] == [UTC, UTC]
    assert [stamp.month for stamp in ts.timestamps] == [1, 3]


def test_series_metadata_describes_the_grid():
    ts = eeo.time_series([scene(1), scene(2)])

    assert ts.shape == (4, 4)
    assert ts.band_count == 1
    assert ts.band_names == [None]
    assert ts.transform == transform()


def test_repr_names_the_count_and_span():
    text = repr(eeo.time_series([scene(4), scene(9)]))

    assert "2 timesteps" in text
    assert "2023-04-01 to 2023-09-01" in text


# --------
# Refusals
# --------
def test_empty_series_is_refused():
    with pytest.raises(ValidationError, match="at least one scene"):
        eeo.time_series([])


def test_a_timestep_without_a_timestamp_is_refused():
    undated = eeo.load_array(np.zeros((4, 4), dtype="uint16"), transform=transform(), crs=SCENE_CRS)

    with pytest.raises(ValidationError, match="carries no timestamp"):
        eeo.time_series([scene(1), undated])


def test_timestamps_length_must_match():
    with pytest.raises(ValidationError, match="one timestamp per dataset"):
        eeo.time_series([scene(1), scene(2)], timestamps=[dt.datetime(2020, 1, 1)])


def test_non_datasets_are_refused():
    with pytest.raises(ValidationError, match="STACItems or from EEORasterDatasets"):
        eeo.time_series([scene(1), "not a dataset"])


def test_something_that_is_not_a_collection_is_refused():
    with pytest.raises(ValidationError, match="got int"):
        eeo.time_series(7)


def test_assets_are_refused_for_loaded_datasets():
    with pytest.raises(ValidationError, match="already\n?\\s*loaded"):
        eeo.time_series([scene(1)], assets=["B04"])


def test_timestamps_entries_must_be_datetimes():
    with pytest.raises(ValidationError, match="must be a datetime"):
        eeo.time_series([scene(1)], timestamps=["2023-01-01"])


# ----------------------------------
# Two timesteps at one acquisition
# ----------------------------------
def processed(month, *, value, baseline=None, at=None):
    """A scene dated the first of ``month``, carrying processing provenance."""
    attrs = {"mission": "sentinel-2"}
    if baseline is not None:
        attrs["processing_baseline"] = baseline
    if at is not None:
        attrs["processing:datetime"] = at
    return eeo.load_array(
        np.full((4, 4), value, dtype="uint16"),
        transform=transform(),
        crs=SCENE_CRS,
        timestamp=dt.datetime(2023, month, 1, tzinfo=UTC),
        attrs=attrs,
    )


def test_two_timesteps_at_one_moment_warn():
    with pytest.warns(UserWarning, match="appear more than once"):
        eeo.time_series([scene(6, value=10), scene(6, value=20)])


def test_the_duplicate_warning_names_both_causes_and_both_fixes():
    with pytest.warns(UserWarning, match="deduplicate") as caught:
        eeo.time_series([scene(6, value=10), scene(6, value=20)])

    message = str(caught[0].message)
    assert "stac_search(...).deduplicate()" in message
    assert "mosaic" in message
    # And it says how many, so a long series is diagnosable from the warning.
    assert "1 acquisition time(s)" in message


def test_a_series_without_duplicates_is_quiet(recwarn):
    eeo.time_series([scene(3), scene(4), scene(5)])

    assert [w for w in recwarn if "more than once" in str(w.message)] == []


def test_duplicate_timestamps_keep_their_arrival_order():
    first, second = scene(6, value=10), scene(6, value=20)

    with pytest.warns(UserWarning, match="appear more than once"):
        ts = eeo.time_series([first, second])

    assert ts[0] is first
    assert ts[1] is second


# ----------------------------------
# Deduplicating a series
# ----------------------------------
def test_the_higher_baseline_survives():
    older = processed(6, value=10, baseline="05.00")
    newer = processed(6, value=20, baseline="05.11")

    with pytest.warns(UserWarning, match="appear more than once"):
        ts = eeo.time_series([older, newer])

    deduplicated = ts.deduplicate()

    assert len(deduplicated) == 1
    assert deduplicated[0] is newer


def test_the_processing_time_decides_where_there_is_no_baseline():
    first = processed(6, value=10, at="2023-06-02T00:00:00Z")
    later = processed(6, value=20, at="2024-06-02T00:00:00Z")

    with pytest.warns(UserWarning, match="appear more than once"):
        ts = eeo.time_series([later, first])

    assert ts.deduplicate()[0] is later


def test_the_first_survives_when_nothing_distinguishes_them():
    first, second = scene(6, value=10), scene(6, value=20)

    with pytest.warns(UserWarning, match="appear more than once"):
        ts = eeo.time_series([first, second])

    assert ts.deduplicate()[0] is first


def test_distinct_acquisitions_all_survive():
    ts = eeo.time_series([scene(3), scene(4), scene(5)])

    assert len(ts.deduplicate()) == 3


def test_a_series_with_nothing_to_deduplicate_is_returned_as_it_is():
    ts = eeo.time_series([scene(3), scene(4)])

    assert ts.deduplicate() is ts


def test_deduplicating_leaves_the_original_series_alone():
    with pytest.warns(UserWarning, match="appear more than once"):
        ts = eeo.time_series([scene(6, value=10), scene(6, value=20), scene(7)])

    ts.deduplicate()

    assert len(ts) == 3


def test_a_deduplicated_series_no_longer_warns(recwarn):
    with pytest.warns(UserWarning, match="appear more than once"):
        ts = eeo.time_series([scene(6, value=10), scene(6, value=20)])
    recwarn.clear()

    ts.deduplicate()

    assert [w for w in recwarn if "more than once" in str(w.message)] == []


def test_deduplicating_is_what_stops_a_composite_double_counting():
    # Two copies of June and one of July: the median of (10, 10, 20) is the
    # duplicated date's value, and of (10, 20) it is the midpoint. This is the
    # bias the whole rule exists to remove.
    with pytest.warns(UserWarning, match="appear more than once"):
        ts = eeo.time_series([scene(6, value=10), scene(6, value=10), scene(7, value=20)])

    assert float(ts.median().to_array()[0, 0, 0]) == pytest.approx(10.0)
    assert float(ts.deduplicate().median().to_array()[0, 0, 0]) == pytest.approx(15.0)


def test_a_deduplicated_series_keeps_the_grid_and_bands():
    with pytest.warns(UserWarning, match="appear more than once"):
        ts = eeo.time_series([scene(6, value=10), scene(6, value=20), scene(7)])

    deduplicated = ts.deduplicate()

    assert deduplicated.shape == ts.shape
    assert deduplicated.band_names == ts.band_names
    assert deduplicated.crs == ts.crs


# ---------------
# From a folder
# ---------------
def write_scene(path, *, value, size=4, crs=SCENE_CRS, origin=SCENE_ORIGIN, band_name="red"):
    """Write a single-band GeoTIFF, the shape a folder of scenes has on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with rio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="uint16",
        crs=crs,
        transform=from_origin(origin[0], origin[1], SCENE_RES, SCENE_RES),
    ) as dst:
        dst.write(np.full((size, size), value, dtype="uint16"), 1)
        dst.set_band_description(1, band_name)
    return path


@pytest.fixture
def folder(tmp_path):
    """Three dated scenes, written out of time order so the sort has work to do."""
    directory = tmp_path / "scenes"
    for name, value in (
        ("ndvi_20230701.tif", 7),
        ("ndvi_20230301.tif", 3),
        ("ndvi_20230501.tif", 5),
    ):
        write_scene(directory / name, value=value)
    return directory


def test_a_folder_becomes_a_series_in_time_order(folder):
    ts = eeo.time_series(folder)

    assert len(ts) == 3
    assert [stamp.date() for stamp in ts.timestamps] == [
        dt.date(2023, 3, 1),
        dt.date(2023, 5, 1),
        dt.date(2023, 7, 1),
    ]
    assert [int(ds.to_array().flat[0]) for ds in ts] == [3, 5, 7]
    ts.close()


def test_the_folder_path_is_file_backed(folder):
    ts = eeo.time_series(folder)

    # The point of the path: one GDAL handle per timestep, no pixels held.
    assert all(isinstance(ds._adapter, RasterioAdapter) for ds in ts)
    assert ts.band_names == ["red"]
    assert [ds.timestamp for ds in ts] == ts.timestamps
    ts.close()


def test_the_classmethod_is_the_same_path(folder):
    ts = EEOTimeSeries.from_folder(folder)

    assert len(ts) == 3
    ts.close()


def test_a_pattern_selects_which_files(folder):
    write_scene(folder / "mask_20230401.tif", value=1)

    ts = eeo.time_series(folder, pattern="ndvi_*.tif")

    assert len(ts) == 3
    assert dt.date(2023, 4, 1) not in [stamp.date() for stamp in ts.timestamps]
    ts.close()


def test_a_recursive_pattern_walks_subdirectories(tmp_path):
    root = tmp_path / "nested"
    write_scene(root / "2023-03" / "ndvi_20230301.tif", value=3)
    write_scene(root / "2023-05" / "ndvi_20230501.tif", value=5)

    ts = eeo.time_series(root, pattern="**/*.tif")

    assert [int(ds.to_array().flat[0]) for ds in ts] == [3, 5]
    ts.close()


def test_a_directory_matching_the_pattern_is_skipped(tmp_path):
    root = tmp_path / "scenes"
    write_scene(root / "ndvi_20230301.tif", value=3)
    (root / "20230401.tif").mkdir()

    ts = eeo.time_series(root, pattern="*")

    assert len(ts) == 1
    ts.close()


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("20230412.tif", dt.datetime(2023, 4, 12, tzinfo=UTC)),
        ("2023-04-12.tif", dt.datetime(2023, 4, 12, tzinfo=UTC)),
        ("ndvi_2023-04-12_clipped.tif", dt.datetime(2023, 4, 12, tzinfo=UTC)),
        # What a Sentinel-2 asset is called as delivered.
        ("T33TUL_20230412T100621_B04.tif", dt.datetime(2023, 4, 12, 10, 6, 21, tzinfo=UTC)),
        ("2023-04-12 10:06:21.tif", dt.datetime(2023, 4, 12, 10, 6, 21, tzinfo=UTC)),
        ("scene_20230412_100621.tif", dt.datetime(2023, 4, 12, 10, 6, 21, tzinfo=UTC)),
        # A Landsat product id: the acquisition date comes before the
        # processing date, and the first date in the name is the one wanted.
        (
            "LC09_L2SP_192029_20240910_20240911_02_T1_SR_B4.TIF",
            dt.datetime(2024, 9, 10, tzinfo=UTC),
        ),
    ],
)
def test_the_recognised_filename_dates(tmp_path, name, expected):
    write_scene(tmp_path / name, value=1)

    ts = eeo.time_series(tmp_path, pattern=name)

    assert ts.timestamps == [expected]
    ts.close()


@pytest.mark.parametrize(
    "name",
    [
        # Eight digits of a longer number are not a date.
        "scene_1234567890.tif",
        # Nor is a half-punctuated one.
        "scene_2023-0412.tif",
        # Nor an impossible one.
        "scene_20231345.tif",
        "plain.tif",
    ],
)
def test_a_filename_without_a_date_is_refused(tmp_path, name):
    write_scene(tmp_path / name, value=1)

    with pytest.raises(ValidationError, match="carries no date in its name"):
        eeo.time_series(tmp_path, pattern=name)


def test_the_undated_refusal_names_the_escape_hatch(tmp_path):
    write_scene(tmp_path / "plain.tif", value=1)

    with pytest.raises(ValidationError, match="timestamp=lambda"):
        eeo.time_series(tmp_path)


def test_a_date_after_an_impossible_one_is_still_found(tmp_path):
    # The scan takes the first match that is a real date, not the first that
    # looks like one: 9999-13-45 is neither, so the real date after it wins.
    write_scene(tmp_path / "v99991345_20230412.tif", value=1)

    ts = eeo.time_series(tmp_path)

    assert ts.timestamps == [dt.datetime(2023, 4, 12, tzinfo=UTC)]
    ts.close()


def test_no_file_is_opened_when_one_name_has_no_date(tmp_path, monkeypatch):
    write_scene(tmp_path / "ndvi_20230301.tif", value=3)
    write_scene(tmp_path / "undated.tif", value=5)
    opened = []
    monkeypatch.setattr(
        "eeo.timeseries.core.load_raster",
        lambda path, **kwargs: opened.append(path),
    )

    with pytest.raises(ValidationError, match="carries no date"):
        eeo.time_series(tmp_path)

    assert opened == [], "files were opened before the dates were known to be readable"


def test_a_timestamp_function_overrides_the_filename(folder):
    dates = {"ndvi_20230301.tif": 1, "ndvi_20230501.tif": 2, "ndvi_20230701.tif": 3}

    ts = eeo.time_series(
        folder, timestamp=lambda path: dt.datetime(2020, dates[path.name], 15, tzinfo=UTC)
    )

    assert [stamp.month for stamp in ts.timestamps] == [1, 2, 3]
    assert all(stamp.year == 2020 for stamp in ts.timestamps)
    ts.close()


def test_a_timestamp_function_can_read_the_parent_directory(tmp_path):
    # The layout the filename rule deliberately does not guess at: one
    # directory per acquisition, every file inside called the same thing.
    root = tmp_path / "nested"
    write_scene(root / "2023-03-01" / "B04.tif", value=3)
    write_scene(root / "2023-05-01" / "B04.tif", value=5)

    ts = eeo.time_series(
        root,
        pattern="*/B04.tif",
        timestamp=lambda path: dt.datetime.fromisoformat(path.parent.name),
    )

    assert [stamp.date() for stamp in ts.timestamps] == [dt.date(2023, 3, 1), dt.date(2023, 5, 1)]
    ts.close()


def test_a_naive_timestamp_from_the_function_is_read_as_utc(folder):
    months = {"ndvi_20230301.tif": 1, "ndvi_20230501.tif": 2, "ndvi_20230701.tif": 3}

    ts = eeo.time_series(folder, timestamp=lambda path: dt.datetime(2020, months[path.name], 1))

    assert all(stamp.tzinfo is not None for stamp in ts.timestamps)
    assert ts.timestamps[0] == dt.datetime(2020, 1, 1, tzinfo=UTC)
    ts.close()


def test_a_timestamp_function_returning_the_wrong_type_names_the_file(folder):
    with pytest.raises(ValidationError, match="ndvi_20230301.tif"):
        eeo.time_series(folder, timestamp=lambda path: "2023-03-01")


def test_timestamp_must_be_callable(folder):
    with pytest.raises(ValidationError, match="function taking a path"):
        eeo.time_series(folder, timestamp=dt.datetime(2023, 3, 1, tzinfo=UTC))


def test_an_empty_match_says_what_a_glob_matches(folder):
    with pytest.raises(ValidationError, match="walks subdirectories"):
        eeo.time_series(folder, pattern="*.TIF")


def test_an_absolute_pattern_is_refused(folder):
    # Path.glob's own message reads as "still not implemented", which is exactly
    # the wrong thing for a method that used to raise NotImplementedError.
    with pytest.raises(ValidationError, match="not a usable glob pattern"):
        eeo.time_series(folder, pattern="/data/*.tif")


def test_a_missing_folder_is_refused(tmp_path):
    with pytest.raises(ValidationError, match="no such folder"):
        eeo.time_series(tmp_path / "absent")


def test_a_file_is_not_a_folder(folder):
    with pytest.raises(ValidationError, match="not a folder"):
        eeo.time_series(folder / "ndvi_20230301.tif")


def test_assets_are_refused_for_a_folder(folder):
    with pytest.raises(ValidationError, match="means nothing for a folder"):
        eeo.time_series(folder, assets=["B04"])


def test_a_folder_of_mismatched_grids_is_refused(folder):
    write_scene(folder / "ndvi_20230901.tif", value=9, size=8)

    with pytest.raises(AlignmentError):
        eeo.time_series(folder)


def test_a_mismatched_folder_can_be_aligned(folder):
    write_scene(folder / "ndvi_20230901.tif", value=9, size=8)

    ts = eeo.time_series(folder, auto_align=True)

    assert len(ts) == 4
    assert ts.shape == (4, 4)
    ts.close()


def test_a_folder_series_reduces_and_extracts(folder):
    ts = eeo.time_series(folder)

    assert float(ts.median().to_array()[0, 0, 0]) == pytest.approx(5.0)
    assert list(ts.extract_at((SCENE_ORIGIN[0] + 5, SCENE_ORIGIN[1] - 5))["red"]) == [3.0, 5.0, 7.0]
    ts.close()


def test_a_folder_can_be_opened_on_the_lazy_backend(folder):
    pytest.importorskip("dask.array")
    pytest.importorskip("rioxarray")

    ts = eeo.time_series(folder, chunks="auto")

    assert all(isinstance(ds._adapter, XarrayAdapter) for ds in ts)
    assert float(ts.median().to_array()[0, 0, 0]) == pytest.approx(5.0)
    ts.close()


def test_an_invalid_chunk_spec_is_refused_before_anything_is_opened(folder):
    with pytest.raises(ValidationError):
        eeo.time_series(folder, chunks={"rows": 256})


# -------------------------
# From a STAC search result
# -------------------------
class FakeAsset:
    def __init__(self, href):
        self.href = str(href)


class FakeItem:
    """Minimal stand-in for a pystac Item, as tests/test_stac_load.py uses."""

    def __init__(self, assets, *, timestamp, item_id="S2A_TEST"):
        self.id = item_id
        self.datetime = timestamp
        self.collection_id = "sentinel-2-l2a"
        self.properties = {"eo:cloud_cover": 4.2, "platform": "Sentinel-2A"}
        self.assets = {name: FakeAsset(href) for name, href in assets.items()}
        self.bbox = list(scene_bbox_wgs84())


def scene_bbox_wgs84():
    left, top = SCENE_ORIGIN
    span = SCENE_SIZE * SCENE_RES
    return transform_bounds(
        SCENE_CRS, "EPSG:4326", left, top - span, left + span, top, densify_pts=21
    )


def write_asset(path, *, fill):
    profile = {
        "driver": "GTiff",
        "height": SCENE_SIZE,
        "width": SCENE_SIZE,
        "count": 1,
        "dtype": "uint16",
        "crs": SCENE_CRS,
        "transform": transform(),
        "nodata": 0,
    }
    with rio.open(path, "w", **profile) as dst:
        dst.write(np.full((1, SCENE_SIZE, SCENE_SIZE), fill, dtype="uint16"))
        dst.set_band_description(1, "B04")
    return path


@pytest.fixture
def items(tmp_path):
    """Three items, handed over newest-first so ordering is actually tested."""
    made = []
    for month, fill in ((9, 300), (4, 100), (6, 200)):
        href = write_asset(tmp_path / f"scene_{month:02d}.tif", fill=fill)
        item = FakeItem(
            {"B04": href},
            timestamp=dt.datetime(2023, month, 12, 10, 6, 21, tzinfo=UTC),
            item_id=f"S2A_2023{month:02d}",
        )
        made.append(eeo.io.STACItem(item))
    return made


def search_result(items):
    return eeo.io.STACSearchResult(
        items, collections=["sentinel-2-l2a"], catalog="https://example.invalid/stac"
    )


def test_from_stac_builds_an_ordered_series(items):
    ts = eeo.time_series(search_result(items), assets="B04")

    assert len(ts) == 3
    assert [stamp.month for stamp in ts.timestamps] == [4, 6, 9]
    assert [int(ds.to_array().flat[0]) for ds in ts] == [100, 200, 300]
    ts.close()


def test_from_stac_carries_provenance_onto_every_timestep(items):
    ts = EEOTimeSeries.from_stac(items, ["B04"])

    for ds in ts:
        assert ds.band_names == ["B04"]
        assert ds.attrs["stac_collection"] == "sentinel-2-l2a"
        assert ds.attrs["mission"] == "Sentinel-2"
        assert ds.timestamp is not None
    assert [ds.attrs["stac_item"] for ds in ts] == [
        "S2A_202304",
        "S2A_202306",
        "S2A_202309",
    ]
    ts.close()


def test_cached_scenes_are_file_backed_and_removed_on_close(items):
    ts = EEOTimeSeries.from_stac(items, ["B04"])

    paths = [Path(ds.path) for ds in ts]
    assert all(path.exists() for path in paths)
    # A file-backed timestep is what keeps a long series out of memory.
    assert all(path.suffix == ".tif" for path in paths)

    ts.close()
    assert not any(path.exists() for path in paths)


def test_cache_directory_is_kept_when_named(items, tmp_path):
    kept = tmp_path / "cache"
    ts = EEOTimeSeries.from_stac(items, ["B04"], cache=kept)

    written = sorted(kept.glob("*.tif"))
    assert len(written) == 3
    # Named by position and acquisition time, so the directory reads in order.
    assert written[0].name.startswith("0000_20230412T")

    ts.close()
    assert sorted(kept.glob("*.tif")) == written


def test_cache_false_keeps_the_scenes_in_memory(items):
    ts = EEOTimeSeries.from_stac(items, ["B04"], cache=False)

    assert all(ds.path is None for ds in ts)
    assert len(ts) == 3
    ts.close()


def test_chunks_without_a_cache_is_refused(items):
    with pytest.raises(ValidationError, match="needs a cache"):
        EEOTimeSeries.from_stac(items, ["B04"], cache=False, chunks="auto")


def test_invalid_chunks_are_refused_before_anything_is_read(items):
    with pytest.raises(ValidationError, match="chunks must be"):
        EEOTimeSeries.from_stac(items, ["B04"], chunks=0)


def test_an_undated_item_is_refused_before_any_asset_is_read(tmp_path, items):
    undated = eeo.io.STACItem(
        FakeItem({"B04": write_asset(tmp_path / "undated.tif", fill=5)}, timestamp=None)
    )

    with pytest.raises(ValidationError, match="no acquisition time"):
        EEOTimeSeries.from_stac([*items, undated], ["B04"])


def test_an_empty_search_result_is_refused():
    with pytest.raises(ValidationError, match="returned no\n?\\s*items"):
        EEOTimeSeries.from_stac(search_result([]), ["B04"])


def test_assets_are_required_for_a_stac_source(items):
    with pytest.raises(ValidationError, match="which assets to read"):
        eeo.time_series(search_result(items))


def test_a_failed_load_leaves_no_cache_behind(items):
    with pytest.raises(ValidationError):
        EEOTimeSeries.from_stac(items, ["B99"])


def test_lazy_chunks_open_cached_scenes_on_the_lazy_backend(items):
    pytest.importorskip("dask.array")
    pytest.importorskip("rioxarray")
    from eeo.core.adapters import XarrayAdapter

    ts = EEOTimeSeries.from_stac(items, ["B04"], chunks="auto")

    assert all(isinstance(ds._adapter, XarrayAdapter) for ds in ts)
    assert [int(ds.to_array().flat[0]) for ds in ts] == [100, 200, 300]
    ts.close()


# ---
# map
# ---
def two_band_scene(month, *, red, nir):
    """A two-band named scene, so the indices can be addressed by name."""
    array = np.stack(
        [
            np.full((4, 4), red, dtype="uint16"),
            np.full((4, 4), nir, dtype="uint16"),
        ]
    )
    return eeo.load_array(
        array,
        transform=transform(),
        crs=SCENE_CRS,
        timestamp=dt.datetime(2023, month, 1, tzinfo=UTC),
        band_names=["red", "nir"],
    )


def test_map_applies_an_index_to_every_timestep():
    ts = eeo.time_series(
        [
            two_band_scene(1, red=1000, nir=3000),
            two_band_scene(2, red=2000, nir=2000),
        ]
    )

    ndvi = ts.map(eeo.ndvi, red="red", nir="nir")

    assert len(ndvi) == 2
    assert [stamp.month for stamp in ndvi.timestamps] == [1, 2]
    assert ndvi.band_count == 1
    assert pytest.approx(float(ndvi[0].to_array().flat[0]), abs=1e-6) == 0.5
    assert pytest.approx(float(ndvi[1].to_array().flat[0]), abs=1e-6) == 0.0


def test_map_chains_and_leaves_the_source_untouched():
    ts = eeo.time_series([scene(3, value=10), scene(4, value=20)])

    doubled = ts.map(eeo.multiply, other=2).map(eeo.multiply, other=3)

    assert [int(ds.to_array().flat[0]) for ds in doubled] == [60, 120]
    # The inputs are the operation's operands, never its outputs.
    assert [int(ds.to_array().flat[0]) for ds in ts] == [10, 20]


def test_map_accepts_a_function_of_your_own():
    ts = eeo.time_series([scene(5, value=7)])

    result = ts.map(lambda ds: ds.multiply(2))

    assert int(result[0].to_array().flat[0]) == 14


def test_map_keeps_overridden_timestamps():
    stamps = [dt.datetime(2019, 1, 1, tzinfo=UTC), dt.datetime(2019, 2, 1, tzinfo=UTC)]
    ts = eeo.time_series([scene(8), scene(9)], timestamps=stamps)

    assert ts.map(eeo.multiply, other=2).timestamps == stamps


def test_map_rejects_the_name_of_an_operation():
    ts = eeo.time_series([scene(1)])

    with pytest.raises(ValidationError, match="not its name"):
        ts.map("ndvi", red="red", nir="nir")


def test_map_rejects_something_uncallable():
    ts = eeo.time_series([scene(1)])

    with pytest.raises(ValidationError, match="needs a callable"):
        ts.map(42)


def test_map_rejects_an_operation_that_returns_a_value():
    ts = eeo.time_series([scene(1)])

    with pytest.raises(ValidationError, match="every result must be an"):
        ts.map(lambda ds: float(ds.to_array().mean()))


def test_map_save_dir_writes_files_and_returns_a_file_backed_series(tmp_path):
    ts = eeo.time_series([scene(6, value=4), scene(7, value=8)])
    out = tmp_path / "doubled"

    result = ts.map(eeo.multiply, other=2, save_dir=out)

    written = sorted(out.glob("*.tif"))
    assert [path.name for path in written] == [
        "0000_20230601T000000.tif",
        "0001_20230701T000000.tif",
    ]
    assert [Path(ds.path) for ds in result] == written
    assert [int(ds.to_array().flat[0]) for ds in result] == [8, 16]


def test_map_save_dir_creates_the_directory_and_overwrites_a_rerun(tmp_path):
    ts = eeo.time_series([scene(6, value=4)])
    out = tmp_path / "nested" / "out"

    first = ts.map(eeo.multiply, other=2, save_dir=out)
    first.close()
    second = ts.map(eeo.multiply, other=3, save_dir=out)

    assert len(sorted(out.glob("*.tif"))) == 1
    assert int(second[0].to_array().flat[0]) == 12


def test_map_preserves_band_names_through_a_saved_round_trip(tmp_path):
    ts = eeo.time_series([two_band_scene(4, red=1000, nir=3000)])

    result = ts.map(eeo.ndvi, red="red", nir="nir", name="greenness", save_dir=tmp_path / "ndvi")

    assert result.band_names == ["greenness"]


def test_map_carries_provenance_as_the_bound_operation_does():
    # An operation called as a bare function skips the decorator that copies
    # timestamp, attrs and band names onto its result; map must not.
    scene = two_band_scene(4, red=1000, nir=3000)
    scene.attrs["mission"] = "Sentinel-2"

    result = eeo.time_series([scene]).map(eeo.multiply, other=2)

    assert result[0].band_names == ["red", "nir"]
    assert result[0].attrs["mission"] == "Sentinel-2"
    assert result[0].timestamp == scene.timestamp


def test_map_keeps_a_lazy_series_lazy_when_saving(items, tmp_path):
    pytest.importorskip("dask.array")
    pytest.importorskip("rioxarray")
    from eeo.core.adapters import XarrayAdapter

    ts = EEOTimeSeries.from_stac(items, ["B04"], chunks="auto")

    result = ts.map(eeo.multiply, other=2, save_dir=tmp_path / "lazy")

    assert all(isinstance(ds._adapter, XarrayAdapter) for ds in result)
    assert [int(ds.to_array().flat[0]) for ds in result] == [200, 400, 600]
    ts.close()
    result.close()


# --------------------------------------------------------------------------
# Grid, CRS and band validation
# --------------------------------------------------------------------------
def shifted_scene(month, *, value=1, offset_pixels=0, size=4, crs=SCENE_CRS, res=SCENE_RES):
    """A scene whose grid can be moved off the reference's."""
    origin_x = SCENE_ORIGIN[0] + offset_pixels * res
    return eeo.load_array(
        np.full((size, size), value, dtype="uint16"),
        transform=from_origin(origin_x, SCENE_ORIGIN[1], res, res),
        crs=crs,
        timestamp=dt.datetime(2023, month, 1, tzinfo=UTC),
    ).to_rasterio()


def test_a_different_shape_is_refused_without_auto_align():
    with pytest.raises(AlignmentError, match="auto_align=True"):
        eeo.time_series([shifted_scene(1), shifted_scene(2, size=6)])


def test_a_shifted_grid_of_the_same_shape_is_refused_too():
    # The shapes match and only the origin differs, which is the case a
    # shape-only check would wave through — and would average different ground.
    with pytest.raises(AlignmentError, match="different grid"):
        eeo.time_series([shifted_scene(1), shifted_scene(2, offset_pixels=2)])


def test_auto_align_warps_onto_the_reference_grid():
    ts = eeo.time_series(
        [shifted_scene(1, value=5), shifted_scene(2, value=9, size=6)],
        auto_align=True,
    )

    assert ts.shape == (4, 4)
    assert all(ds.get_shape() == (4, 4) for ds in ts)
    assert all(ds.get_transform() == ts.transform for ds in ts)
    # Alignment resamples; it does not invent values.
    assert int(ts[1].to_array().flat[0]) == 9


def test_a_shifted_timestep_lands_on_the_reference_grid():
    ts = eeo.time_series([shifted_scene(1), shifted_scene(2, offset_pixels=2)], auto_align=True)

    assert ts[1].get_transform() == ts[0].get_transform()


def test_a_different_crs_is_refused_without_auto_reproject():
    with pytest.raises(CRSMismatchError, match="auto_reproject=True"):
        eeo.time_series([shifted_scene(1), shifted_scene(2, crs="EPSG:32634")])


def test_auto_align_alone_does_not_permit_a_reprojection():
    with pytest.raises(CRSMismatchError):
        eeo.time_series([shifted_scene(1), shifted_scene(2, crs="EPSG:32634")], auto_align=True)


def test_auto_reproject_puts_every_timestep_in_one_crs():
    ts = eeo.time_series(
        [shifted_scene(1), shifted_scene(2, crs="EPSG:32634")], auto_reproject=True
    )

    assert {ds.get_crs().to_epsg() for ds in ts} == {32633}
    assert ts.shape == (4, 4)


def test_crs_spellings_of_one_system_are_not_a_mismatch():
    # A NumPy-backed dataset hands back whatever crs= was given, so an int and a
    # string for the same system must not read as two different CRSs.
    ts = eeo.time_series(
        [
            eeo.load_array(
                np.ones((4, 4), dtype="uint16"),
                transform=transform(),
                crs=32633,
                timestamp=dt.datetime(2023, 1, 1, tzinfo=UTC),
            ),
            eeo.load_array(
                np.ones((4, 4), dtype="uint16"),
                transform=transform(),
                crs="EPSG:32633",
                timestamp=dt.datetime(2023, 2, 1, tzinfo=UTC),
            ),
        ]
    )

    assert ts.crs.to_epsg() == 32633


def test_a_different_band_count_is_refused_whatever_the_flags():
    with pytest.raises(ValidationError, match="same bands"):
        eeo.time_series(
            [scene(1), two_band_scene(2, red=1, nir=2)],
            auto_align=True,
            auto_reproject=True,
        )


def test_bands_that_disagree_about_their_names_are_refused():
    first = two_band_scene(1, red=1, nir=2)
    second = two_band_scene(2, red=1, nir=2)
    second.band_names = ["nir", "red"]

    with pytest.raises(ValidationError, match="must mean the same thing"):
        eeo.time_series([first, second])


def test_an_unnamed_band_does_not_conflict_with_a_named_one():
    named = two_band_scene(1, red=1, nir=2)
    unnamed = two_band_scene(2, red=1, nir=2)
    unnamed.band_names = None

    ts = eeo.time_series([named, unnamed])

    assert ts.band_names == ["red", "nir"]


def test_the_reference_timestep_can_be_chosen():
    ts = eeo.time_series(
        [shifted_scene(1, size=4), shifted_scene(2, size=6)],
        reference=-1,
        auto_align=True,
    )

    assert ts.shape == (6, 6)
    assert ts.reference is ts[1]


def test_an_out_of_range_reference_is_refused():
    with pytest.raises(ValidationError, match="not a timestep"):
        eeo.time_series([scene(1), scene(2)], reference=5)


def test_alignment_says_what_it_did(caplog):
    with caplog.at_level("INFO", logger="eeo.timeseries.core"):
        eeo.time_series([shifted_scene(1), shifted_scene(2, size=6)], auto_align=True)

    assert "aligned timesteps [1]" in caplog.text
    assert "method=nearest" in caplog.text


# --------------------------------------------------------------------------
# The Sentinel-2 baseline 04.00 guard
# --------------------------------------------------------------------------
def s2_scene(when, *, baseline=None, mission="Sentinel-2"):
    """A Sentinel-2 scene with the provenance the baseline check reads."""
    attrs = {"mission": mission}
    if baseline is not None:
        attrs["processing_baseline"] = baseline
    return eeo.load_array(
        np.ones((4, 4), dtype="uint16"),
        transform=transform(),
        crs=SCENE_CRS,
        timestamp=when,
        attrs=attrs,
    )


BEFORE_04_00 = dt.datetime(2021, 6, 1, tzinfo=UTC)
AFTER_04_00 = dt.datetime(2022, 6, 1, tzinfo=UTC)
# A second acquisition on the late side, so a test about baselines does not
# repeat a moment and draw the duplicate-acquisition warning as well.
ALSO_AFTER_04_00 = dt.datetime(2022, 7, 1, tzinfo=UTC)


def test_a_series_spanning_baseline_04_00_warns():
    with pytest.warns(UserWarning, match="baseline 04.00"):
        eeo.time_series([s2_scene(BEFORE_04_00), s2_scene(AFTER_04_00)])


def test_the_baseline_warning_names_the_split_and_the_offset():
    with pytest.warns(UserWarning) as caught:
        eeo.time_series([s2_scene(BEFORE_04_00), s2_scene(AFTER_04_00), s2_scene(ALSO_AFTER_04_00)])

    message = str(caught[0].message)
    assert "1 timestep(s) sit before it and 2 after" in message
    assert "BOA_ADD_OFFSET" in message
    assert "1000 DN" in message


def test_a_series_on_one_side_of_the_boundary_is_quiet():
    # filterwarnings = error, so a stray warning here fails the test outright.
    ts = eeo.time_series([s2_scene(AFTER_04_00), s2_scene(ALSO_AFTER_04_00)])

    assert len(ts) == 2


def test_the_recorded_baseline_beats_the_acquisition_date():
    # Reprocessed archive scenes carry 04.00+ on old acquisitions, so a series
    # of them is consistent even though it straddles the date.
    ts = eeo.time_series(
        [
            s2_scene(BEFORE_04_00, baseline="05.11"),
            s2_scene(AFTER_04_00, baseline="05.11"),
        ]
    )

    assert len(ts) == 2


def test_a_recorded_baseline_below_04_00_puts_a_scene_before_the_change():
    with pytest.warns(UserWarning, match="baseline 04.00"):
        eeo.time_series(
            [
                s2_scene(AFTER_04_00, baseline="03.01"),
                s2_scene(ALSO_AFTER_04_00, baseline="05.11"),
            ]
        )


def test_an_unparsable_recorded_baseline_falls_back_to_the_date():
    with pytest.warns(UserWarning, match="baseline 04.00"):
        eeo.time_series([s2_scene(BEFORE_04_00, baseline=""), s2_scene(AFTER_04_00, baseline="")])


def test_a_series_of_another_mission_is_never_warned_about():
    ts = eeo.time_series(
        [
            s2_scene(BEFORE_04_00, mission="Landsat 9"),
            s2_scene(AFTER_04_00, mission="Landsat 9"),
        ]
    )

    assert len(ts) == 2


def test_scenes_with_no_recorded_mission_are_never_warned_about():
    ts = eeo.time_series([scene(1), scene(2)])

    assert len(ts) == 2


def reprocessed_items(tmp_path, *, grid="MGRS-33TUL"):
    """Two catalog copies of one acquisition, the second processed a year later."""
    sensed = dt.datetime(2023, 4, 12, 10, 6, 21, tzinfo=UTC)
    tmp_path.mkdir(parents=True, exist_ok=True)
    made = []
    for processed, fill in (("2023-04-13T00:00:00Z", 10), ("2024-06-01T00:00:00Z", 20)):
        href = write_asset(tmp_path / f"copy_{fill}.tif", fill=fill)
        item = FakeItem({"B04": href}, timestamp=sensed, item_id=f"S2A_{fill}")
        item.properties["processing:datetime"] = processed
        item.properties["grid:code"] = grid
        made.append(eeo.io.STACItem(item))
    return made


def test_deduplicating_the_search_means_the_duplicate_is_never_read(tmp_path):
    # The path worth taking: the loser is dropped on metadata, so it costs no
    # read at all — and the series never has cause to warn.
    result = eeo.io.STACSearchResult(
        reprocessed_items(tmp_path),
        collections=["sentinel-2-l2a"],
        catalog="https://example.invalid",
    )

    ts = eeo.time_series(result.deduplicate(), assets=["B04"])

    assert len(ts) == 1
    assert float(ts[0].to_array()[0, 0, 0]) == 20
    ts.close()


def test_two_tiles_of_one_overpass_both_survive_a_search_deduplication(tmp_path):
    west = reprocessed_items(tmp_path / "w", grid="MGRS-33TUL")[0]
    east = reprocessed_items(tmp_path / "e", grid="MGRS-33TUM")[0]
    result = eeo.io.STACSearchResult(
        [west, east], collections=["sentinel-2-l2a"], catalog="https://example.invalid"
    )

    assert len(result.deduplicate()) == 2


def test_a_stac_series_deduplicates_on_the_recorded_processing_time(tmp_path):
    # The fallback path: the items are gone, and the scenes' own attrs carry
    # enough — the processing time recorded at load — to pick the same winner.
    with pytest.warns(UserWarning, match="appear more than once"):
        ts = EEOTimeSeries.from_stac(reprocessed_items(tmp_path), ["B04"])

    deduplicated = ts.deduplicate()

    assert len(deduplicated) == 1
    assert float(deduplicated[0].to_array()[0, 0, 0]) == 20
    ts.close()


def test_a_stac_series_straddling_the_baseline_warns_end_to_end(tmp_path):
    # The whole path: the items say which platform took them, the load records
    # the mission, and the series notices the two radiometric conventions.
    made = []
    for year, month in ((2021, 6), (2023, 6)):
        href = write_asset(tmp_path / f"scene_{year}.tif", fill=100)
        made.append(
            eeo.io.STACItem(
                FakeItem(
                    {"B04": href},
                    timestamp=dt.datetime(year, month, 12, tzinfo=UTC),
                    item_id=f"S2A_{year}",
                )
            )
        )

    with pytest.warns(UserWarning, match="baseline 04.00"):
        ts = EEOTimeSeries.from_stac(made, ["B04"])

    ts.close()


def test_a_stac_series_reads_its_baseline_from_the_items(tmp_path):
    # With the baseline recorded, the dates no longer decide: both scenes were
    # reprocessed to 05.11, so the series is consistent and stays quiet.
    made = []
    for year in (2021, 2023):
        href = write_asset(tmp_path / f"reprocessed_{year}.tif", fill=100)
        item = FakeItem(
            {"B04": href},
            timestamp=dt.datetime(year, 6, 12, tzinfo=UTC),
            item_id=f"S2A_{year}",
        )
        item.properties["processing:version"] = "05.11"
        made.append(eeo.io.STACItem(item))

    ts = EEOTimeSeries.from_stac(made, ["B04"])

    assert [ds.attrs["processing_baseline"] for ds in ts] == ["05.11", "05.11"]
    ts.close()


# --------------------------------------------------------------------------
# The synthetic five-timestep season stack (shared fixtures in conftest.py)
# --------------------------------------------------------------------------
def test_the_season_stack_describes_itself_as_one_series(season_series):
    assert len(season_series) == 5
    assert [stamp.date().isoformat() for stamp in season_series.timestamps] == [
        "2023-03-01",
        "2023-04-01",
        "2023-05-01",
        "2023-06-01",
        "2023-07-01",
    ]
    assert season_series.shape == (4, 4)
    assert season_series.band_count == 2
    assert season_series.band_names == ["red", "nir"]
    assert season_series.crs.to_epsg() == 32633
    assert "5 timesteps from 2023-03-01 to 2023-07-01" in repr(season_series)


def test_map_normalized_difference_across_the_stack(season_series, season_reference):
    # The acceptance case for the time-series core: an existing two-raster op,
    # unmodified, applied across every timestep.
    result = season_series.map(eeo.normalized_difference, other=season_reference)

    assert len(result) == 5
    assert result.band_count == 2
    assert result.shape == season_series.shape
    assert result.transform == season_series.transform
    assert result.timestamps == season_series.timestamps
    assert result[0].get_metadata()["dtype"] == "float32"

    # (band - 1000) / (band + 1000) per timestep, at a pixel valid throughout.
    red_expected = [0.0, -0.052632, -0.111111, -0.052632, 0.0]
    nir_expected = [1 / 3, 0.5, 0.6, 0.5, 1 / 3]
    for index, ds in enumerate(result):
        values = ds.to_array()
        assert values[0, 1, 1] == pytest.approx(red_expected[index], abs=1e-6)
        assert values[1, 1, 1] == pytest.approx(nir_expected[index], abs=1e-6)


def test_an_index_across_the_stack_traces_the_season(season_series):
    trajectory = [
        float(ds.to_array()[0, 1, 1]) for ds in season_series.map(eeo.ndvi, red="red", nir="nir")
    ]

    assert trajectory == pytest.approx([1 / 3, 0.538462, 2 / 3, 0.538462, 1 / 3], abs=1e-6)
    # Rises to midsummer and falls back, symmetrically.
    assert trajectory[0] < trajectory[1] < trajectory[2]
    assert trajectory[2] > trajectory[3] > trajectory[4]
    assert trajectory[0] == pytest.approx(trajectory[4])


def test_the_nodata_gaps_survive_a_mapped_index(season_series):
    # Pixel (0, 0) is nodata at the third and fourth timesteps only; the index
    # must mark those and only those, so a reducer can skip them.
    gap_pixel = [
        float(ds.to_array()[0, 0, 0]) for ds in season_series.map(eeo.ndvi, red="red", nir="nir")
    ]

    assert not np.isnan(gap_pixel[0])
    assert not np.isnan(gap_pixel[1])
    assert np.isnan(gap_pixel[2])
    assert np.isnan(gap_pixel[3])
    assert not np.isnan(gap_pixel[4])


def test_slicing_the_season_keeps_a_sub_season(season_series):
    midsummer = season_series[1:4]

    assert len(midsummer) == 3
    assert [stamp.month for stamp in midsummer.timestamps] == [4, 5, 6]
    assert midsummer.band_names == ["red", "nir"]


def test_mapping_the_season_to_files_writes_one_raster_per_timestep(season_series, tmp_path):
    result = season_series.map(eeo.ndvi, red="red", nir="nir", save_dir=tmp_path / "ndvi")

    written = sorted((tmp_path / "ndvi").glob("*.tif"))
    assert [path.name for path in written] == [
        "0000_20230301T000000.tif",
        "0001_20230401T000000.tif",
        "0002_20230501T000000.tif",
        "0003_20230601T000000.tif",
        "0004_20230701T000000.tif",
    ]
    assert [float(ds.to_array()[0, 1, 1]) for ds in result] == pytest.approx(
        [1 / 3, 0.538462, 2 / 3, 0.538462, 1 / 3], abs=1e-6
    )
    result.close()


def test_the_stack_needs_no_alignment_flags(season_stack):
    # One grid, one CRS, one set of bands: the series builds without permission
    # to touch the pixels, which is the case every real workflow should hit.
    ts = eeo.time_series(season_stack)

    assert ts.reference is ts[0]
    assert all(ds.get_transform() == ts.transform for ds in ts)
