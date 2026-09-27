"""Construction and ordering of EEOTimeSeries (eeo/timeseries/core.py).

Covers construction: the three paths (datasets, a STAC
search result, and the folder stub), chronological ordering, the timestamp
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
from eeo.core.exceptions import ValidationError
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


def test_duplicate_timestamps_keep_their_arrival_order():
    first, second = scene(6, value=10), scene(6, value=20)

    ts = eeo.time_series([first, second])

    assert ts[0] is first
    assert ts[1] is second


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


# ---------------
# The folder stub
# ---------------
def test_from_folder_names_its_replacement(tmp_path):
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        EEOTimeSeries.from_folder(tmp_path)


def test_a_path_source_reaches_the_folder_stub(tmp_path):
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        eeo.time_series(str(tmp_path))


def test_the_folder_stub_suggests_the_supported_path(tmp_path):
    with pytest.raises(NotImplementedError, match="stac_search"):
        eeo.time_series(tmp_path, chunks="auto")


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
