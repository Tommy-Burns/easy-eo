"""Choosing between several processings of one acquisition (eeo/io/_dedupe.py).

A catalog publishes a scene more than once and every copy matches a search,
which double-counts that date in any composite. These tests pin the rule that
decides which copy survives, and — as important — the two cases that must *not*
be collapsed: two tiles of one overpass, and two acquisitions that merely fall
in the same minute.

Items are faked, as tests/test_stac_load.py does, so nothing touches the
network. Field names are the ones the specs define: `processing:datetime` and
`processing:version` from the processing extension, `grid:code` from the grid
extension, `created`/`updated` from STAC common metadata.
"""

import datetime as dt

import pytest

from eeo.io._dedupe import (
    acquisition_key,
    deduplicate_items,
    processed_at,
    processing_rank,
    version_rank,
)
from eeo.io.stac import STACSearchResult

UTC = dt.timezone.utc
SENSED = dt.datetime(2023, 4, 12, 10, 6, 21, tzinfo=UTC)


class FakeItem:
    """Minimal stand-in for an STACItem: what the rule actually reads."""

    def __init__(
        self, item_id, *, timestamp=SENSED, collection="sentinel-2-l2a", bbox=None, **props
    ):
        self.id = item_id
        self.timestamp = timestamp
        self.collection = collection
        self.bbox = bbox
        self.properties = {key.replace("__", ":"): value for key, value in props.items()}


def ids(items):
    return [item.id for item in items]


# ------------------------------
# Reading a processing version
# ------------------------------
def test_a_baseline_is_compared_as_numbers_not_text():
    # The trap: as strings, "05.11" sorts below "5.2".
    assert version_rank({"processing:version": "05.11"}) > version_rank(
        {"processing:version": "5.2"}
    )


def test_the_deprecated_sentinel2_spelling_is_read():
    assert version_rank({"s2:processing_baseline": "04.00"}) == (4, 0)


def test_the_attrs_spelling_is_read_too():
    # What eeo/io/stac.py and the Sentinel-2 loaders record on a dataset.
    assert version_rank({"processing_baseline": "05.11"}) == (5, 11)


def test_the_processing_extension_field_wins_over_the_deprecated_one():
    properties = {"processing:version": "05.11", "s2:processing_baseline": "02.12"}

    assert version_rank(properties) == (5, 11)


@pytest.mark.parametrize("value", ["", "v5", "5.x", None, "unknown"])
def test_an_unreadable_version_contributes_nothing(value):
    assert version_rank({"processing:version": value}) == ()


# ------------------------------
# Reading a processing time
# ------------------------------
def test_a_processing_time_is_read_as_utc():
    assert processed_at({"processing:datetime": "2023-04-12T13:43:21Z"}) == dt.datetime(
        2023, 4, 12, 13, 43, 21, tzinfo=UTC
    )


def test_a_naive_processing_time_is_read_as_utc():
    assert processed_at({"processing:datetime": "2023-04-12T13:43:21"}).tzinfo is not None


def test_a_datetime_object_is_accepted():
    stamp = dt.datetime(2023, 4, 12, 13, 43, 21, tzinfo=UTC)

    assert processed_at({"updated": stamp}) == stamp


def test_the_data_field_is_preferred_over_the_metadata_fields():
    # `created` and `updated` describe the STAC record, not the data — the spec
    # says so — so the processing extension's own field must win.
    properties = {
        "processing:datetime": "2023-04-12T13:43:21Z",
        "updated": "2024-01-01T00:00:00Z",
        "created": "2024-01-01T00:00:00Z",
    }

    assert processed_at(properties).year == 2023


def test_updated_stands_in_where_there_is_no_processing_time():
    assert processed_at({"updated": "2024-06-01T00:00:00Z"}).year == 2024


@pytest.mark.parametrize("value", ["", "yesterday", None, "2023-13-45"])
def test_an_unreadable_time_is_no_time(value):
    assert processed_at({"processing:datetime": value}) is None


# ------------------------------
# The ranking rule
# ------------------------------
def test_the_higher_version_wins():
    older = processing_rank({"processing:version": "02.12"}, 0)
    newer = processing_rank({"processing:version": "05.11"}, 1)

    assert newer > older


def test_the_version_outranks_a_newer_metadata_update():
    # The case the ordering exists for: a metadata-only fix bumps `updated` on
    # the worse processing, and must not let it win.
    better = processing_rank({"processing:version": "05.11", "updated": "2023-01-01T00:00:00Z"}, 0)
    touched = processing_rank({"processing:version": "02.12", "updated": "2026-01-01T00:00:00Z"}, 1)

    assert better > touched


def test_the_processing_time_settles_one_version():
    first = processing_rank({"processing:version": "05.11", "processing:datetime": "2023-01-01"}, 0)
    second = processing_rank(
        {"processing:version": "05.11", "processing:datetime": "2024-01-01"}, 1
    )

    assert second > first


def test_the_processing_time_decides_where_there_is_no_version():
    # Landsat publishes no baseline, so time carries the decision alone.
    first = processing_rank({"processing:datetime": "2024-09-11T00:00:00Z"}, 0)
    second = processing_rank({"processing:datetime": "2024-09-20T00:00:00Z"}, 1)

    assert second > first


def test_a_stated_processing_time_beats_none_at_all():
    stated = processing_rank({"processing:datetime": "2000-01-01T00:00:00Z"}, 1)
    silent = processing_rank({}, 0)

    assert stated > silent


def test_arrival_order_settles_a_total_tie():
    assert processing_rank({}, 0) > processing_rank({}, 1)


# ------------------------------
# What counts as one acquisition
# ------------------------------
def test_the_key_is_collection_moment_and_ground():
    item = FakeItem("a", grid__code="MGRS-33TUL")

    assert acquisition_key(item) == ("sentinel-2-l2a", SENSED, "MGRS-33TUL")


def test_sub_second_differences_do_not_split_an_acquisition():
    one = FakeItem("a", timestamp=SENSED, grid__code="MGRS-33TUL")
    other = FakeItem("b", timestamp=SENSED.replace(microsecond=500), grid__code="MGRS-33TUL")

    assert acquisition_key(one) == acquisition_key(other)


def test_the_older_mgrs_field_is_read_where_there_is_no_grid_code():
    assert acquisition_key(FakeItem("a", s2__mgrs_tile="33TUL"))[2] == "33TUL"


def test_a_landsat_path_and_row_identify_the_ground():
    item = FakeItem("a", landsat__wrs_path="192", landsat__wrs_row="029")

    assert acquisition_key(item)[2] == "WRS2-192029"


def test_the_footprint_stands_in_where_no_grid_is_declared():
    item = FakeItem("a", bbox=(11.0, 46.5, 11.2, 46.7))

    assert acquisition_key(item)[2] == (11.0, 46.5, 11.2, 46.7)


def test_an_undated_item_has_no_key():
    assert acquisition_key(FakeItem("a", timestamp=None)) is None


# ------------------------------
# Deduplicating a list of items
# ------------------------------
def test_the_reprocessed_copy_survives():
    original = FakeItem("original", grid__code="MGRS-33TUL", processing__version="02.12")
    reprocessed = FakeItem("reprocessed", grid__code="MGRS-33TUL", processing__version="05.11")

    assert ids(deduplicate_items([original, reprocessed])) == ["reprocessed"]
    # And the arrival order must not decide it either way round.
    assert ids(deduplicate_items([reprocessed, original])) == ["reprocessed"]


def test_two_tiles_of_one_overpass_are_both_kept():
    # The case that must not collapse: for an area straddling a tile boundary
    # each item holds a different half of it.
    west = FakeItem("west", grid__code="MGRS-33TUL")
    east = FakeItem("east", grid__code="MGRS-33TUM")

    assert ids(deduplicate_items([west, east])) == ["west", "east"]


def test_two_collections_are_never_weighed_against_each_other():
    s2 = FakeItem("s2", collection="sentinel-2-l2a")
    landsat = FakeItem("landsat", collection="landsat-c2-l2")

    assert len(deduplicate_items([s2, landsat])) == 2


def test_different_acquisitions_are_all_kept():
    items = [
        FakeItem(f"day{day}", timestamp=SENSED.replace(day=day), grid__code="MGRS-33TUL")
        for day in (12, 17, 22)
    ]

    assert len(deduplicate_items(items)) == 3


def test_surviving_items_keep_their_arrival_order():
    items = [
        FakeItem("may", timestamp=SENSED.replace(month=5), grid__code="MGRS-33TUL"),
        FakeItem("april-v1", grid__code="MGRS-33TUL", processing__version="02.12"),
        FakeItem("june", timestamp=SENSED.replace(month=6), grid__code="MGRS-33TUL"),
        FakeItem("april-v2", grid__code="MGRS-33TUL", processing__version="05.11"),
    ]

    assert ids(deduplicate_items(items)) == ["may", "june", "april-v2"]


def test_an_undated_item_is_always_kept():
    dated = FakeItem("dated", grid__code="MGRS-33TUL")
    undated = FakeItem("undated", timestamp=None)

    assert ids(deduplicate_items([dated, undated])) == ["dated", "undated"]


def test_deduplicating_nothing_returns_nothing():
    assert deduplicate_items([]) == []


# ------------------------------
# On a search result
# ------------------------------
def result(items, **kwargs):
    return STACSearchResult(
        items, collections=["sentinel-2-l2a"], catalog="https://example.invalid", **kwargs
    )


def test_a_search_result_deduplicates_to_a_search_result():
    original = FakeItem("original", grid__code="MGRS-33TUL", processing__version="02.12")
    reprocessed = FakeItem("reprocessed", grid__code="MGRS-33TUL", processing__version="05.11")

    deduplicated = result([original, reprocessed]).deduplicate()

    assert isinstance(deduplicated, STACSearchResult)
    assert ids(deduplicated) == ["reprocessed"]


def test_deduplicating_keeps_the_search_metadata_so_loading_still_crops():
    bbox = (11.0, 46.5, 11.2, 46.7)
    items = [FakeItem("a", grid__code="MGRS-33TUL"), FakeItem("b", grid__code="MGRS-33TUL")]

    deduplicated = result(items, bbox=bbox).deduplicate()

    assert deduplicated.bbox == bbox
    assert deduplicated.collections == ["sentinel-2-l2a"]
    assert deduplicated.catalog == "https://example.invalid"


def test_deduplicating_leaves_the_original_result_alone():
    items = [
        FakeItem("a", grid__code="MGRS-33TUL", processing__version="02.12"),
        FakeItem("b", grid__code="MGRS-33TUL", processing__version="05.11"),
    ]
    original = result(items)

    original.deduplicate()

    assert len(original) == 2
