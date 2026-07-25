"""Tests for eeo.stac_search (eeo/io/stac.py).

Every test runs against an injected fake ``pystac_client`` / ``planetary_computer``
module, so the suite exercises the real search code path with zero network
access and works whether or not the ``stac`` extra is installed.
"""

import datetime as dt
import importlib
import sys
import types

import pytest

import eeo
from eeo.io import PLANETARY_COMPUTER_STAC_URL, STACItem, STACSearchResult

UTC = dt.timezone.utc


# --------------------------------------------------------------------------
# Fake STAC catalog
# --------------------------------------------------------------------------
class FakeAsset:
    def __init__(self, href):
        self.href = href


class FakeItem:
    def __init__(self, item_id, timestamp, *, properties=None, assets=None, collection="col"):
        self.id = item_id
        self.datetime = timestamp
        self.collection_id = collection
        self.properties = dict(properties or {})
        self.assets = assets if assets is not None else {"B04": FakeAsset("https://x/B04.tif")}
        self.bbox = [11.0, 46.5, 11.2, 46.7]


class FakeSearch:
    def __init__(self, items):
        self._items = items

    def items(self):
        return iter(self._items)


class FakeClient:
    """Records how it was opened and searched so tests can assert on it."""

    opened = []
    searched = []
    items_to_return = []

    def __init__(self, url, modifier):
        self.url = url
        self.modifier = modifier

    @classmethod
    def open(cls, url, modifier=None):
        client = cls(url, modifier)
        cls.opened.append(client)
        return client

    def search(self, **params):
        FakeClient.searched.append(params)
        return FakeSearch(list(FakeClient.items_to_return))


def _sign_inplace(item):  # pragma: no cover - identity stand-in for the real signer
    return item


@pytest.fixture
def fake_stac(monkeypatch):
    """Install fake pystac_client / planetary_computer modules."""
    FakeClient.opened = []
    FakeClient.searched = []
    FakeClient.items_to_return = []

    pystac_client = types.ModuleType("pystac_client")
    pystac_client.Client = FakeClient
    planetary_computer = types.ModuleType("planetary_computer")
    planetary_computer.sign_inplace = _sign_inplace

    monkeypatch.setitem(sys.modules, "pystac_client", pystac_client)
    monkeypatch.setitem(sys.modules, "planetary_computer", planetary_computer)
    return FakeClient


def item(item_id, timestamp, **kwargs):
    return FakeItem(item_id, timestamp, **kwargs)


# --------------------------------------------------------------------------
# Query construction
# --------------------------------------------------------------------------
def test_search_passes_collection_bbox_datetime_and_cloud_cover(fake_stac):
    eeo.stac_search(
        "sentinel-2-l2a",
        bbox=(11.0, 46.5, 11.2, 46.7),
        datetime="2023-06-01/2023-08-31",
        cloud_cover=20,
        limit=5,
    )

    params = fake_stac.searched[0]
    assert params["collections"] == ["sentinel-2-l2a"]
    assert params["bbox"] == [11.0, 46.5, 11.2, 46.7]
    assert params["datetime"] == "2023-06-01/2023-08-31"
    assert params["query"] == {"eo:cloud_cover": {"lte": 20.0}}
    # `limit` caps the total item count, not the page size.
    assert params["max_items"] == 5
    assert "limit" not in params


def test_search_accepts_several_collections(fake_stac):
    eeo.stac_search(["sentinel-2-l2a", "landsat-c2-l2"])

    assert fake_stac.searched[0]["collections"] == ["sentinel-2-l2a", "landsat-c2-l2"]


def test_search_omits_unset_filters(fake_stac):
    eeo.stac_search("sentinel-2-l2a")

    assert fake_stac.searched[0] == {"collections": ["sentinel-2-l2a"]}


def test_search_accepts_datetime_objects_and_pairs(fake_stac):
    start, end = dt.datetime(2023, 6, 1, tzinfo=UTC), dt.datetime(2023, 8, 31, tzinfo=UTC)

    eeo.stac_search("sentinel-2-l2a", datetime=(start, end))

    assert fake_stac.searched[0]["datetime"] == (start, end)


def test_search_defaults_to_planetary_computer(fake_stac):
    eeo.stac_search("sentinel-2-l2a")

    assert fake_stac.opened[0].url == PLANETARY_COMPUTER_STAC_URL


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["", [], ["sentinel-2-l2a", ""], 3])
def test_invalid_collection_raises_validation_error(fake_stac, bad):
    with pytest.raises(eeo.ValidationError, match="collection"):
        eeo.stac_search(bad)


@pytest.mark.parametrize(
    "bad",
    [
        (11.0, 46.5, 11.2),  # too few values
        (11.0, 46.5, 11.2, 46.7, 1.0),  # too many
        (11.2, 46.5, 11.0, 46.7),  # minx >= maxx
        (11.0, 46.7, 11.2, 46.5),  # miny >= maxy
        "11,46,12,47",  # a string is not a bbox
    ],
)
def test_invalid_bbox_raises_validation_error(fake_stac, bad):
    with pytest.raises(eeo.ValidationError, match="bbox"):
        eeo.stac_search("sentinel-2-l2a", bbox=bad)


@pytest.mark.parametrize("bad", [-1, 101])
def test_invalid_cloud_cover_raises_validation_error(fake_stac, bad):
    with pytest.raises(eeo.ValidationError, match="cloud_cover"):
        eeo.stac_search("sentinel-2-l2a", cloud_cover=bad)


@pytest.mark.parametrize("good", [0, 100])
def test_cloud_cover_bounds_are_inclusive(fake_stac, good):
    eeo.stac_search("sentinel-2-l2a", cloud_cover=good)

    assert fake_stac.searched[0]["query"] == {"eo:cloud_cover": {"lte": float(good)}}


@pytest.mark.parametrize("bad", [0, -3, 2.5, True])
def test_invalid_limit_raises_validation_error(fake_stac, bad):
    with pytest.raises(eeo.ValidationError, match="limit"):
        eeo.stac_search("sentinel-2-l2a", limit=bad)


def test_validation_happens_before_any_network_call(fake_stac):
    with pytest.raises(eeo.ValidationError):
        eeo.stac_search("sentinel-2-l2a", cloud_cover=200)

    assert fake_stac.opened == []


# --------------------------------------------------------------------------
# Ordering and timestamps (Milestone 3 constraint)
# --------------------------------------------------------------------------
def test_results_are_ordered_oldest_first(fake_stac):
    fake_stac.items_to_return = [
        item("b", dt.datetime(2023, 7, 1, tzinfo=UTC)),
        item("c", dt.datetime(2023, 8, 1, tzinfo=UTC)),
        item("a", dt.datetime(2023, 6, 1, tzinfo=UTC)),
    ]

    results = eeo.stac_search("sentinel-2-l2a")

    assert [r.id for r in results] == ["a", "b", "c"]
    assert results.timestamps == [
        dt.datetime(2023, 6, 1, tzinfo=UTC),
        dt.datetime(2023, 7, 1, tzinfo=UTC),
        dt.datetime(2023, 8, 1, tzinfo=UTC),
    ]


def test_naive_and_aware_timestamps_sort_together(fake_stac):
    fake_stac.items_to_return = [
        item("aware", dt.datetime(2023, 7, 1, tzinfo=UTC)),
        item("naive", dt.datetime(2023, 6, 1)),
    ]

    results = eeo.stac_search("sentinel-2-l2a")

    assert [r.id for r in results] == ["naive", "aware"]


def test_undated_items_sort_last_and_report_none(fake_stac):
    fake_stac.items_to_return = [
        item("undated", None),
        item("dated", dt.datetime(2023, 6, 1, tzinfo=UTC)),
    ]

    results = eeo.stac_search("sentinel-2-l2a")

    assert [r.id for r in results] == ["dated", "undated"]
    assert results[-1].timestamp is None


def test_range_items_fall_back_to_start_datetime(fake_stac):
    fake_stac.items_to_return = [
        item("range", None, properties={"start_datetime": "2023-06-01T10:00:00Z"})
    ]

    results = eeo.stac_search("sentinel-2-l2a")

    assert results[0].timestamp == dt.datetime(2023, 6, 1, 10, 0, tzinfo=UTC)


def test_string_timestamps_are_parsed(fake_stac):
    fake_stac.items_to_return = [item("s", "2023-06-05T10:06:21Z")]

    results = eeo.stac_search("sentinel-2-l2a")

    assert results[0].timestamp == dt.datetime(2023, 6, 5, 10, 6, 21, tzinfo=UTC)


def test_unparseable_timestamp_becomes_none_instead_of_failing(fake_stac):
    fake_stac.items_to_return = [item("bad", "not-a-date")]

    results = eeo.stac_search("sentinel-2-l2a")

    assert results[0].timestamp is None


# --------------------------------------------------------------------------
# Result and item surface
# --------------------------------------------------------------------------
def test_empty_result_is_not_an_error(fake_stac):
    results = eeo.stac_search("sentinel-2-l2a")

    assert isinstance(results, STACSearchResult)
    assert len(results) == 0
    assert list(results) == []
    assert "0 items" in repr(results)


def test_result_is_a_sequence(fake_stac):
    fake_stac.items_to_return = [
        item(name, dt.datetime(2023, 6, day, tzinfo=UTC))
        for day, name in enumerate(["a", "b", "c"], start=1)
    ]

    results = eeo.stac_search("sentinel-2-l2a")

    assert len(results) == 3
    assert isinstance(results[0], STACItem)
    assert [r.id for r in results] == ["a", "b", "c"]
    assert results.collections == ["sentinel-2-l2a"]
    assert results.catalog == PLANETARY_COMPUTER_STAC_URL
    assert "3 items" in repr(results)
    assert "2023-06-01 to 2023-06-03" in repr(results)


def test_result_retains_the_search_area_of_interest(fake_stac):
    results = eeo.stac_search("sentinel-2-l2a", bbox=(11.0, 46.5, 11.2, 46.7))

    # Retained so a later asset read can crop to the same AOI.
    assert results.bbox == (11.0, 46.5, 11.2, 46.7)
    assert results[:1].bbox == (11.0, 46.5, 11.2, 46.7)
    assert eeo.stac_search("sentinel-2-l2a").bbox is None


def test_slicing_returns_a_search_result(fake_stac):
    fake_stac.items_to_return = [
        item(name, dt.datetime(2023, 6, day, tzinfo=UTC))
        for day, name in enumerate(["a", "b", "c"], start=1)
    ]

    subset = eeo.stac_search("sentinel-2-l2a")[:2]

    assert isinstance(subset, STACSearchResult)
    assert [r.id for r in subset] == ["a", "b"]
    assert subset.collections == ["sentinel-2-l2a"]


def test_item_exposes_metadata_and_the_raw_item(fake_stac):
    raw = item(
        "S2A_TEST",
        dt.datetime(2023, 6, 5, 10, 6, 21, tzinfo=UTC),
        properties={"eo:cloud_cover": 4.2, "platform": "sentinel-2a"},
        assets={"B04": FakeAsset("https://x/B04.tif"), "B08": FakeAsset("https://x/B08.tif")},
        collection="sentinel-2-l2a",
    )
    fake_stac.items_to_return = [raw]

    found = eeo.stac_search("sentinel-2-l2a")[0]

    assert found.id == "S2A_TEST"
    assert found.collection == "sentinel-2-l2a"
    assert found.cloud_cover == pytest.approx(4.2)
    assert found.properties["platform"] == "sentinel-2a"
    assert found.asset_names == ["B04", "B08"]
    assert found.assets["B08"].href == "https://x/B08.tif"
    assert found.bbox == (11.0, 46.5, 11.2, 46.7)
    assert found.item is raw
    assert "S2A_TEST" in repr(found)


def test_item_without_cloud_cover_reports_none(fake_stac):
    fake_stac.items_to_return = [item("a", dt.datetime(2023, 6, 1, tzinfo=UTC))]

    assert eeo.stac_search("sentinel-2-l2a")[0].cloud_cover is None


# --------------------------------------------------------------------------
# Asset-URL signing
# --------------------------------------------------------------------------
def test_planetary_computer_assets_are_signed_by_default(fake_stac):
    eeo.stac_search("sentinel-2-l2a")

    assert fake_stac.opened[0].modifier is _sign_inplace


def test_signing_can_be_disabled(fake_stac):
    eeo.stac_search("sentinel-2-l2a", sign=False)

    assert fake_stac.opened[0].modifier is None


def test_other_catalogs_are_not_signed(fake_stac):
    eeo.stac_search("sentinel-2-l2a", catalog="https://earth-search.aws.element84.com/v1")

    assert fake_stac.opened[0].modifier is None


def test_signing_can_be_forced_for_any_catalog(fake_stac):
    eeo.stac_search("sentinel-2-l2a", catalog="https://example.com/stac", sign=True)

    assert fake_stac.opened[0].modifier is _sign_inplace


# --------------------------------------------------------------------------
# Missing extra
# --------------------------------------------------------------------------
def test_search_without_the_stac_extra_raises_missing_dependency_error(monkeypatch):
    real_import_module = importlib.import_module

    def fake_import_module(name):
        if name == "pystac_client":
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real_import_module(name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(eeo.MissingDependencyError, match=r"pip install 'easy-eo\[stac\]'"):
        eeo.stac_search("sentinel-2-l2a")
