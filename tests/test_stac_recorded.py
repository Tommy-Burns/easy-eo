"""Replay recorded STAC responses through the real pystac-client.

``tests/test_stac_search.py`` covers Easy-EO's own logic against a fake client.
These tests go one layer deeper: real ``pystac_client`` and real ``pystac``
objects, driven by responses recorded from live catalogs (see
``tests/data/stac/README.md``), with only the HTTP transport replaced. That is
what catches a wrong request body or an attribute Easy-EO reads that real STAC
items do not have — without any network access.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio as rio
from rasterio.transform import from_origin

import eeo
from eeo.io import PLANETARY_COMPUTER_STAC_URL

requests = pytest.importorskip("requests", reason="needs the stac extra")
pytest.importorskip("pystac_client", reason="needs the stac extra")

RECORDINGS = Path(__file__).parent / "data" / "stac"
EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"


def recording(name):
    return json.loads((RECORDINGS / f"{name}.json").read_text())


class ReplayTransport:
    """Serves recorded documents in place of requests' HTTP transport."""

    def __init__(self, root, page, *, page_status=200):
        self.root = root
        self.page = page
        self.page_status = page_status
        self.requests = []

    # Assigned onto the Session class; an instance is not a descriptor, so no
    # `self` for the session is passed in - only the prepared request.
    def __call__(self, prepared, **kwargs):
        self.requests.append(prepared)
        if "/search" in prepared.url:
            return self._response(self.page, prepared.url, self.page_status)
        return self._response(self.root, prepared.url)

    @staticmethod
    def _response(payload, url, status=200):
        response = requests.Response()
        response.status_code = status
        response._content = json.dumps(payload).encode()
        response.url = url
        response.headers["Content-Type"] = "application/json"
        return response

    @property
    def search_request(self):
        return next(request for request in self.requests if "/search" in request.url)


@pytest.fixture
def planetary_computer(monkeypatch):
    transport = ReplayTransport(recording("pc_root"), recording("pc_search_page"))
    monkeypatch.setattr(requests.Session, "send", transport)
    return transport


@pytest.fixture
def earth_search(monkeypatch):
    transport = ReplayTransport(recording("earthsearch_root"), recording("earthsearch_search_page"))
    monkeypatch.setattr(requests.Session, "send", transport)
    return transport


# --------------------------------------------------------------------------
# What Easy-EO puts on the wire
# --------------------------------------------------------------------------
def test_search_sends_the_filters_the_stac_api_expects(planetary_computer):
    eeo.stac_search(
        "sentinel-2-l2a",
        bbox=(11.0, 46.5, 11.2, 46.7),
        datetime="2026-07-01/2026-07-31",
        cloud_cover=20,
        sign=False,
    )

    request = planetary_computer.search_request
    assert request.method == "POST"
    body = json.loads(request.body)
    assert body["collections"] == ["sentinel-2-l2a"]
    assert body["bbox"] == [11.0, 46.5, 11.2, 46.7]
    # pystac-client expands a bare date range into a full RFC 3339 interval,
    # and the closing date covers its whole day - so the end is inclusive.
    assert body["datetime"] == "2026-07-01T00:00:00Z/2026-07-31T23:59:59Z"
    assert body["query"] == {"eo:cloud_cover": {"lte": 20.0}}


def test_search_reaches_the_catalogs_search_endpoint(planetary_computer):
    eeo.stac_search("sentinel-2-l2a", sign=False)

    assert planetary_computer.search_request.url.startswith(PLANETARY_COMPUTER_STAC_URL)
    assert planetary_computer.search_request.url.endswith("/search")


def test_limit_caps_the_items_returned_not_the_page_size(planetary_computer):
    # The recorded page holds two items; `limit` has to cut the result to one.
    results = eeo.stac_search("sentinel-2-l2a", limit=1, sign=False)

    assert len(results) == 1


# --------------------------------------------------------------------------
# What Easy-EO reads back off real pystac items
# --------------------------------------------------------------------------
def test_real_items_are_wrapped_and_ordered_oldest_first(planetary_computer):
    results = eeo.stac_search("sentinel-2-l2a", sign=False)

    assert len(results) == 2
    # The catalog served these newest-first; Easy-EO re-orders them.
    assert results.timestamps == sorted(results.timestamps)
    assert results[0].id == "S2A_MSIL2A_20260721T102041_R065_T32TPS_20260721T170018"
    assert results[1].id == "S2B_MSIL2A_20260724T101559_R065_T32TPS_20260724T141313"


def test_real_item_metadata_is_read_correctly(planetary_computer):
    item = eeo.stac_search("sentinel-2-l2a", sign=False)[0]

    assert item.collection == "sentinel-2-l2a"
    assert item.timestamp.tzinfo is not None
    assert item.timestamp.year == 2026
    assert item.asset_names == ["B04", "B08", "B11", "SCL"]
    assert isinstance(item.cloud_cover, float)
    assert item.bbox is not None and len(item.bbox) == 4
    # The wrapper does not hide the real thing.
    assert type(item.item).__name__ == "Item"


def test_a_different_catalog_parses_the_same_way(earth_search):
    results = eeo.stac_search("sentinel-2-l2a", catalog=EARTH_SEARCH_URL)

    assert len(results) == 2
    assert results.catalog == EARTH_SEARCH_URL
    assert results[0].asset_names == ["red", "nir", "swir16", "scl"]
    assert results.timestamps == sorted(results.timestamps)


def test_recorded_asset_hrefs_are_usable_urls(planetary_computer):
    item = eeo.stac_search("sentinel-2-l2a", sign=False)[0]

    href = item.assets["B04"].href
    assert href.startswith("https://")
    # Recordings must never carry credentials into the repo.
    assert "?" not in href


# --------------------------------------------------------------------------
# Loading from a real item
# --------------------------------------------------------------------------
def test_load_reads_assets_of_a_real_item(planetary_computer, tmp_path):
    item = eeo.stac_search("sentinel-2-l2a", bbox=(11.0, 46.5, 11.2, 46.7), sign=False)[0]

    # Point the real pystac assets at local rasters: the remote read itself
    # needs the network, everything around it does not.
    transform = from_origin(660000.0, 5180000.0, 10.0, 10.0)
    for name, value in (("B04", 1200), ("B08", 3400)):
        path = tmp_path / f"{name}.tif"
        with rio.open(
            path,
            "w",
            driver="GTiff",
            height=100,
            width=100,
            count=1,
            dtype="uint16",
            crs="EPSG:32632",
            transform=transform,
            nodata=0,
        ) as dst:
            dst.write(np.full((1, 100, 100), value, dtype="uint16"))
        item.assets[name].href = str(path)

    scene = item.load(["B04", "B08"], crop=False)

    assert scene.band_names == ["B04", "B08"]
    assert scene.get_count() == 2
    assert scene.timestamp == item.timestamp
    assert scene.attrs["stac_item"] == item.id
    assert np.all(scene.to_array()[1] == 3400)


# --------------------------------------------------------------------------
# Catalog failures
# --------------------------------------------------------------------------
def test_a_failing_catalog_surfaces_the_clients_error(monkeypatch):
    # Documents today's contract: transport errors propagate from
    # pystac-client rather than being swallowed.
    transport = ReplayTransport(
        recording("pc_root"), {"message": "Gateway Timeout"}, page_status=504
    )
    monkeypatch.setattr(requests.Session, "send", transport)

    from pystac_client.exceptions import APIError

    with pytest.raises(APIError):
        eeo.stac_search("sentinel-2-l2a", sign=False)
