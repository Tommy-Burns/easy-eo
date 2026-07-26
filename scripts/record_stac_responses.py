#!/usr/bin/env python
"""Record trimmed STAC API responses for the offline test suite.

Maintainer tool. The STAC tests replay these recordings through the real
pystac-client, so the fixtures must stay faithful to what the catalogs
actually serve - hand-editing them defeats the point. Refresh with:

    python scripts/record_stac_responses.py

Each catalog yields two files in ``tests/data/stac/``: the landing page
(``*_root.json``, which pystac-client reads to discover the search endpoint and
conformance classes) and one page of search results (``*_search_page.json``).

Responses are trimmed - a handful of assets and properties per item - to keep
the fixtures readable in review and small in the repo. Hrefs are left exactly as
served; catalog hrefs are unsigned, so no credentials end up in the repo.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "data" / "stac"

# Links worth keeping on a landing page: pystac-client needs "search" to find
# the endpoint and "self"/"root" to resolve relative hrefs. Dropping "child"
# links removes one entry per collection (over a hundred on some catalogs).
KEPT_ROOT_RELS = {"self", "root", "search", "data", "conformance"}

# Properties kept on each item: the ones Easy-EO reads, plus enough context for
# a fixture to still look like a real scene in review.
KEPT_PROPERTIES = (
    "datetime",
    "start_datetime",
    "end_datetime",
    "eo:cloud_cover",
    "platform",
    "constellation",
    "instruments",
    "gsd",
    "proj:epsg",
    "proj:code",
)

CATALOGS: dict[str, dict[str, Any]] = {
    "pc": {
        "url": "https://planetarycomputer.microsoft.com/api/stac/v1",
        "collection": "sentinel-2-l2a",
        # Two 10 m bands and one 20 m band, so a fixture can exercise both the
        # aligned read and the resample-onto-the-grid path.
        "assets": ["B04", "B08", "B11", "SCL"],
    },
    "earthsearch": {
        "url": "https://earth-search.aws.element84.com/v1",
        "collection": "sentinel-2-l2a",
        "assets": ["red", "nir", "swir16", "scl"],
    },
}

BBOX = [11.0, 46.5, 11.2, 46.7]
FEATURE_COUNT = 2


def _get(url: str, payload: dict | None = None) -> dict:
    """Fetch JSON, POSTing ``payload`` when given."""
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return json.load(response)


def trim_root(document: dict) -> dict:
    """Drop the per-collection child links from a landing page."""
    trimmed = dict(document)
    trimmed["links"] = [
        link for link in document.get("links", []) if link.get("rel") in KEPT_ROOT_RELS
    ]
    return trimmed


def trim_feature(feature: dict, assets: list[str]) -> dict:
    """Keep a few assets and properties; drop links and per-asset extras."""
    properties = {
        key: value for key, value in feature.get("properties", {}).items() if key in KEPT_PROPERTIES
    }
    kept_assets = {}
    for name in assets:
        asset = feature.get("assets", {}).get(name)
        if asset is None:
            continue
        kept_assets[name] = {
            key: value for key, value in asset.items() if key in ("href", "type", "title", "roles")
        }
    return {
        "type": feature.get("type", "Feature"),
        "stac_version": feature.get("stac_version", "1.0.0"),
        "id": feature["id"],
        "collection": feature.get("collection"),
        "geometry": feature.get("geometry"),
        "bbox": feature.get("bbox"),
        "properties": properties,
        "assets": kept_assets,
        "links": [],
    }


def fetch_page(catalog: dict) -> dict:
    """Fetch one page of results, preferring the search endpoint."""
    url, collection = catalog["url"], catalog["collection"]
    body = {"collections": [collection], "bbox": BBOX, "limit": FEATURE_COUNT}
    try:
        page = _get(f"{url}/search", body)
    except urllib.error.HTTPError as err:
        # The items endpoint returns the same FeatureCollection shape and is a
        # usable stand-in while a catalog's search backend is unavailable.
        print(f"  search endpoint returned HTTP {err.code}; falling back to /items")
        query = f"?limit={FEATURE_COUNT}&bbox={','.join(str(v) for v in BBOX)}"
        page = _get(f"{url}/collections/{collection}/items{query}")

    features = [trim_feature(feature, catalog["assets"]) for feature in page["features"]]
    return {
        "type": "FeatureCollection",
        "stac_version": page.get("stac_version", "1.0.0"),
        "features": features,
        # No "next" link: one page, so a replay terminates.
        "links": [],
    }


def main() -> None:
    """Record every catalog's landing page and one page of results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, catalog in CATALOGS.items():
        print(f"recording {name} ({catalog['url']})")
        for suffix, document in (
            ("root", trim_root(_get(catalog["url"]))),
            ("search_page", fetch_page(catalog)),
        ):
            path = OUTPUT_DIR / f"{name}_{suffix}.json"
            path.write_text(json.dumps(document, indent=2) + "\n")
            print(f"  wrote {path.relative_to(OUTPUT_DIR.parent.parent.parent)}")


if __name__ == "__main__":
    main()
