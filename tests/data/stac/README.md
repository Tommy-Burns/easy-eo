# Recorded STAC responses

Real responses from live STAC APIs, trimmed for size, replayed by
`tests/test_stac_recorded.py` so the search path can be exercised through the
actual `pystac-client` without touching the network.

| File | Source |
|---|---|
| `pc_root.json` | `GET https://planetarycomputer.microsoft.com/api/stac/v1` |
| `pc_search_page.json` | `POST .../api/stac/v1/search` (sentinel-2-l2a, 2 items) |
| `earthsearch_root.json` | `GET https://earth-search.aws.element84.com/v1` |
| `earthsearch_search_page.json` | `POST .../v1/search` (sentinel-2-l2a, 2 items) |

Recorded 2026-07-26 over the bbox `(11.0, 46.5, 11.2, 46.7)`.

Trimming keeps a few assets and properties per item and drops the landing
pages' per-collection `child` links; everything kept is byte-for-byte what the
catalog served, including asset hrefs (catalog hrefs are unsigned, so no
credentials live here). Both pages are ordered newest-first, as the catalogs
return them, which is what lets the tests prove Easy-EO re-orders results
chronologically.

Do not hand-edit these files — a fixture that drifts from what the catalogs
actually serve stops being evidence. Refresh them with:

    python scripts/record_stac_responses.py
