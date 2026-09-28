"""Choosing between several processings of one acquisition.

A catalog publishes a scene more than once: the original processing, then
whatever reprocessing campaigns have swept the archive since. Every copy matches
a search, which is correct of the catalog and wrong for a time series — two
copies of one morning weight that morning twice in a median, and the composite
is quietly pulled toward whichever dates happen to be duplicated.

Deciding which copy wins needs a rule, and the rule is stated once here because
the same question is asked of catalog items (where the metadata is richest) and
of a series already built from them.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from typing import Any

_UTC = dt.timezone.utc

# Never older than any real processing time, so a copy that states one always
# outranks a copy that states none.
_UNDATED = dt.datetime.min.replace(tzinfo=_UTC)

# The processing extension's own field, the deprecated Sentinel-2 spelling, and
# the name the Sentinel-2 loaders and eeo/io/stac.py record it under in attrs.
# https://github.com/stac-extensions/processing
_VERSION_KEYS = ("processing:version", "s2:processing_baseline", "processing_baseline")

# `processing:datetime` is "processing date and time of the corresponding data"
# — a statement about this copy of the data. `created` and `updated` in an
# item's properties are about the *metadata* record, not the data (STAC common
# metadata says so explicitly), so they only stand in where the direct field is
# absent.
# https://github.com/radiantearth/stac-spec/blob/master/commons/common-metadata.md
# `processed_at` is the name eeo/io/stac.py records the resolved value under on a
# loaded scene, so a series can be deduplicated by the same rule as a search.
_PROCESSED_AT_KEYS = ("processing:datetime", "updated", "created", "processed_at")

# Which piece of ground the scene covers. `grid:code` is the grid extension's
# field, spelled "MGRS-33TUL" for Sentinel-2 and "WRS2-192029" for Landsat;
# `s2:mgrs_tile` is what catalogs published before it.
# https://github.com/stac-extensions/grid
_GRID_KEYS = ("grid:code", "s2:mgrs_tile")


def version_rank(properties: Mapping[str, Any]) -> tuple[int, ...]:
    """Return a processing version as a comparable tuple, or ``()`` if unreadable.

    Compared as numbers rather than text: Sentinel-2 baselines are written
    zero-padded, and as strings ``"05.11"`` sorts below ``"5.2"``. A version
    that is not a dot-separated number contributes nothing to the ordering
    instead of contributing a wrong answer.

    Parameters
    ----------
    properties : mapping
        STAC item properties, or a dataset's ``attrs``. Read from
        ``processing:version``, then ``s2:processing_baseline``, then
        ``processing_baseline``, whichever is present first.

    Returns
    -------
    tuple of int
        The version's dot-separated parts as integers — ``"05.11"`` is
        ``(5, 11)`` — or ``()`` when there is no version, or none that reads as
        numbers.
    """
    for key in _VERSION_KEYS:
        value = properties.get(key)
        if value is None:
            continue
        parts = str(value).strip().split(".")
        if parts and all(part.isdigit() for part in parts):
            return tuple(int(part) for part in parts)
    return ()


def processed_at(properties: Mapping[str, Any]) -> dt.datetime | None:
    """Return when this copy was processed, as UTC, or None if it does not say.

    Parameters
    ----------
    properties : mapping
        STAC item properties, or a dataset's ``attrs``. Read from
        ``processing:datetime``, then ``updated``, then ``created``, then
        ``processed_at``, whichever first holds a readable time.

    Returns
    -------
    datetime.datetime or None
        The processing time, timezone-aware in UTC (a naive value is read as
        UTC), or None when no field holds one.
    """
    for key in _PROCESSED_AT_KEYS:
        value = properties.get(key)
        if isinstance(value, dt.datetime):
            return value if value.tzinfo is not None else value.replace(tzinfo=_UTC)
        if not isinstance(value, str) or not value.strip():
            continue
        text = value.strip()
        # RFC 3339 allows the "Z" that fromisoformat only learned in 3.11.
        if text.endswith(("Z", "z")):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = dt.datetime.fromisoformat(text)
        except ValueError:
            continue
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=_UTC)
    return None


def processing_rank(
    properties: Mapping[str, Any], arrival: int
) -> tuple[tuple[int, ...], dt.datetime, int]:
    """Rank one copy of an acquisition; the largest rank in a group wins.

    The documented rule, in order:

    1. **The highest processing version.** For Sentinel-2 that is the
       processing baseline, which is the one thing that actually describes the
       pixels — a later baseline is a better atmospheric correction, and 04.00
       onward also stores its values differently.
    2. **Then the most recent processing time.** It settles two copies at one
       version, and carries the decision on its own for a mission that
       publishes no version.
    3. **Then whichever arrived first**, so a result that distinguishes nothing
       is left in the order the catalog gave it rather than shuffled.

    Version comes before time because ``updated``, which stands in where a
    processing time is absent, is a statement about the metadata record: a
    metadata fix can make an older processing look newer, and it must not be
    allowed to beat a genuinely better one.

    Parameters
    ----------
    properties : mapping
        STAC item properties, or a dataset's ``attrs``.
    arrival : int
        Position of this copy in the order it was received, which breaks a
        total tie in favour of the earliest.

    Returns
    -------
    tuple
        ``(version, processing time, -arrival)``, comparable with ``>`` against
        the rank of any other copy of the same acquisition.
    """
    return (version_rank(properties), processed_at(properties) or _UNDATED, -arrival)


def _ground(item: Any) -> Any:
    """Return what the item says about which ground it covers.

    Two copies of one acquisition cover the same ground; two tiles of one
    overpass do not, and must never be collapsed into each other — for an area
    straddling a tile boundary they hold different halves of it. The grid code
    says which tile outright. Failing that, the footprint does: a reprocessing
    keeps it, adjacent tiles differ in it, and rounding leaves room for a
    footprint that was recomputed slightly.
    """
    properties = item.properties
    for key in _GRID_KEYS:
        value = properties.get(key)
        if value not in (None, ""):
            return str(value)

    path = properties.get("landsat:wrs_path")
    row = properties.get("landsat:wrs_row")
    if path not in (None, "") and row not in (None, ""):
        return f"WRS2-{path}{row}"

    bbox = item.bbox
    return None if bbox is None else tuple(round(float(value), 4) for value in bbox)


def acquisition_key(item: Any) -> tuple[Any, ...] | None:
    """Return what makes two items the same acquisition, or None if undecidable.

    One moment, one collection, one piece of ground. The collection is in the
    key because a search may cover several, and a Landsat scene must not be
    weighed against a Sentinel-2 one; the timestamp is taken to the second,
    which is how a sensing time is published.

    Parameters
    ----------
    item : STACItem
        Catalog item, read for its ``timestamp``, ``collection``, ``bbox`` and
        ``properties``.

    Returns
    -------
    tuple or None
        ``(collection, acquisition time to the second, ground)``, equal for two
        copies of one acquisition and different for two tiles of one overpass.
        None for an item with no acquisition time, which cannot be placed in
        time at all and so cannot be found to duplicate anything.
    """
    stamp = item.timestamp
    if stamp is None:
        return None
    return (item.collection, stamp.replace(microsecond=0), _ground(item))


def deduplicate_items(items: Sequence[Any]) -> list[Any]:
    """Keep one item per acquisition, by the rule :func:`processing_rank` states.

    Parameters
    ----------
    items : sequence of STACItem
        Catalog items, in the order they were received.

    Returns
    -------
    list of STACItem
        The surviving items, in the order they arrived, minus the copies that
        lost. An item with no acquisition time is always kept: it cannot be
        shown to duplicate anything.
    """
    Rank = tuple[tuple[int, ...], dt.datetime, int]
    winners: dict[tuple[Any, ...], tuple[Rank, int, Any]] = {}
    undated: list[tuple[int, Any]] = []

    for arrival, item in enumerate(items):
        key = acquisition_key(item)
        if key is None:
            undated.append((arrival, item))
            continue
        rank = processing_rank(item.properties, arrival)
        standing = winners.get(key)
        if standing is None or rank > standing[0]:
            winners[key] = (rank, arrival, item)

    kept = [(arrival, item) for _, arrival, item in winners.values()] + undated
    kept.sort(key=lambda pair: pair[0])
    return [item for _, item in kept]
