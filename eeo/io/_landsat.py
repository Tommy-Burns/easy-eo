"""Identifying and reading the metadata of a Landsat Collection 2 product.

A downloaded Landsat product is a directory of single-band GeoTIFFs beside one
metadata file, written three times over in different syntaxes: ``*_MTL.xml``,
``*_MTL.json``, and ``*_MTL.txt``. All three carry the same content under the
same group names, so they are parsed into one shape — a mapping of group name
to that group's key-value pairs — and read from there.

Reading is **group-scoped, never by key name alone**. A Level-2 metadata file
documents the Level-1 product it was derived from as well as itself, so the same
key appears twice with different values: ``PROCESSING_LEVEL`` is ``L2SP`` under
``PRODUCT_CONTENTS`` and ``L1TP`` under ``LEVEL1_PROCESSING_RECORD``, and
``REFLECTANCE_MULT_BAND_4`` is the surface-reflectance coefficient
(``2.75e-05``) under ``LEVEL2_SURFACE_REFLECTANCE_PARAMETERS`` but the
top-of-atmosphere one (``2.0000E-05``) under ``LEVEL1_RADIOMETRIC_RESCALING``.
Searching for a bare key name would pick whichever came first, which is a
property of the file's ordering rather than of its meaning.

Landsat has no "L1C" or "L2A" — that vocabulary belongs to Sentinel-2 and does
not transfer. Landsat levels are ``L1TP``/``L1GT``/``L1GS`` and ``L2SP``/
``L2SR``, and only the Level-2 pair is supported here.

Every field read here was checked against real USGS output rather than
documentation, using ``LC09_L2SP_193028_20260822_02_T1`` and
``LE07_L2SP_193028_20231024_02_T2`` in all three syntaxes. Three details worth
keeping in mind: ``SCENE_CENTER_TIME`` is written with seven fractional digits
(``10:04:22.1183580Z``), which is more precision than
:meth:`datetime.datetime.fromisoformat` accepts; ``WRS_ROW`` is zero-padded in
the ODL text (``028``) but not in the JSON (``28``); and Landsat 7 has no
band 6 in surface reflectance and no band 9 at all, so a per-band table read
from one sensor must never be assumed to fit another.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from xml.etree import ElementTree as ET

from eeo.core.exceptions import ValidationError
from eeo.core.types import StrPath
from eeo.io._archive import Located, ProductSource, nested_archive, open_product

#: Metadata syntaxes, in the order they are preferred when several are present.
_METADATA_SUFFIXES = ("_MTL.xml", "_MTL.json", "_MTL.txt")

#: The document's outermost group, which wraps every other group.
_ROOT_GROUP = "LANDSAT_METADATA_FILE"

#: Groups the fields below are read from.
_CONTENTS = "PRODUCT_CONTENTS"
_ATTRIBUTES = "IMAGE_ATTRIBUTES"
_REFLECTANCE = "LEVEL2_SURFACE_REFLECTANCE_PARAMETERS"
_TEMPERATURE = "LEVEL2_SURFACE_TEMPERATURE_PARAMETERS"

#: Product-id prefix to mission number and instrument. Landsat 8 and 9 share a
#: product layout and band numbering; Landsat 4, 5, and 7 share a different one.
_SENSORS = {
    "LT04": (4, "TM"),
    "LT05": (5, "TM"),
    "LE07": (7, "ETM+"),
    "LC08": (8, "OLI/TIRS"),
    "LC09": (9, "OLI-2/TIRS-2"),
}

#: Level-2 processing levels: a science product (reflectance and temperature)
#: and a reflectance-only product for scenes where temperature cannot be made.
_SUPPORTED_LEVELS = ("L2SP", "L2SR")

#: Level-1 processing levels, recognised only so they can be refused by name.
_LEVEL1_LEVELS = ("L1TP", "L1GT", "L1GS")

_MULT = re.compile(r"^REFLECTANCE_MULT_BAND_(\w+)$")
_TEMP_MULT = re.compile(r"^TEMPERATURE_MULT_BAND_(\w+)$")


@dataclass(frozen=True)
class LandsatProduct:
    """What a Landsat Collection 2 metadata file says about its product.

    Attributes
    ----------
    source : ProductSource
        Where the product's files are read from — a directory, a tar, or a zip.
    root : PurePosixPath
        The product's root within that source. Every file the metadata names
        sits directly under it.
    name : str
        The product's name, normally the directory's or the archive's.
    metadata : PurePosixPath
        The metadata file that was read, relative to :attr:`source`.
    product_id : str
        Full ``LANDSAT_PRODUCT_ID``, e.g.
        ``"LC09_L2SP_193028_20260822_20260823_02_T1"``.
    level : str
        Processing level, ``"L2SP"`` or ``"L2SR"``.
    mission : int
        Landsat mission number, e.g. ``9``.
    instrument : str
        Instrument name, e.g. ``"OLI-2/TIRS-2"``.
    spacecraft : str
        The file's own ``SPACECRAFT_ID``, e.g. ``"LANDSAT_9"``.
    wrs_path, wrs_row : str
        WRS-2 path and row, zero-padded to three digits so they read the same
        whichever syntax the metadata was written in.
    acquired : datetime.datetime
        Scene centre acquisition time, timezone-aware in UTC.
    collection_number, collection_category : str
        Collection number (``"02"``) and tier (``"T1"``, ``"T2"``, ``"RT"``).
    reflectance_scaling : dict of str to tuple of float
        ``(multiplier, additive)`` per surface-reflectance band, keyed by the
        band's file token (``"SR_B4"``). Converts a digital number to
        reflectance.
    temperature_scaling : dict of str to tuple of float
        ``(multiplier, additive)`` per surface-temperature band, keyed by file
        token (``"ST_B10"``). Converts a digital number to kelvin.
    band_files : dict of str to str
        Image filenames the metadata lists, keyed by file token
        (``"SR_B4"``, ``"ST_B10"``, ``"QA_PIXEL"``).
    groups : dict of str to dict
        The whole parsed document, group by group, for anything not lifted out
        above.
    """

    source: ProductSource
    root: PurePosixPath
    name: str
    metadata: PurePosixPath
    product_id: str
    level: str
    mission: int
    instrument: str
    spacecraft: str
    wrs_path: str
    wrs_row: str
    acquired: dt.datetime
    collection_number: str
    collection_category: str
    reflectance_scaling: dict[str, tuple[float, float]]
    temperature_scaling: dict[str, tuple[float, float]]
    band_files: dict[str, str]
    groups: dict[str, dict[str, str]]


def find_metadata(source: StrPath | ProductSource) -> Located:
    """Locate a Landsat product's metadata file within a source.

    Parameters
    ----------
    source : str or pathlib.Path or ProductSource
        A product directory, a directory or archive holding one, a metadata
        file directly, or an already-opened source.

    Returns
    -------
    Located
        The source the metadata was found in, and its path **relative to that
        source**, preferring ``_MTL.xml`` over ``_MTL.json`` over ``_MTL.txt``
        when more than one is present. The path's parent is the product root.
        The source is not always the one passed in: pointing at a folder that
        holds a ``.tar`` resolves to the tar.

    Raises
    ------
    ValidationError
        If the path does not exist or holds no metadata file.

    Examples
    --------
    >>> find_metadata("LC09_L2SP_193028_20260822_02_T1").path  # doctest: +SKIP
    PurePosixPath('LC09_L2SP_193028_20260822_20260823_02_T1_MTL.xml')
    """
    opened = source if isinstance(source, ProductSource) else open_product(source, "Landsat")
    if opened.named_entry is not None:
        return Located(opened, opened.named_entry)

    # A USGS tar holds its files at the root; a directory the product was
    # unpacked into holds them one level down. Either way there must be exactly
    # one — a download folder holding several scenes has no defensible answer,
    # and picking the alphabetically first would silently analyse the wrong one.
    for depth in ("", "*/"):
        for suffix in _METADATA_SUFFIXES:
            matches = opened.glob(f"{depth}*{suffix}")
            if len(matches) > 1:
                raise ValidationError(
                    f"{str(opened.location)!r} holds {len(matches)} Landsat products, so "
                    f"which one to load cannot be told from the path alone: "
                    f"{', '.join(sorted(found.name for found in matches))}. "
                    f"Name the one you mean."
                )
            if matches:
                return Located(opened, matches[0])

    # The folder may hold the product still tarred, which is how it arrives.
    archive = nested_archive(opened, "Landsat")
    if archive is not None:
        return find_metadata(archive)

    raise ValidationError(
        f"{str(opened.location)!r} holds no Landsat metadata file; expected one ending in "
        f"{', '.join(_METADATA_SUFFIXES)}, or a .tar holding one"
    )


def _parse_odl(text: str) -> dict[str, dict[str, str]]:
    """Parse the ODL text syntax into groups of key-value pairs."""
    groups: dict[str, dict[str, str]] = {}
    stack: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line == "END":
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key == "GROUP":
            stack.append(value)
            groups.setdefault(value, {})
        elif key == "END_GROUP":
            if stack:
                stack.pop()
        elif stack:
            groups[stack[-1]][key] = value.strip('"')
    groups.pop(_ROOT_GROUP, None)
    return groups


def _parse_json(text: str) -> dict[str, dict[str, str]]:
    """Parse the JSON syntax into groups of key-value pairs."""
    document = json.loads(text)
    root = document.get(_ROOT_GROUP, document)
    return {
        group: {key: str(value) for key, value in members.items()}
        for group, members in root.items()
        if isinstance(members, Mapping)
    }


def _parse_xml(text: str) -> dict[str, dict[str, str]]:
    """Parse the XML syntax into groups of key-value pairs."""
    root = ET.fromstring(text)
    return {
        group.tag: {member.tag: (member.text or "").strip() for member in group if len(member) == 0}
        for group in root
    }


def read_metadata_groups(
    source: ProductSource, metadata: PurePosixPath
) -> dict[str, dict[str, str]]:
    """Parse a Landsat metadata file of any syntax into groups.

    Parameters
    ----------
    source : ProductSource
        Where to read the file from.
    metadata : PurePosixPath
        An ``_MTL.xml``, ``_MTL.json``, or ``_MTL.txt`` file, relative to
        ``source``. The syntax is chosen by suffix, since all three carry
        identical content.

    Returns
    -------
    dict
        Group name to that group's key-value pairs, values as strings. The
        outermost ``LANDSAT_METADATA_FILE`` wrapper is unwrapped.

    Raises
    ------
    ValidationError
        If the file cannot be parsed in its syntax.
    """
    text = source.read_bytes(metadata).decode(errors="replace")
    parsers = {".xml": _parse_xml, ".json": _parse_json}
    parser = parsers.get(metadata.suffix.lower(), _parse_odl)
    try:
        return parser(text)
    except (ET.ParseError, json.JSONDecodeError, ValueError) as err:
        raise ValidationError(f"{metadata.name} is not readable: {err}") from err


def _require(groups: Mapping[str, Mapping[str, str]], group: str, key: str) -> str:
    """Read one key from one group, failing with both names when it is absent."""
    value = groups.get(group, {}).get(key)
    if value is None or not value.strip():
        raise ValidationError(
            f"the Landsat metadata states no {key} in {group}, so the product cannot be identified"
        )
    return value.strip()


def _scaling(group: Mapping[str, str], pattern: re.Pattern[str], prefix: str) -> dict:
    """Pair each MULT coefficient in a group with its matching ADD coefficient."""
    found: dict[str, tuple[float, float]] = {}
    for key, value in group.items():
        match = pattern.match(key)
        if match is None:
            continue
        band = match.group(1)
        add = group.get(key.replace("_MULT_", "_ADD_"))
        if add is None:
            continue
        try:
            coefficients = (float(value), float(add))
        except ValueError as err:
            raise ValidationError(
                f"scaling coefficients for band {band} are not numbers; got {value!r} and {add!r}"
            ) from err
        # Reflectance keys name the band by number alone; temperature keys
        # already carry the ST_ token.
        found[band if band.startswith(prefix) else f"{prefix}{band}"] = coefficients
    return found


def _acquired(groups: Mapping[str, Mapping[str, str]]) -> dt.datetime:
    """Combine the acquisition date and scene centre time into one instant."""
    date = _require(groups, _ATTRIBUTES, "DATE_ACQUIRED")
    time = _require(groups, _ATTRIBUTES, "SCENE_CENTER_TIME").rstrip("Zz")
    # USGS writes seven fractional digits; fromisoformat accepts three or six,
    # so the value is truncated to microseconds rather than rejected.
    if "." in time:
        whole, _, fraction = time.partition(".")
        time = f"{whole}.{fraction[:6].ljust(6, '0')}"
    try:
        parsed = dt.datetime.fromisoformat(f"{date}T{time}")
    except ValueError as err:
        raise ValidationError(
            f"could not read the acquisition time; got DATE_ACQUIRED={date!r} "
            f"and SCENE_CENTER_TIME={time!r}"
        ) from err
    return parsed.replace(tzinfo=dt.timezone.utc)


def read_product(path: StrPath | ProductSource, *, level: str | None = None) -> LandsatProduct:
    """Identify a Landsat Collection 2 product and read its metadata.

    Parameters
    ----------
    path : str or pathlib.Path or ProductSource
        A product directory, a directory or archive holding one, a metadata
        file, or an already-opened source.
    level : str or None, default None
        Assert the processing level rather than detecting it. Exists for a
        product whose metadata is damaged; when the metadata states a level and
        this contradicts it, the metadata wins and this raises.

    Returns
    -------
    LandsatProduct
        The product's identity, geometry, scaling coefficients, and file list.

    Raises
    ------
    ValidationError
        If no metadata file is found, it cannot be parsed, the product is a
        Level-1 product, its mission is not recognised, or ``level``
        contradicts the metadata.

    Notes
    -----
    Level-1 products are refused. They are scaled radiance rather than surface
    reflectance, with different coefficients and a different band list.

    Examples
    --------
    >>> product = read_product("LC09_L2SP_193028_20260822_02_T1")  # doctest: +SKIP
    >>> product.mission, product.level, product.wrs_row  # doctest: +SKIP
    (9, 'L2SP', '028')
    """
    source, metadata = find_metadata(
        path if isinstance(path, ProductSource) else open_product(path, "Landsat")
    )
    groups = read_metadata_groups(source, metadata)

    detected = groups.get(_CONTENTS, {}).get("PROCESSING_LEVEL", "").strip().upper()
    if level is not None:
        wanted = level.strip().upper()
        if detected and detected != wanted:
            raise ValidationError(
                f"level={level!r} contradicts the product, which reports {detected} "
                f"in {metadata.name}. Remove the override to use the product's own level."
            )
        detected = wanted
    if not detected:
        raise ValidationError(
            f"{metadata.name} states no PROCESSING_LEVEL in {_CONTENTS}, so the "
            f"product cannot be identified"
        )
    if detected in _LEVEL1_LEVELS:
        raise ValidationError(
            f"Level-1 is not supported; {metadata.name} reports a {detected} product. "
            f"Easy-EO reads Collection 2 Level-2 surface reflectance, so download the "
            f"L2SP or L2SR product for this scene."
        )
    if detected not in _SUPPORTED_LEVELS:
        raise ValidationError(
            f"unrecognised Landsat processing level {detected!r} in {metadata.name}; "
            f"expected one of {', '.join(_SUPPORTED_LEVELS)}"
        )

    product_id = _require(groups, _CONTENTS, "LANDSAT_PRODUCT_ID")
    prefix = product_id[:4].upper()
    if prefix not in _SENSORS:
        raise ValidationError(
            f"unrecognised Landsat mission {prefix!r} in product id {product_id!r}; "
            f"supported missions are {', '.join(sorted(_SENSORS))}"
        )
    mission, instrument = _SENSORS[prefix]

    band_files = {}
    for key, value in groups.get(_CONTENTS, {}).items():
        if key.startswith("FILE_NAME_") and value.upper().endswith(".TIF"):
            token = PurePosixPath(value).stem
            band_files[token.removeprefix(f"{product_id}_")] = value

    root = metadata.parent
    return LandsatProduct(
        source=source,
        root=root,
        name=root.name or source.name,
        metadata=metadata,
        product_id=product_id,
        level=detected,
        mission=mission,
        instrument=instrument,
        spacecraft=groups.get(_ATTRIBUTES, {}).get("SPACECRAFT_ID", "").strip(),
        wrs_path=_require(groups, _ATTRIBUTES, "WRS_PATH").zfill(3),
        wrs_row=_require(groups, _ATTRIBUTES, "WRS_ROW").zfill(3),
        acquired=_acquired(groups),
        collection_number=groups.get(_CONTENTS, {}).get("COLLECTION_NUMBER", "").strip(),
        collection_category=groups.get(_CONTENTS, {}).get("COLLECTION_CATEGORY", "").strip(),
        reflectance_scaling=_scaling(groups.get(_REFLECTANCE, {}), _MULT, "SR_B"),
        temperature_scaling=_scaling(groups.get(_TEMPERATURE, {}), _TEMP_MULT, "ST_"),
        band_files=band_files,
        groups=groups,
    )
