"""Identifying and reading the metadata of a Sentinel-2 product.

A downloaded Sentinel-2 product is a ``.SAFE`` directory whose root holds one
product manifest — ``MTD_MSIL2A.xml`` for Level-2A, ``MTD_MSIL1C.xml`` for
Level-1C — and whose ``GRANULE/<granule id>/`` subdirectory holds a tile
manifest, ``MTD_TL.xml``, carrying the projection. Between them they say what
the product is, when it was acquired, and how its bands are laid out, without
opening a single image file.

Processing level is **detected, never assumed**. The manifest's own filename
distinguishes the two levels, but a renamed file must not be able to mislead the
reader, so the authority is ``<PRODUCT_TYPE>`` inside the document and the
filename is only a fallback. Level-1C is identified precisely so it can be
refused with an explanation, rather than failing later on a directory layout
that was never going to be there.

Namespaces are matched by local name throughout. The manifest's root element
sits in a namespace whose URI carries the Product Specification Document
version (``.../PSD/User_Product_Level-2A.xsd`` under a versioned host), so
pinning a URI would break on the next PSD revision, while its children sit in no
namespace at all. Comparing local names is stable across both.

Every element and attribute read here was checked against real ESA output
rather than taken from documentation, using three products: an L2A scene at
baseline 05.11 (``S2B_MSIL2A_20240929T100719_R022_T32TPS``), an L2A scene at
baseline 02.12 (``S2B_MSIL2A_20200930T100729_R022_T32TPS``), and the matching
L1C product (``S2B_MSIL1C_20240929T100719_N0511_R022_T32TPS``). Three details
that would otherwise be easy to get wrong were confirmed there: the offset list
keys bands by ``band_id`` while the spectral list keys them by ``bandId``;
``physicalBand`` spells single-digit bands ``B1`` where the image files spell
them ``B01``; and a pre-04.00 product carries no ``BOA_ADD_OFFSET_VALUES_LIST``
element at all, rather than one holding zeros. Note also that L1C names its
equivalents ``QUANTIFICATION_VALUE`` and ``RADIO_ADD_OFFSET`` — not the ``BOA_``
spellings — which does not arise here only because L1C is refused first.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import PurePosixPath
from xml.etree import ElementTree as ET

from eeo.core.exceptions import ValidationError
from eeo.core.types import StrPath
from eeo.io._archive import ProductSource, open_product

#: Product manifests, newest level first, as found at a ``.SAFE`` root.
_MANIFESTS = ("MTD_MSIL2A.xml", "MTD_MSIL1C.xml")

#: The granule's own manifest, which carries the projection.
_TILE_MANIFEST = "MTD_TL.xml"

#: ``<PRODUCT_TYPE>`` values mapped to the short level names used in messages.
_PRODUCT_TYPES = {"S2MSI2A": "L2A", "S2MSI1C": "L1C"}

#: Levels this reader supports. Level-1C is recognised only to refuse it.
_SUPPORTED_LEVELS = ("L2A",)

# Spectral_Information spells the single-digit bands "B1".."B9" while the image
# files spell them "B01".."B09"; "B8A" is spelled the same in both.
_SINGLE_DIGIT_BAND = re.compile(r"^B(\d)$")


@dataclass(frozen=True)
class Sentinel2Product:
    """What a Sentinel-2 product's manifests say about it.

    Attributes
    ----------
    source : ProductSource
        Where the product's files are read from — a directory, a zip, or a tar.
    root : PurePosixPath
        The ``.SAFE`` root within that source. Every image path the manifest
        lists is written relative to it.
    name : str
        The product's name, normally the ``.SAFE`` directory's.
    manifest : PurePosixPath
        The product manifest that was read, relative to :attr:`source`.
    level : str
        Processing level, ``"L2A"`` or ``"L1C"``.
    product_type : str
        The manifest's own ``<PRODUCT_TYPE>``, e.g. ``"S2MSI2A"``.
    tile_id : str
        MGRS tile, e.g. ``"T32TPS"``.
    sensing_time : datetime.datetime
        Acquisition time, timezone-aware in UTC.
    processing_baseline : str
        Baseline the product was processed with, e.g. ``"05.11"``. The band
        encoding changed at ``"04.00"``.
    crs : str
        The tile's projection as an authority code, e.g. ``"EPSG:32632"``.
    quantification_value : float or None
        Digital numbers per unit reflectance, normally 10000.
    band_offsets : dict of str to float
        Per-band additive offset in digital numbers, keyed by the band's image
        file spelling (``"B04"``, ``"B8A"``). Empty for products predating
        baseline 04.00, which carry no offset element because they have none.
    image_files : tuple of str
        Image paths the manifest lists, relative to :attr:`root` and without
        their file extension, in manifest order.
    granule : PurePosixPath or None
        The granule directory whose tile manifest was read.
    """

    source: ProductSource
    root: PurePosixPath
    name: str
    manifest: PurePosixPath
    level: str
    product_type: str
    tile_id: str
    sensing_time: dt.datetime
    processing_baseline: str
    crs: str
    quantification_value: float | None
    band_offsets: dict[str, float]
    image_files: tuple[str, ...]
    granule: PurePosixPath | None


def _local(tag: str) -> str:
    """Strip any namespace from an element tag, leaving its local name."""
    return tag.rpartition("}")[2]


def _iter_named(root: ET.Element, name: str) -> Iterator[ET.Element]:
    """Yield every descendant whose local name matches, root included."""
    for element in root.iter():
        if _local(element.tag) == name:
            yield element


def _text(root: ET.Element, name: str) -> str | None:
    """Return the stripped text of the first element with this local name."""
    for element in _iter_named(root, name):
        if element.text is not None and element.text.strip():
            return element.text.strip()
    return None


def _parse_time(value: str, *, source: str) -> dt.datetime:
    """Parse a Sentinel-2 timestamp into a timezone-aware UTC datetime.

    Parameters
    ----------
    value : str
        An ISO 8601 instant as the manifests write it, ending in ``Z``.
    source : str
        Element name the value came from, for the error message.

    Returns
    -------
    datetime.datetime
        The instant, in UTC.

    Raises
    ------
    ValidationError
        If the value is not a parseable instant.
    """
    # Python 3.10's fromisoformat rejects a trailing "Z"; the package supports
    # 3.10, so normalise it rather than relying on 3.11+ behaviour.
    text = value.strip()
    normalised = f"{text[:-1]}+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = dt.datetime.fromisoformat(normalised)
    except ValueError as err:
        raise ValidationError(f"could not read {source} as a date and time; got {value!r}") from err
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


def find_manifest(source: StrPath | ProductSource) -> PurePosixPath:
    """Locate a Sentinel-2 product manifest within a source.

    Parameters
    ----------
    source : str or pathlib.Path or ProductSource
        A ``.SAFE`` directory, a directory or archive holding one, a manifest
        file directly, or an already-opened source.

    Returns
    -------
    PurePosixPath
        The manifest's path **relative to the source**. Its parent is the
        ``.SAFE`` root, which every image path in the manifest is written
        relative to.

    Raises
    ------
    ValidationError
        If the path does not exist, or holds no recognisable manifest.

    Examples
    --------
    >>> find_manifest("S2B_MSIL2A_20240830T100559.SAFE")  # doctest: +SKIP
    PurePosixPath('MTD_MSIL2A.xml')
    """
    opened = source if isinstance(source, ProductSource) else open_product(source, "Sentinel-2")
    if opened.named_entry is not None:
        return opened.named_entry

    for name in _MANIFESTS:
        if opened.exists(name):
            return PurePosixPath(name)

    # A user may point at the folder, or the zip, the .SAFE sits inside. That
    # is only unambiguous while there is one: a download folder holding several
    # scenes has no defensible answer, and picking the alphabetically first
    # would silently analyse the wrong date.
    for name in _MANIFESTS:
        nested = opened.glob(f"*.SAFE/{name}")
        if len(nested) > 1:
            raise ValidationError(
                f"{str(opened.location)!r} holds {len(nested)} Sentinel-2 products, so "
                f"which one to load cannot be told from the path alone: "
                f"{', '.join(sorted(found.parent.name for found in nested))}. "
                f"Name the one you mean."
            )
        if nested:
            return nested[0]

    raise ValidationError(
        f"{str(opened.location)!r} holds no Sentinel-2 product manifest; expected one of "
        f"{', '.join(_MANIFESTS)} at the root of a .SAFE directory"
    )


def _detect_level(
    root: ET.Element, manifest: PurePosixPath, override: str | None
) -> tuple[str, str]:
    """Determine the processing level, preferring the document over the filename.

    Returns
    -------
    tuple of str
        ``(level, product_type)``.

    Raises
    ------
    ValidationError
        If the level cannot be determined at all, or if ``override``
        contradicts what the document states.
    """
    product_type = _text(root, "PRODUCT_TYPE")
    detected = _PRODUCT_TYPES.get(product_type or "")

    if detected is None:
        # An unreadable or absent PRODUCT_TYPE is the only case where the
        # filename, or a caller's override, gets to decide.
        for name, level in (("MTD_MSIL2A.xml", "L2A"), ("MTD_MSIL1C.xml", "L1C")):
            if manifest.name == name:
                detected = level
                break

    if override is not None:
        wanted = override.strip().upper()
        if wanted not in _PRODUCT_TYPES.values():
            raise ValidationError(
                f"level must be one of {', '.join(sorted(_PRODUCT_TYPES.values()))}; "
                f"got {override!r}"
            )
        if detected is not None and detected != wanted:
            raise ValidationError(
                f"level={override!r} contradicts the product, which reports "
                f"{detected} ({product_type or 'no PRODUCT_TYPE'}) in "
                f"{manifest.name}. Remove the override to use the product's own level."
            )
        detected = wanted

    if detected is None:
        raise ValidationError(
            f"{manifest.name} states no PRODUCT_TYPE and its name identifies no "
            f"processing level, so the product cannot be identified"
        )
    return detected, product_type or ""


def _band_offsets(root: ET.Element) -> dict[str, float]:
    """Map each band's image-file name to its additive offset in digital numbers.

    The offsets are keyed by numeric ``band_id`` in the manifest, and the
    ``band_id``-to-band mapping lives in a separate spectral information list,
    so the two are joined here. A product from before baseline 04.00 carries no
    offset element and yields an empty mapping — it has no offset, rather than
    an unknown one.
    """
    by_id: dict[str, str] = {}
    for element in _iter_named(root, "Spectral_Information"):
        band_id = element.get("bandId")
        physical = element.get("physicalBand")
        if band_id is not None and physical is not None:
            by_id[band_id] = _SINGLE_DIGIT_BAND.sub(r"B0\1", physical.strip())

    offsets: dict[str, float] = {}
    for element in _iter_named(root, "BOA_ADD_OFFSET"):
        band_id = element.get("band_id")
        if band_id is None or element.text is None:
            continue
        name = by_id.get(band_id)
        if name is None:
            continue
        try:
            offsets[name] = float(element.text.strip())
        except ValueError as err:
            raise ValidationError(
                f"BOA_ADD_OFFSET for band {name} is not a number; got {element.text!r}"
            ) from err
    return offsets


def _granule(
    source: ProductSource, root: PurePosixPath, image_files: tuple[str, ...]
) -> PurePosixPath | None:
    """Find the granule whose tile manifest can be read.

    Prefers the granule the manifest's own image paths name, and falls back to
    searching the layout for a product whose manifest lists none.
    """
    for relative in image_files:
        parts = PurePosixPath(relative).parts
        if len(parts) >= 2 and parts[0] == "GRANULE":
            granule = root / parts[0] / parts[1]
            if source.exists(granule / _TILE_MANIFEST):
                return granule

    found = source.glob(str(root / "GRANULE" / "*" / _TILE_MANIFEST))
    return found[0].parent if found else None


def read_product(path: StrPath | ProductSource, *, level: str | None = None) -> Sentinel2Product:
    """Identify a Sentinel-2 product and read its metadata.

    Reads the product manifest and, where present, the granule's tile manifest
    for the projection. No image data is opened.

    Parameters
    ----------
    path : str or pathlib.Path or ProductSource
        A ``.SAFE`` directory, a directory or archive holding one, a manifest
        file, or an already-opened source.
    level : str or None, default None
        Assert the processing level rather than detecting it. Exists for a
        product whose manifest is damaged; when the manifest states a level and
        this contradicts it, the manifest wins and this raises.

    Returns
    -------
    Sentinel2Product
        The product's identity and metadata.

    Raises
    ------
    ValidationError
        If no manifest is found, the manifest is not readable XML, the
        processing level cannot be determined, ``level`` contradicts the
        manifest, or the product is a level this reader does not support.

    Notes
    -----
    Level-1C is recognised and refused. It is top-of-atmosphere radiance with no
    atmospheric correction, and its band layout differs from Level-2A's
    resolution subdirectories.

    Examples
    --------
    >>> product = read_product("S2B_MSIL2A_20240830T100559.SAFE")  # doctest: +SKIP
    >>> product.level, product.tile_id, product.processing_baseline  # doctest: +SKIP
    ('L2A', 'T32TPS', '05.11')
    """
    source = path if isinstance(path, ProductSource) else open_product(path, "Sentinel-2")
    manifest = find_manifest(source)
    try:
        root = ET.fromstring(source.read_bytes(manifest))
    except ET.ParseError as err:
        raise ValidationError(f"{manifest.name} is not readable XML: {err}") from err

    detected, product_type = _detect_level(root, manifest, level)
    if detected not in _SUPPORTED_LEVELS:
        raise ValidationError(
            f"Level-1C is not supported; {manifest.name} reports a {detected} product. "
            f"Easy-EO reads Level-2A surface reflectance, so download the L2A product "
            f"for this scene."
        )

    safe = manifest.parent
    image_files = tuple(
        element.text.strip() for element in _iter_named(root, "IMAGE_FILE") if element.text
    )
    granule = _granule(source, safe, image_files)

    quantification = _text(root, "BOA_QUANTIFICATION_VALUE")
    if quantification is not None:
        try:
            quantification_value: float | None = float(quantification)
        except ValueError as err:
            raise ValidationError(
                f"BOA_QUANTIFICATION_VALUE is not a number; got {quantification!r}"
            ) from err
    else:
        quantification_value = None

    # The tile manifest carries the projection and the tile's own sensing time;
    # the product manifest carries the datatake's start. Prefer the tile's.
    crs = None
    tile_time = None
    if granule is not None:
        tile_manifest = granule / _TILE_MANIFEST
        try:
            tile_root = ET.fromstring(source.read_bytes(tile_manifest))
        except ET.ParseError as err:
            raise ValidationError(f"{tile_manifest.name} is not readable XML: {err}") from err
        crs = _text(tile_root, "HORIZONTAL_CS_CODE")
        tile_time = _text(tile_root, "SENSING_TIME")

    if crs is None:
        raise ValidationError(
            f"the product states no projection; expected HORIZONTAL_CS_CODE in a "
            f"granule's {_TILE_MANIFEST} under {source.location}"
        )

    when = tile_time or _text(root, "PRODUCT_START_TIME")
    if when is None:
        raise ValidationError(
            f"{manifest.name} states no acquisition time; expected SENSING_TIME or "
            f"PRODUCT_START_TIME"
        )

    name = safe.name or source.name
    uri = _text(root, "PRODUCT_URI") or name
    tile_match = re.search(r"_(T\d{2}[A-Z]{3})_", uri)

    return Sentinel2Product(
        source=source,
        root=safe,
        name=name,
        manifest=manifest,
        level=detected,
        product_type=product_type,
        tile_id=tile_match.group(1) if tile_match else "",
        sensing_time=_parse_time(when, source="the acquisition time"),
        processing_baseline=_text(root, "PROCESSING_BASELINE") or "",
        crs=crs,
        quantification_value=quantification_value,
        band_offsets=_band_offsets(root),
        image_files=image_files,
        granule=granule,
    )
