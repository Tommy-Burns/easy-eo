"""Per-sensor band tables, and resolving a band name to a file on disk.

Every mission numbers its bands its own way, and the numbering is not
transferable: **Landsat 7's red is band 3 where Landsat 8 and 9's red is
band 4**, so ``SR_B4`` is red on one sensor and near-infrared on the other.
Asking for a band by number is therefore a question that cannot be answered
without knowing the mission, and a script that moves between missions silently
computes an index over the wrong wavelengths. Common names — ``"red"``,
``"nir"``, ``"swir16"`` — are the primary spelling here for that reason; the
native ids are accepted as aliases within a sensor's own table.

The common names follow the STAC Electro-Optical extension, which is what the
catalogues use and therefore what :mod:`eeo.io.stac` already returns, so a band
called ``"red"`` means the same thing whether a scene was loaded from a
catalogue or from a folder.

Notes
-----
Every entry below was derived from real scenes rather than from documentation:
the Landsat mapping by reading which ``SR_B*`` file each common-name asset of a
real Landsat 5, 7, 8 and 9 product resolves to, and the Sentinel-2 mapping from
the ``eo:bands`` of the live ``sentinel-2-l2a`` collections
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from eeo.core.exceptions import ValidationError

# Sentinel-2 spells single-digit bands with a leading zero ("B04"), so a caller
# who types "B4" is asking for the same band.
_SHORT_S2_BAND = re.compile(r"^B(\d)$", re.IGNORECASE)


@dataclass(frozen=True)
class BandInfo:
    """One band of one sensor.

    Attributes
    ----------
    common_name : str
        Cross-sensor name, e.g. ``"red"``. The primary spelling.
    band_id : str
        The mission's own identifier and the token its files carry, e.g.
        ``"B04"`` for Sentinel-2 or ``"SR_B4"`` for Landsat.
    wavelength : float or None
        Centre wavelength in micrometres, or None for a band that measures no
        single wavelength, such as a quality mask.
    resolution : int
        Pixel size in metres as the band is *delivered*, which is not always
        the instrument's own resolution — Landsat's thermal band is sensed at
        100 m and delivered resampled to 30 m.
    kind : str
        ``"reflectance"``, ``"temperature"``, ``"quality"``, or ``"ancillary"``.
        Quality bands hold class numbers or packed bits, so they must never be
        resampled by anything but nearest neighbour.
    """

    common_name: str
    band_id: str
    wavelength: float | None
    resolution: int
    kind: str


def _table(entries: tuple[BandInfo, ...]) -> dict[str, BandInfo]:
    """Key a band table by common name."""
    return {band.common_name: band for band in entries}


#: Sentinel-2 Level-2A. No ``B10``: the cirrus band is not written to L2A.
SENTINEL2 = _table(
    (
        BandInfo("coastal", "B01", 0.443, 60, "reflectance"),
        BandInfo("blue", "B02", 0.490, 10, "reflectance"),
        BandInfo("green", "B03", 0.560, 10, "reflectance"),
        BandInfo("red", "B04", 0.665, 10, "reflectance"),
        BandInfo("rededge1", "B05", 0.704, 20, "reflectance"),
        BandInfo("rededge2", "B06", 0.740, 20, "reflectance"),
        BandInfo("rededge3", "B07", 0.783, 20, "reflectance"),
        BandInfo("nir", "B08", 0.842, 10, "reflectance"),
        BandInfo("nir08", "B8A", 0.865, 20, "reflectance"),
        BandInfo("nir09", "B09", 0.945, 60, "reflectance"),
        BandInfo("swir16", "B11", 1.610, 20, "reflectance"),
        BandInfo("swir22", "B12", 2.190, 20, "reflectance"),
        BandInfo("scl", "SCL", None, 20, "quality"),
        BandInfo("aot", "AOT", None, 10, "ancillary"),
        BandInfo("wvp", "WVP", None, 10, "ancillary"),
        BandInfo("tci", "TCI", None, 10, "ancillary"),
    )
)

#: Landsat 8 and 9 (OLI/TIRS and OLI-2/TIRS-2), which share a band numbering.
LANDSAT_OLI = _table(
    (
        BandInfo("coastal", "SR_B1", 0.44, 30, "reflectance"),
        BandInfo("blue", "SR_B2", 0.48, 30, "reflectance"),
        BandInfo("green", "SR_B3", 0.56, 30, "reflectance"),
        BandInfo("red", "SR_B4", 0.65, 30, "reflectance"),
        BandInfo("nir08", "SR_B5", 0.87, 30, "reflectance"),
        BandInfo("swir16", "SR_B6", 1.61, 30, "reflectance"),
        BandInfo("swir22", "SR_B7", 2.20, 30, "reflectance"),
        BandInfo("lwir11", "ST_B10", 10.90, 30, "temperature"),
        BandInfo("qa_pixel", "QA_PIXEL", None, 30, "quality"),
        BandInfo("qa_radsat", "QA_RADSAT", None, 30, "quality"),
        BandInfo("qa_aerosol", "SR_QA_AEROSOL", None, 30, "quality"),
    )
)

#: Landsat 4, 5 and 7 (TM and ETM+). Numbering differs from OLI: red is band 3
#: here and band 4 there, and there is no coastal band at all.
LANDSAT_TM = _table(
    (
        BandInfo("blue", "SR_B1", 0.48, 30, "reflectance"),
        BandInfo("green", "SR_B2", 0.56, 30, "reflectance"),
        BandInfo("red", "SR_B3", 0.66, 30, "reflectance"),
        BandInfo("nir08", "SR_B4", 0.84, 30, "reflectance"),
        BandInfo("swir16", "SR_B5", 1.65, 30, "reflectance"),
        BandInfo("swir22", "SR_B7", 2.20, 30, "reflectance"),
        BandInfo("lwir", "ST_B6", 11.45, 30, "temperature"),
        BandInfo("qa_pixel", "QA_PIXEL", None, 30, "quality"),
        BandInfo("qa_radsat", "QA_RADSAT", None, 30, "quality"),
        BandInfo("atmos_opacity", "SR_ATMOS_OPACITY", None, 30, "ancillary"),
        BandInfo("cloud_qa", "SR_CLOUD_QA", None, 30, "quality"),
    )
)

#: Landsat mission number to its band table.
_LANDSAT_TABLES = {4: LANDSAT_TM, 5: LANDSAT_TM, 7: LANDSAT_TM, 8: LANDSAT_OLI, 9: LANDSAT_OLI}


def landsat_bands(mission: int) -> dict[str, BandInfo]:
    """Return the band table for one Landsat mission.

    Parameters
    ----------
    mission : int
        Landsat mission number, e.g. ``9``.

    Returns
    -------
    dict of str to BandInfo
        The mission's bands, keyed by common name.

    Raises
    ------
    ValidationError
        If the mission has no Collection 2 Level-2 band table.

    Examples
    --------
    >>> landsat_bands(9)["red"].band_id
    'SR_B4'
    >>> landsat_bands(7)["red"].band_id
    'SR_B3'
    """
    try:
        return _LANDSAT_TABLES[mission]
    except KeyError:
        raise ValidationError(
            f"no band table for Landsat {mission}; supported missions are "
            f"{', '.join(str(m) for m in sorted(_LANDSAT_TABLES))}"
        ) from None


def _aliases(bands: Mapping[str, BandInfo]) -> dict[str, BandInfo]:
    """Build every accepted spelling for a table, lowercased."""
    lookup: dict[str, BandInfo] = {}
    for band in bands.values():
        lookup[band.common_name.lower()] = band
        lookup[band.band_id.lower()] = band
        # "SR_B4" is also reachable as "B4"; "B04" is also reachable as "B4".
        _, _, bare = band.band_id.partition("_")
        if bare:
            lookup.setdefault(bare.lower(), band)
        short = _SHORT_S2_BAND.match(band.band_id)
        if short is None and band.band_id.lower().startswith("b0"):
            lookup.setdefault(f"b{band.band_id[2:]}".lower(), band)
    return lookup


def resolve_band(spec: str, bands: Mapping[str, BandInfo]) -> BandInfo:
    """Resolve one band name or native id against a sensor's table.

    Matching is case-insensitive and ignores surrounding whitespace, but is
    otherwise exact: no fuzzy matching and no guessing between sensors.

    Parameters
    ----------
    spec : str
        A common name (``"red"``), a native id (``"B04"``, ``"SR_B4"``), or a
        native id without its product prefix (``"B4"``).
    bands : Mapping of str to BandInfo
        The sensor's table, from :data:`SENTINEL2` or :func:`landsat_bands`.

    Returns
    -------
    BandInfo
        The matching band.

    Raises
    ------
    ValidationError
        If ``spec`` is not a string, or names no band in this table. The
        message lists the names the sensor does have.

    Examples
    --------
    >>> resolve_band("red", SENTINEL2).band_id
    'B04'
    >>> resolve_band(" NIR ", SENTINEL2).common_name
    'nir'
    >>> resolve_band("SR_B3", landsat_bands(7)).common_name
    'red'
    """
    if not isinstance(spec, str):
        raise ValidationError(
            f"a band must be named by a string, such as 'red' or 'B04'; got {spec!r}"
        )
    band = _aliases(bands).get(spec.strip().lower())
    if band is None:
        raise ValidationError(
            f"no band {spec!r} on this sensor; available bands are "
            f"{', '.join(sorted(bands))} (or their ids: "
            f"{', '.join(b.band_id for b in bands.values())})"
        )
    return band


def resolve_bands(specs: Sequence[str], bands: Mapping[str, BandInfo]) -> list[BandInfo]:
    """Resolve several band names, preserving order and rejecting duplicates.

    Parameters
    ----------
    specs : sequence of str
        Band names or ids, in the order they should become bands.
    bands : Mapping of str to BandInfo
        The sensor's table.

    Returns
    -------
    list of BandInfo
        The matching bands, in the order given.

    Raises
    ------
    ValidationError
        If ``specs`` is empty, names an unknown band, or names one band twice
        — including twice under different spellings, since ``"red"`` and
        ``"B04"`` would otherwise produce two identical bands.

    Examples
    --------
    >>> [b.band_id for b in resolve_bands(["red", "nir"], SENTINEL2)]
    ['B04', 'B08']
    """
    if isinstance(specs, str):
        raise ValidationError(
            f"expected a sequence of band names, such as ['red', 'nir']; got {specs!r}"
        )
    resolved = [resolve_band(spec, bands) for spec in specs]
    if not resolved:
        raise ValidationError("name at least one band to load; got an empty sequence")

    seen: dict[str, str] = {}
    for spec, band in zip(specs, resolved, strict=True):
        if band.common_name in seen:
            raise ValidationError(
                f"band {band.common_name!r} named twice, as {seen[band.common_name]!r} "
                f"and {spec!r}; each band can only be loaded once"
            )
        seen[band.common_name] = spec
    return resolved


def finest_resolution(bands: Sequence[BandInfo]) -> int:
    """Return the finest native pixel size among some bands, in metres.

    This is the default output grid for a load: it never discards detail a
    requested band actually has, and never invents detail by upsampling every
    band to a resolution only one of them was sensed at.

    Parameters
    ----------
    bands : sequence of BandInfo
        The bands being loaded.

    Returns
    -------
    int
        The smallest :attr:`BandInfo.resolution` among them.

    Raises
    ------
    ValidationError
        If no bands are given.

    Examples
    --------
    >>> finest_resolution(resolve_bands(["red", "swir16"], SENTINEL2))
    10
    """
    if not bands:
        raise ValidationError("name at least one band; got an empty sequence")
    return min(band.resolution for band in bands)


def sentinel2_available_resolutions(image_files: Sequence[str], band: BandInfo) -> list[int]:
    """List the resolutions a Sentinel-2 product actually holds a band at.

    Availability is read from the product rather than assumed, because it is
    irregular and baseline-dependent: ``B01`` is sensed at 60 m yet written at
    both 20 m and 60 m from baseline 04.00 onward, ``B08`` is written only at
    10 m, and ``B09`` only at 60 m.

    Parameters
    ----------
    image_files : sequence of str
        The manifest's image paths, extensionless, as
        :attr:`~eeo.io._sentinel2.Sentinel2Product.image_files` gives them.
    band : BandInfo
        The band to look for.

    Returns
    -------
    list of int
        Resolutions in metres, ascending. Empty when the product holds no file
        for the band at all.

    Examples
    --------
    >>> files = ["GRANULE/g/IMG_DATA/R20m/T32TPS_20240929T100719_B01_20m"]
    >>> sentinel2_available_resolutions(files, SENTINEL2["coastal"])
    [20]
    """
    pattern = re.compile(rf"/R(\d+)m/.*_{re.escape(band.band_id)}_(\d+)m$", re.IGNORECASE)
    found = {int(m.group(1)) for m in (pattern.search(f) for f in image_files) if m}
    return sorted(found)


def sentinel2_band_file(image_files: Sequence[str], band: BandInfo, resolution: int) -> str:
    """Find a Sentinel-2 band's image path at one resolution.

    Parameters
    ----------
    image_files : sequence of str
        The manifest's image paths, extensionless.
    band : BandInfo
        The band to find.
    resolution : int
        Pixel size in metres, one of 10, 20, or 60.

    Returns
    -------
    str
        The path, relative to the ``.SAFE`` directory and without a file
        extension — the manifest does not record one.

    Raises
    ------
    ValidationError
        If the product holds no file for the band at that resolution. The
        message names the resolutions it does hold, so a caller asking for a
        10 m ``swir16`` is told it exists at 20 m and 60 m rather than merely
        that something is missing.
    """
    pattern = re.compile(
        rf"/R{resolution}m/.*_{re.escape(band.band_id)}_{resolution}m$", re.IGNORECASE
    )
    for path in image_files:
        if pattern.search(path):
            return path

    available = sentinel2_available_resolutions(image_files, band)
    if not available:
        raise ValidationError(
            f"this product holds no {band.common_name!r} ({band.band_id}) band at all"
        )
    raise ValidationError(
        f"{band.common_name!r} ({band.band_id}) is not available at {resolution} m in "
        f"this product; it is written at {', '.join(f'{r} m' for r in available)}"
    )


def landsat_band_file(band_files: Mapping[str, str], band: BandInfo) -> str:
    """Find a Landsat band's image filename.

    Parameters
    ----------
    band_files : Mapping of str to str
        Filenames keyed by band token, as
        :attr:`~eeo.io._landsat.LandsatProduct.band_files` gives them.
    band : BandInfo
        The band to find.

    Returns
    -------
    str
        The filename, relative to the product directory.

    Raises
    ------
    ValidationError
        If the product lists no file for the band. A Level-2 reflectance-only
        product legitimately has no thermal band, so this is a normal outcome
        rather than a corrupt product.
    """
    filename = band_files.get(band.band_id)
    if filename is None:
        raise ValidationError(
            f"this product holds no {band.common_name!r} ({band.band_id}) band; "
            f"it holds {', '.join(sorted(band_files))}"
        )
    return filename
