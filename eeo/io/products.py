"""Load a satellite scene from a product downloaded to disk.

:func:`load_sentinel2` opens a Copernicus ``.SAFE`` product and
:func:`load_landsat` a USGS Collection 2 Level-2 product, the way
:mod:`eeo.io.stac` opens a catalogue item, and all three return the same kind
of :class:`~eeo.core.core.EEORasterDataset`. Every path shares one reader, so a
scene loaded from a folder and the same scene loaded from a catalogue land on
the same grid with the same values and the same band names.

Bands are named, never numbered. ``load_sentinel2(path, ["red", "nir"])`` says
what it means on any sensor, where band 4 does not — and it is the same call on
Landsat, where ``"red"`` is band 4 on Landsat 8 and band 3 on Landsat 7: see
:mod:`eeo.io._bands`.

**Values are the product's stored integers** — the digital numbers a GIS shows,
not reflectance. Converting them is a documented one-liner rather than a
default, because the default here is to read what the file holds. See the
loading guide for the per-mission encodings; the coefficients for the product
in hand are carried on the result's ``attrs``.

Only surface-reflectance levels are read: Sentinel-2 Level-2A and Landsat
Level-2 (``L2SP``/``L2SR``). Their top-of-atmosphere counterparts hold
different quantities under a different layout, and are refused by name rather
than left to fail on a missing directory.

There is one loader per mission family, not one per satellite. Landsat 8 and 9
share a product format exactly, and Landsat 4, 5, and 7 differ from them only
in which band number carries which wavelength — a difference the band table
already holds, so four functions would be four copies of one.

Either loader reads a product as it was downloaded — the unpacked directory,
or the archive it arrived in — because :mod:`eeo.io._archive` answers the
questions about *where a file is* that identifying a product asks. A compressed
archive is the one container refused, and refused with instructions; see that
module for why.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from eeo.common import normalize_resampling_method
from eeo.core.core import EEORasterDataset
from eeo.core.exceptions import ValidationError
from eeo.core.loader import load_array
from eeo.core.types import ResamplingMethod, StrPath
from eeo.io._bands import (
    SENTINEL2,
    BandInfo,
    finest_resolution,
    landsat_band_file,
    landsat_bands,
    resolve_bands,
    sentinel2_available_resolutions,
    sentinel2_band_file,
)
from eeo.io._landsat import LandsatProduct
from eeo.io._landsat import read_product as read_landsat_product
from eeo.io._sentinel2 import Sentinel2Product
from eeo.io._sentinel2 import read_product as read_sentinel2_product
from eeo.io._stacking import read_onto_common_grid

#: Resolutions a Sentinel-2 Level-2A product is written at.
_S2_RESOLUTIONS = (10, 20, 60)

#: The single grid every Collection 2 Level-2 band is delivered on, in metres.
#: The thermal band reaches it by USGS resampling from its coarser native
#: sampling — 120 m on TM, 60 m on ETM+, 100 m on TIRS — before delivery.
_LANDSAT_RESOLUTION = 30

#: Extensions to try for an image the manifest lists without one. Products are
#: JPEG 2000; a few pipelines rewrite them as GeoTIFF while keeping the layout.
_IMAGE_SUFFIXES = (".jp2", ".JP2", ".tif", ".TIF", ".tiff", ".TIFF")


def _image_path(product: Sentinel2Product, relative: str) -> str:
    """Resolve a manifest image path, which is recorded without its extension."""
    for suffix in _IMAGE_SUFFIXES:
        candidate = product.root / f"{relative}{suffix}"
        if product.source.exists(candidate):
            return product.source.href(candidate)
    raise ValidationError(
        f"the manifest lists an image the product does not hold: "
        f"{relative!r} (tried {', '.join(_IMAGE_SUFFIXES)} in {product.source.location})"
    )


def _band_source(product: Sentinel2Product, band: BandInfo, resolution: int) -> tuple[str, int]:
    """Pick the file for one band, and report the resolution it was found at.

    Prefers the requested resolution, and otherwise takes the finest the
    product actually holds — a 20 m band asked for at 10 m is read at 20 m and
    warped onto the 10 m grid, exactly as the catalogue path does with a 20 m
    asset stacked onto a 10 m one.
    """
    available = sentinel2_available_resolutions(product.image_files, band)
    if not available:
        raise ValidationError(
            f"this product holds no {band.common_name!r} ({band.band_id}) band at all"
        )
    chosen = resolution if resolution in available else available[0]
    return _image_path(product, sentinel2_band_file(product.image_files, band, chosen)), chosen


def load_sentinel2(
    path: StrPath,
    bands: Sequence[str],
    *,
    bbox: Sequence[float] | None = None,
    resolution: int | None = None,
    resampling: ResamplingMethod | Any = "nearest",
    level: str | None = None,
) -> EEORasterDataset:
    """Load bands from a downloaded Sentinel-2 Level-2A product.

    Parameters
    ----------
    path : str or pathlib.Path
        A ``.SAFE`` directory, a directory or ``.zip`` holding one, or the
        product manifest itself. A Copernicus download is read as it arrives,
        zipped, without being unpacked first.
    bands : sequence of str
        Bands to load, in the order they should become bands of the result.
        Named by common name (``"red"``, ``"nir"``, ``"swir16"``, ``"scl"``) or
        by the product's own id (``"B04"``, ``"B8A"``). **There is no default**:
        a Level-2A product holds several gigabytes of imagery — one 10 m band
        alone is about 241 MB — so the set to read is always an explicit
        choice.
    bbox : sequence of float or None, default None
        Area to read, as ``(minx, miny, maxx, maxy)`` in **WGS 84 lon/lat
        degrees**. None reads the whole 110 km tile, which for a 10 m band is
        about 240 MB before any others are stacked onto it.
    resolution : int or None, default None
        Output pixel size in metres — 10, 20, or 60. None picks the finest
        *native* resolution among the bands actually requested, so a load never
        upsamples every band to a resolution only one of them was sensed at.
        A band the product does not hold at the chosen resolution is read at
        the finest it does hold and warped onto the grid.
    resampling : str or rasterio.enums.Resampling, default "nearest"
        Method used where a band has to be warped onto the output grid. Applied
        to continuous bands only: a quality band such as ``"scl"`` holds class
        numbers, so it is always read by nearest neighbour whatever is asked
        for here, since blending class numbers invents classes.
    level : str or None, default None
        Assert the processing level rather than detecting it. The manifest is
        the authority, so this raises if it disagrees.

    Returns
    -------
    EEORasterDataset
        Rasterio-backed dataset holding the requested bands as the product's
        own ``uint16`` digital numbers — the values a GIS shows, not
        reflectance. Carries the tile's CRS, a transform matching the requested
        window, the first band's nodata value, the granule's acquisition time
        as ``timestamp``, common names as ``band_names``, and the product's
        identity plus its reflectance coefficients in ``attrs``.

    Raises
    ------
    ValidationError
        If no product is found, it is not Level-2A, a band is not recognised or
        not present, ``resolution`` is not 10, 20, or 60, no requested band
        exists at ``resolution``, or ``bbox`` does not overlap the tile.

    See Also
    --------
    load_landsat : The same call for a USGS Collection 2 Level-2 product.

    Notes
    -----
    Holds the requested window in memory. A ``bbox`` is what keeps that
    bounded; without one, an all-band load of a whole tile will not fit on a
    laptop.

    Reading straight from the ``.zip`` costs something: a deflated member is
    decompressed from its own start, so a small window of one band still pays
    for the image ahead of it. It is bounded by the size of one image, and it
    saves unpacking a product to read two bands of it. Unpack first if the same
    scene will be read many times.

    Examples
    --------
    >>> scene = load_sentinel2(
    ...     "S2B_MSIL2A_20240929T100719.SAFE", ["red", "nir"]
    ... )  # doctest: +SKIP
    >>> scene.band_names  # doctest: +SKIP
    ['red', 'nir']
    >>> ndvi = scene.ndvi(red="red", nir="nir")  # doctest: +SKIP
    """
    product = read_sentinel2_product(path, level=level)
    resolved = resolve_bands(bands, SENTINEL2)

    if resolution is None:
        resolution = finest_resolution(resolved)
    elif resolution not in _S2_RESOLUTIONS:
        raise ValidationError(
            f"resolution must be one of {', '.join(f'{r} m' for r in _S2_RESOLUTIONS)}; "
            f"got {resolution!r}"
        )

    located = [_band_source(product, band, resolution) for band in resolved]

    # The first source read defines the output grid, so it has to be a band the
    # product actually holds at the requested resolution. The caller's order is
    # restored afterwards, so which band leads is invisible from outside.
    lead = next((i for i, (_, found) in enumerate(located) if found == resolution), None)
    if lead is None:
        holdings = ", ".join(
            f"{band.common_name} at "
            f"{', '.join(f'{r} m' for r in sentinel2_available_resolutions(product.image_files, band))}"
            for band in resolved
        )
        raise ValidationError(
            f"none of the requested bands is written at {resolution} m in this "
            f"product, so there is no grid to read onto ({holdings}). Request a "
            f"resolution one of them has, or add a band that does."
        )

    order = [lead] + [i for i in range(len(resolved)) if i != lead]
    sources = [(located[i][0], resolved[i].common_name) for i in order]
    # Class numbers and packed bits must never be blended.
    methods = ["nearest" if resolved[i].kind == "quality" else resampling for i in order]

    stacked, grid, nodata, _ = read_onto_common_grid(
        sources,
        bbox=bbox,
        resampling=[normalize_resampling_method(m) for m in methods],
    )

    # Undo the read order so the bands come back as the caller asked for them.
    restore = [order.index(i) for i in range(len(resolved))]
    dataset = load_array(
        stacked[restore],
        transform=grid.transform,
        crs=grid.crs,
        nodata=nodata,
        timestamp=product.sensing_time,
        attrs={
            "product": product.name,
            "mission": "Sentinel-2",
            "level": product.level,
            "tile": product.tile_id,
            "processing_baseline": product.processing_baseline,
            "resolution": resolution,
            "bands": [band.common_name for band in resolved],
            "band_ids": [band.band_id for band in resolved],
            "quantification_value": product.quantification_value,
            "band_offsets": product.band_offsets,
        },
        band_names=[band.common_name for band in resolved],
    )
    return dataset.to_rasterio()


def _landsat_image_path(product: LandsatProduct, band: BandInfo) -> str:
    """Resolve one Landsat band to a file, which the metadata names in full."""
    filename = landsat_band_file(product.band_files, band)
    candidate = product.root / filename
    if not product.source.exists(candidate):
        raise ValidationError(
            f"the metadata lists {filename!r} for {band.common_name!r}, but the "
            f"product does not hold it ({product.source.location})"
        )
    return product.source.href(candidate)


def load_landsat(
    path: StrPath,
    bands: Sequence[str],
    *,
    bbox: Sequence[float] | None = None,
    level: str | None = None,
) -> EEORasterDataset:
    """Load bands from a downloaded Landsat Collection 2 Level-2 product.

    One function covers Landsat 4, 5, 7, 8, and 9. The mission is read from the
    product's own metadata and only decides which band table the names are
    resolved against, so ``["red", "nir08"]`` is the same request on every one
    of them even though it means bands 3 and 4 on Landsat 7 and bands 4 and 5
    on Landsat 8.

    Parameters
    ----------
    path : str or pathlib.Path
        A product directory, a directory or ``.tar`` holding one, or the
        ``*_MTL.xml``, ``*_MTL.json``, or ``*_MTL.txt`` metadata file itself. A
        USGS download is read as it arrives, tarred, without being unpacked
        first — and an uncompressed tar costs nothing to read this way, since
        its members are stored whole.
    bands : sequence of str
        Bands to load, in the order they should become bands of the result.
        Named by common name (``"red"``, ``"nir08"``, ``"swir16"``,
        ``"qa_pixel"``) or by the product's own token (``"SR_B4"``,
        ``"ST_B10"``), with or without its prefix (``"B4"``). **There is no
        default**: the set to read is always an explicit choice.
    bbox : sequence of float or None, default None
        Area to read, as ``(minx, miny, maxx, maxy)`` in **WGS 84 lon/lat
        degrees**. None reads the whole scene, which is roughly 7700 x 7900
        pixels — about 120 MB per band.
    level : str or None, default None
        Assert the processing level rather than detecting it. The metadata is
        the authority, so this raises if it disagrees.

    Returns
    -------
    EEORasterDataset
        Rasterio-backed dataset holding the requested bands as the product's
        own ``uint16`` digital numbers — the values a GIS shows, not
        reflectance or kelvin. Carries the scene's CRS, a transform matching
        the requested window, a nodata value, the acquisition time as
        ``timestamp``, common names as ``band_names``, and the product's
        identity plus its scaling coefficients in ``attrs``.

    Raises
    ------
    ValidationError
        If no product is found, it is not Collection 2 Level-2, its mission has
        no band table, a band is not recognised or not present in the product,
        a listed file is missing from the directory, or ``bbox`` does not
        overlap the scene.

    See Also
    --------
    load_sentinel2 : The same call for a Copernicus ``.SAFE`` product.

    Notes
    -----
    **There is no resolution or resampling parameter, because nothing is ever
    resampled.** Every band of a Collection 2 Level-2 product — reflectance,
    surface temperature, and the quality rasters alike — is delivered on one
    30 m grid of identical size and origin, the thermal band having already
    been resampled onto it by USGS from its coarser native sampling. So the
    bands are read by exact windows and stacked, and a parameter offering to
    choose a method would describe something that never happens.

    **A reflectance-only product has no thermal band.** ``L2SR`` is issued for
    scenes where surface temperature could not be produced, so asking for
    ``"lwir11"`` there fails on a band the product genuinely does not hold
    rather than on a bad name.

    **Landsat 7 after 31 May 2003 is missing about 22% of each scene.** The
    scan line corrector failed, leaving wedge-shaped gaps that widen toward the
    scene edges. Those pixels carry the fill value, not a measurement, so they
    are nodata and must be excluded from statistics rather than read as zero
    reflectance. It is a property of the sensor, not of the product or of this
    reader.

    Fill is ``0`` in the reflectance and temperature bands but ``1`` in the
    quality rasters, and a dataset carries one nodata value. The first band
    read supplies it, and a data band is always read first when one was asked
    for, so the value on the result describes the imagery.

    Holds the requested window in memory. A ``bbox`` is what keeps that
    bounded.

    Examples
    --------
    >>> scene = load_landsat(
    ...     "LC09_L2SP_193028_20260822_20260823_02_T1", ["red", "nir08"]
    ... )  # doctest: +SKIP
    >>> scene.band_names  # doctest: +SKIP
    ['red', 'nir08']
    >>> ndvi = scene.ndvi(red="red", nir="nir08")  # doctest: +SKIP
    """
    product = read_landsat_product(path, level=level)
    resolved = resolve_bands(bands, landsat_bands(product.mission))

    # Quality rasters declare a fill value of 1 where the imagery declares 0,
    # and the first source read is the one that supplies the result's nodata.
    # Leading with a data band keeps that value describing the measurements.
    # The caller's order is restored afterwards, so this is invisible outside.
    lead = next((i for i, band in enumerate(resolved) if band.kind != "quality"), 0)
    order = [lead] + [i for i in range(len(resolved)) if i != lead]
    sources = [(_landsat_image_path(product, resolved[i]), resolved[i].common_name) for i in order]

    stacked, grid, nodata, _ = read_onto_common_grid(
        sources,
        bbox=bbox,
        # Every band shares one grid, so this is never reached. Nearest is what
        # it should be if a reprocessed product ever breaks that assumption:
        # it is the only method that cannot invent a class number.
        resampling=normalize_resampling_method("nearest"),
    )

    restore = [order.index(i) for i in range(len(resolved))]
    dataset = load_array(
        stacked[restore],
        transform=grid.transform,
        crs=grid.crs,
        nodata=nodata,
        timestamp=product.acquired,
        attrs={
            "product": product.product_id,
            "mission": f"Landsat {product.mission}",
            "instrument": product.instrument,
            "spacecraft": product.spacecraft,
            "level": product.level,
            "wrs_path": product.wrs_path,
            "wrs_row": product.wrs_row,
            "collection_number": product.collection_number,
            "collection_category": product.collection_category,
            "resolution": _LANDSAT_RESOLUTION,
            "bands": [band.common_name for band in resolved],
            "band_ids": [band.band_id for band in resolved],
            "reflectance_scaling": product.reflectance_scaling,
            "temperature_scaling": product.temperature_scaling,
        },
        band_names=[band.common_name for band in resolved],
    )
    return dataset.to_rasterio()
