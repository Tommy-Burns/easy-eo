"""Load a satellite scene from a product downloaded to disk.

:func:`load_sentinel2` opens a Copernicus ``.SAFE`` product the way
:mod:`eeo.io.stac` opens a catalogue item, and returns the same kind of
:class:`~eeo.core.core.EEORasterDataset`. Both paths share one reader, so a
scene loaded from a folder and the same scene loaded from a catalogue land on
the same grid with the same values and the same band names.

Bands are named, never numbered. ``load_sentinel2(path, ["red", "nir"])`` says
what it means on any sensor, where band 4 does not: see :mod:`eeo.io._bands`.

**Values are the product's stored integers** — the digital numbers a GIS shows,
not reflectance. Converting them is a documented one-liner rather than a
default, because the default here is to read what the file holds. See the
loading guide for the per-mission encodings; the coefficients for the product
in hand are carried on the result's ``attrs``.

Only Level-2A is read. Level-1C is top-of-atmosphere radiance with a different
layout, and is refused by name rather than failing on a missing directory.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
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
    resolve_bands,
    sentinel2_available_resolutions,
    sentinel2_band_file,
)
from eeo.io._sentinel2 import Sentinel2Product
from eeo.io._sentinel2 import read_product as read_sentinel2_product
from eeo.io._stacking import read_onto_common_grid

#: Resolutions a Sentinel-2 Level-2A product is written at.
_S2_RESOLUTIONS = (10, 20, 60)

#: Extensions to try for an image the manifest lists without one. Products are
#: JPEG 2000; a few pipelines rewrite them as GeoTIFF while keeping the layout.
_IMAGE_SUFFIXES = (".jp2", ".JP2", ".tif", ".TIF", ".tiff", ".TIFF")


def _image_path(safe: Path, relative: str) -> str:
    """Resolve a manifest image path, which is recorded without its extension."""
    for suffix in _IMAGE_SUFFIXES:
        candidate = safe / f"{relative}{suffix}"
        if candidate.is_file():
            return str(candidate)
    raise ValidationError(
        f"the manifest lists an image the product does not hold: "
        f"{relative!r} (tried {', '.join(_IMAGE_SUFFIXES)} under {safe})"
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
    return _image_path(product.path, sentinel2_band_file(product.image_files, band, chosen)), chosen


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
        A ``.SAFE`` directory, a directory holding one, or its product
        manifest.
    bands : sequence of str
        Bands to load, in the order they should become bands of the result.
        Named by common name (``"red"``, ``"nir"``, ``"swir16"``, ``"scl"``) or
        by the product's own id (``"B04"``, ``"B8A"``). **There is no default**:
        a full Level-2A product is roughly 2.9 GB of imagery at 10 m, so the
        set to read is always an explicit choice.
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

    Notes
    -----
    Holds the requested window in memory. A ``bbox`` is what keeps that
    bounded; without one, an all-band load of a whole tile will not fit on a
    laptop.

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
            "product": product.path.name,
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
