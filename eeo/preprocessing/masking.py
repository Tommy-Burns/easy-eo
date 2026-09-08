"""Turning a decoded quality layer into masked pixels.

:mod:`eeo.preprocessing.quality` reads Sentinel-2's ``SCL`` and Landsat's
``QA_PIXEL`` and reports which pixels a user does not want. This module is
what acts on that answer: one chainable operation that sets those pixels to
nodata across every band, so that the rest of the library — every index, every
statistic — excludes them without knowing anything about clouds.

There is one operation rather than one per mission because the decision is
sensor-agnostic once the quality layer has been decoded. Which decoder runs is
settled by the quality band's own name, so a workflow written for Sentinel-2
reads identically on Landsat.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import rasterio as rio

from eeo.common import _declared_nodata_mask, get_nodata, resolve_band_index
from eeo.core.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_op
from eeo.core.exceptions import AlignmentError, ValidationError
from eeo.preprocessing.quality import (
    QA_PIXEL_DEFAULT_MIN_CLOUD_CONFIDENCE,
    QAConfidence,
    QAPixelFlag,
    SCLClass,
    qa_pixel_mask,
    scl_mask,
)

#: Quality band names this operation knows how to decode, lowercased, mapping
#: to the decoder each one needs. Both the common name and the mission's own
#: id are listed, since a dataset may carry either.
_QUALITY_BANDS = {
    "scl": "scl",
    "qa_pixel": "qa_pixel",
}

#: Pulls the mission number out of the prose a loader records, e.g.
#: ``"Landsat 9"``. Sentinel-2 has no number to find and needs none.
_MISSION_NUMBER = re.compile(r"(\d+)\s*$")


def _quality_kind(name: str) -> str | None:
    """Return which decoder a band name calls for, or None if it is not one."""
    return _QUALITY_BANDS.get(name.strip().casefold())


def _find_quality_band(ds: EEORasterDataset) -> tuple[int, str]:
    """Find the one quality band in a dataset, by name.

    Returns its 1-based index and its kind. Raises when there is no quality
    band to be found, or more than one: picking between two silently would
    make the mask depend on band order.
    """
    found = [
        (index, kind)
        for index, name in enumerate(ds.band_names, start=1)
        if name and (kind := _quality_kind(name)) is not None
    ]
    if len(found) == 1:
        return found[0]
    declared = [n for n in ds.band_names if n]
    if not found:
        raise ValidationError(
            f"no quality band to mask from: none of this dataset's bands is named "
            f"{' or '.join(sorted(_QUALITY_BANDS))}. Its bands are "
            f"{', '.join(repr(n) for n in declared) if declared else 'unnamed'}. "
            f"Load the scene with its quality band (bands=[..., 'scl'] for "
            f"Sentinel-2, bands=[..., 'qa_pixel'] for Landsat), name the band with "
            f"mask_band=, or pass a separate mask dataset with mask="
        )
    names = ", ".join(repr(ds.band_names[i - 1]) for i, _ in found)
    raise ValidationError(
        f"this dataset has more than one quality band ({names}); name the one to "
        f"mask from with mask_band="
    )


def _resolve_mission(ds: EEORasterDataset, mission: int | None) -> int:
    """Settle which Landsat mission a QA_PIXEL band belongs to.

    An explicit argument wins. Otherwise the mission is read from the
    dataset's own ``attrs``, where a loader records it as prose.
    """
    if mission is not None:
        return mission
    recorded = ds.attrs.get("mission")
    if isinstance(recorded, str):
        found = _MISSION_NUMBER.search(recorded)
        if found is not None:
            return int(found.group(1))
    raise ValidationError(
        "reading QA_PIXEL needs to know which Landsat took the scene, because the "
        "same bit is not the same flag on every mission: bit 2 is cirrus on "
        "Landsat 8 and 9 and Unused on 4, 5 and 7. This dataset records no mission "
        f"({recorded!r}), so pass mission=9 (or whichever it is) explicitly"
    )


def _resolve_nodata(ds: EEORasterDataset, dtype: np.dtype, nodata) -> int | float:
    """Settle the value a masked pixel is set to, per the nodata contract."""
    declared = ds.get_metadata().get("nodata") if nodata is None else nodata
    if declared is None:
        if np.issubdtype(dtype, np.floating):
            return float("nan")
        raise ValidationError(
            f"this raster is {dtype} and declares no nodata value, so there is no "
            f"value a masked pixel could be set to — an integer array cannot hold "
            f"NaN. Pass nodata= explicitly (Sentinel-2 L2A and Landsat Collection 2 "
            f"both use 0 as their fill value), or convert the raster to float first"
        )
    if np.issubdtype(dtype, np.integer):
        as_float = float(declared)
        if as_float != int(as_float):
            raise ValidationError(
                f"nodata={declared!r} is not a whole number, so a {dtype} raster "
                f"cannot hold it; pass a whole number or convert to float first"
            )
        info = np.iinfo(dtype)
        if not info.min <= int(as_float) <= info.max:
            raise ValidationError(
                f"nodata={declared!r} does not fit in {dtype} "
                f"(which holds {info.min} to {info.max})"
            )
        return int(as_float)
    return float(declared)


def _quality_array(
    ds: EEORasterDataset,
    mask: EEORasterDataset | None,
    mask_band: int | str | None,
) -> tuple[np.ndarray, str]:
    """Read the quality band, from this dataset or from a separate one."""
    source = ds if mask is None else mask
    if mask is not None and (
        ds.get_shape() != mask.get_shape() or ds.get_transform() != mask.get_transform()
    ):
        raise AlignmentError(
            f"the mask must be on the same pixel grid as the raster it masks; "
            f"got shape {mask.get_shape()} vs {ds.get_shape()}. Reproject or "
            f"resample the mask onto this grid first — never with an "
            f"interpolating method, which would blend class numbers into "
            f"classes nobody measured"
        )
    index: int
    kind: str | None
    if mask_band is None:
        if mask is not None and mask.get_count() == 1 and not any(mask.band_names):
            # A single unnamed band in a dataset passed as the mask can only
            # be the mask, but its encoding still has to be stated.
            raise ValidationError(
                "the mask dataset's band is unnamed, so there is no way to tell "
                "whether it holds Sentinel-2 scene classes or Landsat quality bits; "
                "name it with mask_band='scl' or mask_band='qa_pixel'"
            )
        index, kind = _find_quality_band(source)
        return source.read()[index - 1], kind

    # A name that is itself a quality band's name settles the encoding even
    # when the dataset never declared one.
    kind = _quality_kind(mask_band) if isinstance(mask_band, str) else None
    try:
        index = resolve_band_index(source, mask_band)
    except ValidationError:
        # "scl" naming no band of a one-band mask dataset is not a mistake:
        # it is the caller stating what that band holds.
        if kind is None or source.get_count() != 1:
            raise
        index = 1
    if kind is None:
        name = source.band_names[index - 1] or ""
        kind = _quality_kind(name)
    if kind is None:
        raise ValidationError(
            f"band {mask_band!r} is not a quality band this operation can decode; "
            f"it knows {' and '.join(sorted(_QUALITY_BANDS))}"
        )
    return source.read()[index - 1], kind


@eeo_raster_op
def mask_clouds(
    ds: EEORasterDataset,
    *,
    mask_band: int | str | None = None,
    mask: EEORasterDataset | None = None,
    classes: Iterable[SCLClass | int | str] | None = None,
    flags: Iterable[QAPixelFlag | int | str] | None = None,
    min_cloud_confidence: QAConfidence | int | None = QA_PIXEL_DEFAULT_MIN_CLOUD_CONFIDENCE,
    mission: int | None = None,
    nodata: int | float | None = None,
) -> EEORasterDataset:
    """Set cloudy pixels to nodata, using the scene's own quality band.

    Decodes Sentinel-2's ``SCL`` or Landsat's ``QA_PIXEL`` — whichever the
    dataset carries — and writes nodata wherever it says the pixel is not a
    view of the ground. Which decoder runs is settled by the quality band's
    name, so the same call works on either mission.

    Parameters
    ----------
    ds : EEORasterDataset
        Raster to mask, carrying a quality band among its own bands unless
        ``mask`` is given.
    mask_band : int or str, optional
        Which band holds the quality layer, as a 1-based index or a band name.
        Defaults to the one band named ``"scl"`` or ``"qa_pixel"``; having
        none, or more than one, is an error rather than a guess.
    mask : EEORasterDataset, optional
        A separately loaded quality layer, on the same pixel grid as ``ds``.
        Use it when the mask was not loaded alongside the data.
    classes : iterable of SCLClass or int or str, optional
        For an ``SCL`` band: which scene classes to mask. Defaults to
        :data:`~eeo.preprocessing.quality.SCL_DEFAULT_MASKED`.
    flags : iterable of QAPixelFlag or int or str, optional
        For a ``QA_PIXEL`` band: which flags to mask on. Defaults to
        :data:`~eeo.preprocessing.quality.QA_PIXEL_DEFAULT_MASKED`.
    min_cloud_confidence : QAConfidence or int or None, optional
        For a ``QA_PIXEL`` band: also mask pixels whose cloud confidence is at
        least this level, Medium by default. See
        :func:`~eeo.preprocessing.quality.qa_pixel_mask` for why the flags
        alone are not enough.
    mission : int, optional
        For a ``QA_PIXEL`` band: which Landsat took the scene. Read from the
        dataset's ``attrs`` when a loader recorded it, and required otherwise,
        because the same bit is not the same flag on every mission.
    nodata : int or float, optional
        The value masked pixels are set to. Defaults to the raster's declared
        nodata; for a float raster with none declared, NaN.

    Returns
    -------
    EEORasterDataset
        A new dataset of the same shape, band count and dtype, with masked
        pixels set to nodata in **every** band and that value recorded in the
        result's metadata. The quality band is masked along with the rest, so
        it cannot be used as a mask a second time; keep the original if you
        need it. Band names, CRS, transform and attrs are preserved.

    Raises
    ------
    ValidationError
        If no quality band can be found or more than one is present; if the
        named band is not a quality layer; if ``classes`` is given for a
        ``QA_PIXEL`` band or ``flags`` for an ``SCL`` band; if a ``QA_PIXEL``
        band's mission is neither recorded nor given; or if the raster is an
        integer type declaring no nodata and none is passed, leaving no value
        a masked pixel could hold.
    AlignmentError
        If ``mask`` is not on the same pixel grid as ``ds``.

    Notes
    -----
    Reads the full array into memory rather than streaming block-wise.

    A quality band must never have been resampled by anything but nearest
    neighbour: class numbers and packed bits are labels, and interpolating
    them produces values nobody measured. The loaders honour this.

    Examples
    --------
    >>> ds = eeo.load_sentinel2(path, bands=["red", "nir", "scl"])  # doctest: +SKIP
    >>> clear = ds.mask_clouds(nodata=0)  # doctest: +SKIP
    >>> ndvi = clear.normalized_difference("nir", "red")  # doctest: +SKIP

    Keeping thin cirrus, which darkens rather than hides the ground:

    >>> ds.mask_clouds(classes=[0, 1, 3, 8, 9], nodata=0)  # doctest: +SKIP
    """
    if mask is not None and mask_band is None and mask.get_count() > 1:
        raise ValidationError(
            "the mask dataset has more than one band; name the quality band with mask_band="
        )
    quality, kind = _quality_array(ds, mask, mask_band)

    if kind == "scl":
        if flags is not None:
            raise ValidationError(
                "flags= describes Landsat QA_PIXEL bits, but this is a Sentinel-2 "
                "SCL band, which holds one scene class per pixel; use classes="
            )
        flagged = scl_mask(quality, classes=classes)
    else:
        if classes is not None:
            raise ValidationError(
                "classes= describes Sentinel-2 scene classes, but this is a Landsat "
                "QA_PIXEL band, which packs independent flags into bits; use flags="
            )
        flagged = qa_pixel_mask(
            quality,
            mission=_resolve_mission(ds, mission),
            flags=flags,
            min_cloud_confidence=min_cloud_confidence,
        )

    data = ds.read()
    dtype = np.dtype(data.dtype)
    fill = _resolve_nodata(ds, dtype, nodata)
    out = np.where(flagged, np.array(fill, dtype=dtype), data)

    meta = ds.get_metadata()
    meta.update(dtype=dtype, nodata=fill)
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(out)
    return EEORasterDataset.from_rasterio(out_ds)


@eeo_raster_op
def clear_fraction(ds: EEORasterDataset, *, band: int | str | None = None) -> float:
    """Report the share of pixels that still hold a measurement.

    The number a user needs to decide whether a scene is worth keeping. After
    :func:`mask_clouds` it is the clear fraction in the usual sense — the
    proportion of the raster that is neither cloud nor fill — and on any other
    raster it is simply how much of it is not nodata.

    Parameters
    ----------
    ds : EEORasterDataset
        Raster to measure.
    band : int or str, optional
        Restrict the count to one band, as a 1-based index or a band name.
        By default a pixel counts as clear only where **every** band holds a
        measurement, following the rule that nodata is contagious. After
        ``mask_clouds`` the two agree, because it writes one mask to every
        band; on a raster whose bands were masked separately they need not.

    Returns
    -------
    float
        A fraction in ``[0, 1]``. A raster declaring no nodata returns
        ``1.0``: with nothing marked absent, every pixel counts as a
        measurement, which is the same reading the rest of the library takes.

    Raises
    ------
    IndexError
        If ``band`` is an index outside the range of available bands.
    ValidationError
        If ``band`` is a name that is unknown or matches more than one band.

    Notes
    -----
    Reads the array into memory rather than streaming block-wise.

    It counts *every* absent pixel, which on a whole satellite scene includes
    the fill outside the sensor's footprint — a Landsat scene on a north-up
    grid is roughly a third fill before any cloud is masked at all, so a
    clear fraction over the full scene answers "how much of this raster is
    usable", not "how cloudy was it". Clip to the area you care about first
    if you want the second question answered.

    The difference is not small. On a real Landsat 9 scene the whole-scene
    figure falls from 63.1% to 60.4% under masking, so only 2.7 points of the
    39.6% lost is cloud and the rest is the scene's own edge; the same scene
    clipped to its centre reads 100% before masking and 96.8% after.

    Examples
    --------
    >>> scene.mask_clouds().clear_fraction()  # doctest: +SKIP
    0.6036
    >>> scene.clip_raster_with_bbox(aoi).mask_clouds().clear_fraction()  # doctest: +SKIP
    0.9677
    """
    nodata = get_nodata(ds)
    if nodata is None:
        return 1.0

    data = ds.read() if band is None else ds.read()[resolve_band_index(ds, band) - 1][np.newaxis]
    absent = _declared_nodata_mask(data, nodata)
    if absent is None:  # pragma: no cover - guarded by the nodata check above
        return 1.0
    # Nodata is contagious: a pixel missing from any band is not a measurement.
    per_pixel = absent.any(axis=0)
    return float(1.0 - per_pixel.sum() / per_pixel.size)
