"""Spectral indices built on the raster algebra primitives.

Each index (NDVI, NDWI, NDMI, NDBI, EVI, SAVI) is a thin, chainable wrapper
that combines a handful of bands into a nodata-safe, float32 index raster.

Every band argument accepts an :class:`~eeo.core.core.EEORasterDataset` (a
separate single-band raster), an ``int`` (a 1-based band index into the
receiver), or a ``str`` (the name of one of the receiver's bands), so the same
method serves both a per-band collection of rasters and one stacked multi-band
scene, addressed by number or by name. The primary band defaults to index
``1``, which keeps the separate-band case a clean one-liner
(``nir_ds.ndvi(red_ds)``).

An index band is a new measurement that maps to no input band, so it is never
named after the operation: the output band is unnamed unless the caller passes
``name=``.

Every index returns a new single-band ``EEORasterDataset``, so a result always
keeps chaining. When the raw values are what you want, read them off the
result: ``.get_band(1)`` for the 2D band, or ``.to_array()`` for the
``(bands, height, width)`` array.

The general two-operand primitive, :func:`normalized_difference`, is also kept
here: any normalized-difference index can be expressed with it directly.
"""

import numpy as np
import rasterio as rio

from eeo.common import align_raster_to_target, resolve_band_index
from eeo.core.blockwise import BlockSource, apply_blockwise
from eeo.core.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_op
from eeo.core.exceptions import AlignmentError, ValidationError

BandSpec = EEORasterDataset | int | str

_ALIGN_MISMATCH = (
    "rasters must share the same grid for this operation; "
    "got shape {other} vs {ds}. "
    "Pass auto_align=True to resample the other raster onto this grid."
)


def _safe_ratio(numerator, denominator):
    """Divide element-wise, mapping a zero denominator to 0 instead of inf/nan.

    Uses ``numpy.where`` (not in-place assignment) so the expression stays
    dispatchable to lazy array backends, and returns float32.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        quotient = numerator / denominator
    return np.where(denominator != 0, quotient, np.float32(0)).astype(rio.float32)


def _resolve_band(ds, spec, *, auto_align, method) -> BlockSource:
    """Resolve a band spec to the :class:`BlockSource` that reads it.

    ``spec`` is a 1-based ``int`` band index into ``ds``, the ``str`` name of
    one of ``ds``'s bands, or a separate ``EEORasterDataset`` (its first band
    is used, aligned onto ``ds``'s grid when ``auto_align`` is True). Index and
    name specs both go through the shared resolver, so a string is always a
    name and never an index. The source reads the band in its own dtype, so
    the nodata mask compares against the declared sentinel exactly.
    """
    if isinstance(spec, EEORasterDataset):
        other = spec
        if ds.get_shape() != other.get_shape() or ds.get_transform() != other.get_transform():
            if auto_align:
                other = align_raster_to_target(other, ds, method=method)
            else:
                raise AlignmentError(
                    _ALIGN_MISMATCH.format(other=other.get_shape(), ds=ds.get_shape())
                )
        return BlockSource.from_dataset(other, band=1)
    if isinstance(spec, (int, str)) and not isinstance(spec, bool):
        return BlockSource.from_dataset(ds, band=resolve_band_index(ds, spec))
    raise ValidationError(
        "band must be an EEORasterDataset, a 1-based int band index, or a "
        f"band name; got {type(spec).__name__}"
    )


def _compute_index(ds, band_specs, formula, *, auto_align, method, name=None):
    """Resolve band specs, apply ``formula`` block-wise, and package the result.

    ``band_specs`` is the ordered list of band specs; the first is the primary
    band. ``formula`` maps the list of float32 band arrays to a 2D result, and
    is called once per window rather than once over the whole scene, so an
    index never holds more than a block of each band in memory. The result is
    masked per the nodata contract (contagious across every band) and returned
    as a single-band float32 ``EEORasterDataset`` whose band carries ``name``
    (unnamed when ``name`` is None).
    """
    ds = ds.to_rasterio()
    sources = [_resolve_band(ds, spec, auto_align=auto_align, method=method) for spec in band_specs]

    def compute(*blocks):
        # The formulas are written against float32 bands; the engine masks
        # against these raw blocks, which keep their source dtype.
        return formula([block.astype(rio.float32) for block in blocks])

    result = apply_blockwise(ds, compute, sources=sources, fractional=True)
    if name is not None:
        result.set_band_name(1, name)
    return result


def _normalized_difference(bands):
    """Return ``(a - b) / (a + b)`` for the two-band normalized-difference family."""
    a, b = bands
    return _safe_ratio(a - b, a + b)


@eeo_raster_op(propagate_band_names=False)
def normalized_difference(
    ds: EEORasterDataset,
    other: EEORasterDataset,
    *,
    auto_align: bool = True,
    method: str = "bilinear",
    name: str | None = None,
) -> EEORasterDataset:
    """Compute the normalized difference ``(ds - other) / (ds + other)``.

    This is the general primitive underlying the normalized-difference indices
    (NDVI, NDWI, NDMI, NDBI): ``ds`` is the first band of the pair, ``other``
    the second. Unlike the named indices, it operates on whole datasets
    band-by-band rather than selecting a single band. NumPy-backed inputs are
    promoted to rasterio, and ``other`` is resampled onto ``ds``'s grid when
    ``auto_align`` is True.

    Parameters
    ----------
    ds : EEORasterDataset
        First operand (e.g. NIR for NDVI).
    other : EEORasterDataset
        Second operand (e.g. Red for NDVI).
    auto_align : bool, default True
        If True, resample ``other`` onto ``ds``'s grid when their shape or
        transform differ. If False, a mismatch raises ``AlignmentError``.
    method : str, default "bilinear"
        Resampling method used when ``auto_align`` triggers alignment; one of
        rasterio's resampling names (e.g. ``"nearest"``, ``"bilinear"``).
    name : str or None, default None
        Optional name for the output band. The result is never named
        automatically; ``name`` applies only to a single-band result — for a
        multi-band one, assign ``band_names`` on the result instead.

    Returns
    -------
    EEORasterDataset
        Float32 result in ``[-1, 1]``, one band per input band. Pixels where
        ``ds + other == 0`` are set to 0. A pixel that is nodata in either operand is nodata (NaN) in the
        output; the output nodata value is NaN when either input declares
        nodata, otherwise None.

    Raises
    ------
    AlignmentError
        If the two rasters are on different grids and ``auto_align`` is False.
    ValidationError
        If ``name`` is given for a result with more than one band.

    Notes
    -----
    Nodata pixels are masked before the ratio; separately, a zero denominator
    (``ds + other == 0``) is guarded by setting those pixels to 0.

    Examples
    --------
    >>> ndvi = ds_nir.normalized_difference(ds_red)
    >>> ndvi_array = ds_nir.normalized_difference(ds_red).to_array()
    """
    ds = ds.to_rasterio()
    if ds.get_shape() != other.get_shape() or ds.get_transform() != other.get_transform():
        if auto_align:
            other = align_raster_to_target(other, ds, method=method)
        else:
            raise AlignmentError(_ALIGN_MISMATCH.format(other=other.get_shape(), ds=ds.get_shape()))

    def difference(a_block, b_block):
        # Masking is the engine's job and happens after this returns, so a
        # masked pixel is NaN regardless of the ratio computed for it here.
        a = a_block.astype(rio.float32)
        b = b_block.astype(rio.float32)
        return _safe_ratio(a - b, a + b)

    result = apply_blockwise(
        ds,
        difference,
        sources=[BlockSource.from_dataset(ds), BlockSource.from_dataset(other)],
        fractional=True,
    )
    if name is not None:
        if result.get_count() != 1:
            raise ValidationError(
                f"name applies to a single-band result, but this normalized_difference "
                f"produced {result.get_count()} bands; assign result.band_names instead"
            )
        result.set_band_name(1, name)
    return result


@eeo_raster_op(propagate_band_names=False)
def ndvi(
    ds: EEORasterDataset,
    red: BandSpec,
    *,
    nir: BandSpec = 1,
    auto_align: bool = True,
    method: str = "bilinear",
    name: str | None = None,
) -> EEORasterDataset:
    """Compute the Normalized Difference Vegetation Index.

    ``NDVI = (NIR - Red) / (NIR + Red)``. Higher values indicate denser, more
    photosynthetically active vegetation; values fall in ``[-1, 1]``.

    Each band is either a separate ``EEORasterDataset``, a 1-based ``int`` band
    index into ``ds``, or the name of one of ``ds``'s bands. The NIR band defaults
    to band ``1`` of ``ds``, so calling on a single-band NIR raster and passing the
    Red raster is enough.

    Parameters
    ----------
    ds : EEORasterDataset
        Receiver raster. When ``nir`` is an int index, this is the multi-band
        scene the bands are read from; when ``nir`` defaults to band 1, this is
        the NIR band itself.
    red : EEORasterDataset or int or str
        Red band, as a separate raster, a 1-based band index into ``ds``, or
        one of ``ds``'s band names. Sentinel-2: B4; Landsat 8/9 (OLI): B4.
    nir : EEORasterDataset or int or str, default 1
        NIR band, as a separate raster, a 1-based band index into ``ds``, or
        one of ``ds``'s band names. Sentinel-2: B8; Landsat 8/9 (OLI): B5.
    auto_align : bool, default True
        If True, resample a dataset band onto ``ds``'s grid when their shape or
        transform differ. If False, a mismatch raises ``AlignmentError``.
    method : str, default "bilinear"
        Resampling method used when ``auto_align`` triggers alignment; one of
        rasterio's resampling names (e.g. ``"nearest"``, ``"bilinear"``).
    name : str or None, default None
        Optional name for the output band. An index band maps to no input
        band, so it is never named automatically: it stays unnamed unless a
        name is given here or assigned later via ``band_names`` /
        ``set_band_name``.

    Returns
    -------
    EEORasterDataset
        Single-band float32 NDVI in ``[-1, 1]``. Pixels
        where ``NIR + Red == 0`` are set to 0. A pixel that is nodata in any
        input band is nodata (NaN) in the output; the output nodata value is
        NaN when any input band declares nodata, otherwise None.

    Raises
    ------
    AlignmentError
        If a dataset band is on a different grid and ``auto_align`` is False.
    IndexError
        If an int band index is outside the range of available bands.
    ValidationError
        If a band argument is not an ``EEORasterDataset``, an int index, or a
        band name, or if a band name is unknown or matches more than one band.

    Examples
    --------
    >>> ndvi = nir_ds.ndvi(red_ds)          # separate band rasters
    >>> ndvi = scene.ndvi(red=4, nir=8)     # bands of one Sentinel-2 stack
    """
    return _compute_index(
        ds,
        [nir, red],
        _normalized_difference,
        auto_align=auto_align,
        method=method,
        name=name,
    )


@eeo_raster_op(propagate_band_names=False)
def ndwi(
    ds: EEORasterDataset,
    nir: BandSpec,
    *,
    green: BandSpec = 1,
    auto_align: bool = True,
    method: str = "bilinear",
    name: str | None = None,
) -> EEORasterDataset:
    """Compute the Normalized Difference Water Index (McFeeters, 1996).

    ``NDWI = (Green - NIR) / (Green + NIR)``. Highlights open water, which
    takes positive values while vegetation and soil go negative; values fall
    in ``[-1, 1]``.

    Each band is either a separate ``EEORasterDataset``, a 1-based ``int`` band
    index into ``ds``, or the name of one of ``ds``'s bands. The Green band defaults
    to band ``1`` of ``ds``, so calling on a single-band Green raster and passing
    the NIR raster is enough.

    Parameters
    ----------
    ds : EEORasterDataset
        Receiver raster (the multi-band scene, or the Green band when ``green``
        defaults to band 1).
    nir : EEORasterDataset or int or str
        NIR band, as a separate raster, a 1-based band index into ``ds``, or
        one of ``ds``'s band names. Sentinel-2: B8; Landsat 8/9 (OLI): B5.
    green : EEORasterDataset or int or str, default 1
        Green band, as a separate raster, a 1-based band index into ``ds``, or
        one of ``ds``'s band names. Sentinel-2: B3; Landsat 8/9 (OLI): B3.
    auto_align : bool, default True
        If True, resample a dataset band onto ``ds``'s grid when their shape or
        transform differ. If False, a mismatch raises ``AlignmentError``.
    method : str, default "bilinear"
        Resampling method used when ``auto_align`` triggers alignment; one of
        rasterio's resampling names (e.g. ``"nearest"``, ``"bilinear"``).
    name : str or None, default None
        Optional name for the output band. An index band maps to no input
        band, so it is never named automatically: it stays unnamed unless a
        name is given here or assigned later via ``band_names`` /
        ``set_band_name``.

    Returns
    -------
    EEORasterDataset
        Single-band float32 NDWI in ``[-1, 1]``. Pixels
        where ``Green + NIR == 0`` are set to 0. A pixel that is nodata in any
        input band is nodata (NaN) in the output; the output nodata value is
        NaN when any input band declares nodata, otherwise None.

    Raises
    ------
    AlignmentError
        If a dataset band is on a different grid and ``auto_align`` is False.
    IndexError
        If an int band index is outside the range of available bands.
    ValidationError
        If a band argument is not an ``EEORasterDataset``, an int index, or a
        band name, or if a band name is unknown or matches more than one band.

    Notes
    -----
    This is McFeeters' water NDWI; the moisture variant is :meth:`ndmi`.

    Examples
    --------
    >>> ndwi = green_ds.ndwi(nir_ds)          # separate band rasters
    >>> ndwi = scene.ndwi(nir=8, green=3)     # bands of one Sentinel-2 stack
    """
    return _compute_index(
        ds,
        [green, nir],
        _normalized_difference,
        auto_align=auto_align,
        method=method,
        name=name,
    )


@eeo_raster_op(propagate_band_names=False)
def ndmi(
    ds: EEORasterDataset,
    swir: BandSpec,
    *,
    nir: BandSpec = 1,
    auto_align: bool = True,
    method: str = "bilinear",
    name: str | None = None,
) -> EEORasterDataset:
    """Compute the Normalized Difference Moisture Index.

    ``NDMI = (NIR - SWIR1) / (NIR + SWIR1)``. Tracks vegetation water content;
    higher values indicate wetter canopies. Values fall in ``[-1, 1]``.

    Each band is either a separate ``EEORasterDataset``, a 1-based ``int`` band
    index into ``ds``, or the name of one of ``ds``'s bands. The NIR band defaults
    to band ``1`` of ``ds``, so calling on a single-band NIR raster and passing the
    SWIR1 raster is enough.

    Parameters
    ----------
    ds : EEORasterDataset
        Receiver raster (the multi-band scene, or the NIR band when ``nir``
        defaults to band 1).
    swir : EEORasterDataset or int or str
        SWIR1 band, as a separate raster, a 1-based band index into ``ds``, or
        one of ``ds``'s band names. Sentinel-2: B11; Landsat 8/9 (OLI): B6.
    nir : EEORasterDataset or int or str, default 1
        NIR band, as a separate raster, a 1-based band index into ``ds``, or
        one of ``ds``'s band names. Sentinel-2: B8 (or B8A); Landsat 8/9 (OLI): B5.
    auto_align : bool, default True
        If True, resample a dataset band onto ``ds``'s grid when their shape or
        transform differ. If False, a mismatch raises ``AlignmentError``.
    method : str, default "bilinear"
        Resampling method used when ``auto_align`` triggers alignment; one of
        rasterio's resampling names (e.g. ``"nearest"``, ``"bilinear"``).
    name : str or None, default None
        Optional name for the output band. An index band maps to no input
        band, so it is never named automatically: it stays unnamed unless a
        name is given here or assigned later via ``band_names`` /
        ``set_band_name``.

    Returns
    -------
    EEORasterDataset
        Single-band float32 NDMI in ``[-1, 1]``. Pixels
        where ``NIR + SWIR1 == 0`` are set to 0. A pixel that is nodata in any
        input band is nodata (NaN) in the output; the output nodata value is
        NaN when any input band declares nodata, otherwise None.

    Raises
    ------
    AlignmentError
        If a dataset band is on a different grid and ``auto_align`` is False.
    IndexError
        If an int band index is outside the range of available bands.
    ValidationError
        If a band argument is not an ``EEORasterDataset``, an int index, or a
        band name, or if a band name is unknown or matches more than one band.

    Notes
    -----
    Sentinel-2's SWIR1 (B11) is 20 m; pass ``auto_align=True``
    (the default) to resample it onto a 10 m NIR grid.

    Examples
    --------
    >>> ndmi = nir_ds.ndmi(swir1_ds)          # separate band rasters
    >>> ndmi = scene.ndmi(swir=11, nir=8)     # bands of one Sentinel-2 stack
    """
    return _compute_index(
        ds,
        [nir, swir],
        _normalized_difference,
        auto_align=auto_align,
        method=method,
        name=name,
    )


@eeo_raster_op(propagate_band_names=False)
def ndbi(
    ds: EEORasterDataset,
    nir: BandSpec,
    *,
    swir: BandSpec = 1,
    auto_align: bool = True,
    method: str = "bilinear",
    name: str | None = None,
) -> EEORasterDataset:
    """Compute the Normalized Difference Built-up Index.

    ``NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)``. Highlights built-up and
    impervious surfaces, which take higher values than vegetation. Values fall
    in ``[-1, 1]``.

    Each band is either a separate ``EEORasterDataset``, a 1-based ``int`` band
    index into ``ds``, or the name of one of ``ds``'s bands. The SWIR1 band defaults
    to band ``1`` of ``ds``, so calling on a single-band SWIR1 raster and passing
    the NIR raster is enough.

    Parameters
    ----------
    ds : EEORasterDataset
        Receiver raster (the multi-band scene, or the SWIR1 band when ``swir``
        defaults to band 1).
    nir : EEORasterDataset or int or str
        NIR band, as a separate raster, a 1-based band index into ``ds``, or
        one of ``ds``'s band names. Sentinel-2: B8; Landsat 8/9 (OLI): B5.
    swir : EEORasterDataset or int or str, default 1
        SWIR1 band, as a separate raster, a 1-based band index into ``ds``, or
        one of ``ds``'s band names. Sentinel-2: B11; Landsat 8/9 (OLI): B6.
    auto_align : bool, default True
        If True, resample a dataset band onto ``ds``'s grid when their shape or
        transform differ. If False, a mismatch raises ``AlignmentError``.
    method : str, default "bilinear"
        Resampling method used when ``auto_align`` triggers alignment; one of
        rasterio's resampling names (e.g. ``"nearest"``, ``"bilinear"``).
    name : str or None, default None
        Optional name for the output band. An index band maps to no input
        band, so it is never named automatically: it stays unnamed unless a
        name is given here or assigned later via ``band_names`` /
        ``set_band_name``.

    Returns
    -------
    EEORasterDataset
        Single-band float32 NDBI in ``[-1, 1]``. Pixels
        where ``SWIR1 + NIR == 0`` are set to 0. A pixel that is nodata in any
        input band is nodata (NaN) in the output; the output nodata value is
        NaN when any input band declares nodata, otherwise None.

    Raises
    ------
    AlignmentError
        If a dataset band is on a different grid and ``auto_align`` is False.
    IndexError
        If an int band index is outside the range of available bands.
    ValidationError
        If a band argument is not an ``EEORasterDataset``, an int index, or a
        band name, or if a band name is unknown or matches more than one band.

    Notes
    -----
    Sentinel-2's SWIR1 (B11) is 20 m; pass ``auto_align=True``
    (the default) to resample it onto a 10 m NIR grid.

    Examples
    --------
    >>> ndbi = swir1_ds.ndbi(nir_ds)          # separate band rasters
    >>> ndbi = scene.ndbi(nir=8, swir=11)     # bands of one Sentinel-2 stack
    """
    return _compute_index(
        ds,
        [swir, nir],
        _normalized_difference,
        auto_align=auto_align,
        method=method,
        name=name,
    )


@eeo_raster_op(propagate_band_names=False)
def evi(
    ds: EEORasterDataset,
    red: BandSpec,
    blue: BandSpec,
    *,
    nir: BandSpec = 1,
    auto_align: bool = True,
    method: str = "bilinear",
    name: str | None = None,
) -> EEORasterDataset:
    """Compute the Enhanced Vegetation Index.

    ``EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)`` using the
    standard MODIS/Sentinel-2 coefficients (gain 2.5; aerosol terms 6 and 7.5;
    canopy background 1). EVI reduces the atmospheric and soil-background
    effects that saturate NDVI over dense canopies.

    EVI assumes surface reflectance scaled to roughly ``[0, 1]``; if your bands
    are integer DN or reflectance scaled by 10000, divide by the scale factor
    first (e.g. ``scene.divide(10000).evi(...)``).

    Each band is either a separate ``EEORasterDataset``, a 1-based ``int`` band
    index into ``ds``, or the name of one of ``ds``'s bands. The NIR band defaults
    to band ``1`` of ``ds``.

    Parameters
    ----------
    ds : EEORasterDataset
        Receiver raster (the multi-band scene, or the NIR band when ``nir``
        defaults to band 1).
    red : EEORasterDataset or int or str
        Red band, as a separate raster, a 1-based band index into ``ds``, or
        one of ``ds``'s band names. Sentinel-2: B4; Landsat 8/9 (OLI): B4.
    blue : EEORasterDataset or int or str
        Blue band, as a separate raster, a 1-based band index into ``ds``, or
        one of ``ds``'s band names. Sentinel-2: B2; Landsat 8/9 (OLI): B2.
    nir : EEORasterDataset or int or str, default 1
        NIR band, as a separate raster, a 1-based band index into ``ds``, or
        one of ``ds``'s band names. Sentinel-2: B8; Landsat 8/9 (OLI): B5.
    auto_align : bool, default True
        If True, resample a dataset band onto ``ds``'s grid when their shape or
        transform differ. If False, a mismatch raises ``AlignmentError``.
    method : str, default "bilinear"
        Resampling method used when ``auto_align`` triggers alignment; one of
        rasterio's resampling names (e.g. ``"nearest"``, ``"bilinear"``).
    name : str or None, default None
        Optional name for the output band. An index band maps to no input
        band, so it is never named automatically: it stays unnamed unless a
        name is given here or assigned later via ``band_names`` /
        ``set_band_name``.

    Returns
    -------
    EEORasterDataset
        Single-band float32 EVI (typically ``[-1, 1]`` for valid reflectance).
        Pixels where the denominator is 0 are set
        to 0. A pixel that is nodata in any input band is nodata (NaN) in the
        output; the output nodata value is NaN when any input band declares
        nodata, otherwise None.

    Raises
    ------
    AlignmentError
        If a dataset band is on a different grid and ``auto_align`` is False.
    IndexError
        If an int band index is outside the range of available bands.
    ValidationError
        If a band argument is not an ``EEORasterDataset``, an int index, or a
        band name, or if a band name is unknown or matches more than one band.

    Examples
    --------
    >>> evi = nir_ds.evi(red_ds, blue_ds)             # separate band rasters
    >>> evi = scene.evi(red=4, blue=2, nir=8)         # bands of one S2 stack
    """
    return _compute_index(
        ds,
        [nir, red, blue],
        _evi_formula,
        auto_align=auto_align,
        method=method,
        name=name,
    )


def _evi_formula(bands):
    """Standard-coefficient EVI over ``[nir, red, blue]`` float32 bands."""
    nir, red, blue = bands
    return _safe_ratio(2.5 * (nir - red), nir + 6.0 * red - 7.5 * blue + 1.0)


@eeo_raster_op(propagate_band_names=False)
def savi(
    ds: EEORasterDataset,
    red: BandSpec,
    *,
    nir: BandSpec = 1,
    soil_factor: float = 0.5,
    auto_align: bool = True,
    method: str = "bilinear",
    name: str | None = None,
) -> EEORasterDataset:
    """Compute the Soil-Adjusted Vegetation Index.

    ``SAVI = (1 + L) * (NIR - Red) / (NIR + Red + L)`` where ``L`` is the soil
    brightness correction (``soil_factor``). SAVI dampens soil-background
    influence in sparsely vegetated areas; with ``L = 0`` it reduces to NDVI.

    Each band is either a separate ``EEORasterDataset``, a 1-based ``int`` band
    index into ``ds``, or the name of one of ``ds``'s bands. The NIR band defaults
    to band ``1`` of ``ds``.

    Parameters
    ----------
    ds : EEORasterDataset
        Receiver raster (the multi-band scene, or the NIR band when ``nir``
        defaults to band 1).
    red : EEORasterDataset or int or str
        Red band, as a separate raster, a 1-based band index into ``ds``, or
        one of ``ds``'s band names. Sentinel-2: B4; Landsat 8/9 (OLI): B4.
    nir : EEORasterDataset or int or str, default 1
        NIR band, as a separate raster, a 1-based band index into ``ds``, or
        one of ``ds``'s band names. Sentinel-2: B8; Landsat 8/9 (OLI): B5.
    soil_factor : float, default 0.5
        Soil brightness correction ``L`` in ``[0, 1]``: 0 for dense
        vegetation (equivalent to NDVI), 1 for very sparse cover; 0.5 is the
        common default for intermediate cover.
    auto_align : bool, default True
        If True, resample a dataset band onto ``ds``'s grid when their shape or
        transform differ. If False, a mismatch raises ``AlignmentError``.
    method : str, default "bilinear"
        Resampling method used when ``auto_align`` triggers alignment; one of
        rasterio's resampling names (e.g. ``"nearest"``, ``"bilinear"``).
    name : str or None, default None
        Optional name for the output band. An index band maps to no input
        band, so it is never named automatically: it stays unnamed unless a
        name is given here or assigned later via ``band_names`` /
        ``set_band_name``.

    Returns
    -------
    EEORasterDataset
        Single-band float32 SAVI in ``[-1, 1]``. Pixels
        where ``NIR + Red + L == 0`` are set to 0. A pixel that is nodata in
        any input band is nodata (NaN) in the output; the output nodata value
        is NaN when any input band declares nodata, otherwise None.

    Raises
    ------
    AlignmentError
        If a dataset band is on a different grid and ``auto_align`` is False.
    IndexError
        If an int band index is outside the range of available bands.
    ValidationError
        If a band argument is not an ``EEORasterDataset``, an int index, or a
        band name, or if a band name is unknown or matches more than one band.

    Examples
    --------
    >>> savi = nir_ds.savi(red_ds, soil_factor=0.5)   # separate band rasters
    >>> savi = scene.savi(red=4, nir=8)               # bands of one S2 stack
    """
    return _compute_index(
        ds,
        [nir, red],
        lambda bands: _savi_formula(bands, soil_factor),
        auto_align=auto_align,
        method=method,
        name=name,
    )


def _savi_formula(bands, soil_factor):
    """Soil-adjusted vegetation index over ``[nir, red]`` float32 bands."""
    nir, red = bands
    return _safe_ratio((1.0 + soil_factor) * (nir - red), nir + red + soil_factor)
