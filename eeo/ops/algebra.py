"""Pixel-wise raster algebra operations."""

import numpy as np
import rasterio as rio

from eeo.common import align_raster_to_target, apply_nodata_contract, get_nodata
from eeo.core.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_op
from eeo.core.exceptions import AlignmentError

_ALIGN_MISMATCH = (
    "rasters must share the same grid for arithmetic; "
    "got shape {other} vs {ds}. "
    "Pass auto_align=True to resample the other raster onto this grid."
)


def _write_result(ds: EEORasterDataset, data, nodata) -> EEORasterDataset:
    """Write ``data`` into a new in-memory raster sharing ``ds``'s georeferencing.

    The output dtype and nodata value are taken from ``data`` and ``nodata``
    so the result records the dtype and nodata the operation actually produced.
    """
    meta = ds.get_metadata()
    meta.update(dtype=data.dtype, nodata=nodata)
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(data)
    return EEORasterDataset.from_rasterio(out_ds)


def _resolve_operand(ds, other, *, auto_align, method):
    """Return ``(other_array, other_nodata)`` for a raster or scalar operand.

    Aligns a raster operand onto ``ds``'s grid when needed; a scalar operand
    is returned unchanged with a None nodata (scalars carry no nodata).
    """
    if isinstance(other, EEORasterDataset):
        if ds.get_shape() != other.get_shape() or ds.get_transform() != other.get_transform():
            if auto_align:
                other = align_raster_to_target(other, ds, method=method)
            else:
                raise AlignmentError(
                    _ALIGN_MISMATCH.format(other=other.get_shape(), ds=ds.get_shape())
                )
        return other.read(), get_nodata(other)
    return other, None


# ARITHMETIC AND ALGEBRA
@eeo_raster_op
def add(
    ds: EEORasterDataset,
    other: EEORasterDataset | float | int,
    *,
    auto_align: bool = True,
    method: str = "bilinear",
) -> EEORasterDataset:
    """Add a raster or scalar to this raster, pixel by pixel.

    When ``other`` is a dataset whose grid differs from ``ds`` and
    ``auto_align`` is True, ``other`` is resampled onto ``ds``'s grid before
    the operation; otherwise a grid mismatch is an error.

    Parameters
    ----------
    ds : EEORasterDataset
        Left operand.
    other : EEORasterDataset or float or int
        Right operand. A dataset is added band-by-band; a scalar is added to
        every pixel.
    auto_align : bool, default True
        If True, resample ``other`` onto ``ds``'s grid when their shape or
        transform differ. If False, a mismatch raises ``AlignmentError``.
    method : str, default "bilinear"
        Resampling method used when ``auto_align`` triggers alignment; one of
        rasterio's resampling names (e.g. ``"nearest"``, ``"bilinear"``).

    Returns
    -------
    EEORasterDataset
        New dataset whose dtype follows NumPy type promotion of the operands,
        with floating results emitted as float32 (so ``uint16 + 0.5`` becomes
        float32 rather than truncating, while integer-only arithmetic stays
        integer). A pixel that is nodata in either operand is nodata in the
        output — NaN for floating outputs, the input's integer sentinel for
        integer outputs.

    Raises
    ------
    AlignmentError
        If ``other`` is a dataset on a different grid and ``auto_align`` is
        False.

    Notes
    -----
    Reads the full array(s) into memory rather than streaming block-wise.

    Examples
    --------
    >>> ds = load_array(np.random.rand(64, 64), crs=4326)
    >>> brighter = ds.add(0.1)
    """
    ds_nodata = get_nodata(ds)
    other_data, other_nodata = _resolve_operand(ds, other, auto_align=auto_align, method=method)
    src = ds.read()
    result = src + other_data

    operands = [(src, ds_nodata)]
    if isinstance(other, EEORasterDataset):
        operands.append((other_data, other_nodata))

    data, out_nodata = apply_nodata_contract(
        result, operands, fractional=False, ds_nodata=ds_nodata
    )
    return _write_result(ds, data, out_nodata)


@eeo_raster_op
def subtract(
    ds: EEORasterDataset,
    other: EEORasterDataset | float | int,
    *,
    auto_align: bool = True,
    method: str = "bilinear",
) -> EEORasterDataset:
    """Subtract a raster or scalar from this raster, pixel by pixel.

    Computes ``ds - other``. When ``other`` is a dataset whose grid differs
    from ``ds`` and ``auto_align`` is True, ``other`` is resampled onto
    ``ds``'s grid first; otherwise a grid mismatch is an error.

    Parameters
    ----------
    ds : EEORasterDataset
        Left operand (the minuend).
    other : EEORasterDataset or float or int
        Right operand (the subtrahend). A dataset is subtracted band-by-band;
        a scalar is subtracted from every pixel.
    auto_align : bool, default True
        If True, resample ``other`` onto ``ds``'s grid when their shape or
        transform differ. If False, a mismatch raises ``AlignmentError``.
    method : str, default "bilinear"
        Resampling method used when ``auto_align`` triggers alignment; one of
        rasterio's resampling names (e.g. ``"nearest"``, ``"bilinear"``).

    Returns
    -------
    EEORasterDataset
        New dataset whose dtype follows NumPy type promotion of the operands,
        with floating results emitted as float32. Integer-only arithmetic
        stays integer (an unsigned difference can still wrap; cast to a signed
        or floating dtype first if negatives matter). A pixel that is nodata in
        either operand is nodata in the output — NaN for floating outputs, the
        input's integer sentinel for integer outputs.

    Raises
    ------
    AlignmentError
        If ``other`` is a dataset on a different grid and ``auto_align`` is
        False.

    Notes
    -----
    Reads the full array(s) into memory rather than streaming block-wise.

    Examples
    --------
    >>> change = ds_after.subtract(ds_before)
    """
    ds_nodata = get_nodata(ds)
    other_data, other_nodata = _resolve_operand(ds, other, auto_align=auto_align, method=method)
    src = ds.read()
    result = src - other_data

    operands = [(src, ds_nodata)]
    if isinstance(other, EEORasterDataset):
        operands.append((other_data, other_nodata))

    data, out_nodata = apply_nodata_contract(
        result, operands, fractional=False, ds_nodata=ds_nodata
    )
    return _write_result(ds, data, out_nodata)


@eeo_raster_op
def multiply(
    ds: EEORasterDataset,
    other: EEORasterDataset | float | int,
    *,
    auto_align: bool = True,
    method: str = "bilinear",
) -> EEORasterDataset:
    """Multiply this raster by a raster or scalar, pixel by pixel.

    When ``other`` is a dataset whose grid differs from ``ds`` and
    ``auto_align`` is True, ``other`` is resampled onto ``ds``'s grid first;
    otherwise a grid mismatch is an error.

    Parameters
    ----------
    ds : EEORasterDataset
        Left operand.
    other : EEORasterDataset or float or int
        Right operand. A dataset is multiplied band-by-band; a scalar scales
        every pixel.
    auto_align : bool, default True
        If True, resample ``other`` onto ``ds``'s grid when their shape or
        transform differ. If False, a mismatch raises ``AlignmentError``.
    method : str, default "bilinear"
        Resampling method used when ``auto_align`` triggers alignment; one of
        rasterio's resampling names (e.g. ``"nearest"``, ``"bilinear"``).

    Returns
    -------
    EEORasterDataset
        New dataset whose dtype follows NumPy type promotion of the operands,
        with floating results emitted as float32 (so ``uint16 * 0.5`` becomes
        float32 rather than truncating, while integer-only arithmetic stays
        integer). A pixel that is nodata in either operand is nodata in the
        output — NaN for floating outputs, the input's integer sentinel for
        integer outputs.

    Raises
    ------
    AlignmentError
        If ``other`` is a dataset on a different grid and ``auto_align`` is
        False.

    Notes
    -----
    Reads the full array(s) into memory rather than streaming block-wise.

    Examples
    --------
    >>> scaled = ds.multiply(100)
    """
    ds_nodata = get_nodata(ds)
    other_data, other_nodata = _resolve_operand(ds, other, auto_align=auto_align, method=method)
    src = ds.read()
    result = src * other_data

    operands = [(src, ds_nodata)]
    if isinstance(other, EEORasterDataset):
        operands.append((other_data, other_nodata))

    data, out_nodata = apply_nodata_contract(
        result, operands, fractional=False, ds_nodata=ds_nodata
    )
    return _write_result(ds, data, out_nodata)


@eeo_raster_op
def divide(
    ds: EEORasterDataset,
    other: EEORasterDataset | float | int,
    *,
    auto_align: bool = True,
    method: str = "bilinear",
    safe: bool = True,
) -> EEORasterDataset:
    """Divide this raster by a raster or scalar, pixel by pixel.

    Computes ``ds / other``. When ``other`` is a dataset whose grid differs
    from ``ds`` and ``auto_align`` is True, ``other`` is resampled onto
    ``ds``'s grid first; otherwise a grid mismatch is an error.

    Parameters
    ----------
    ds : EEORasterDataset
        Numerator.
    other : EEORasterDataset or float or int
        Denominator. A dataset divides band-by-band; a scalar divides every
        pixel.
    auto_align : bool, default True
        If True, resample ``other`` onto ``ds``'s grid when their shape or
        transform differ. If False, a mismatch raises ``AlignmentError``.
    method : str, default "bilinear"
        Resampling method used when ``auto_align`` triggers alignment; one of
        rasterio's resampling names (e.g. ``"nearest"``, ``"bilinear"``).
    safe : bool, default True
        If True, pixels where the denominator is zero are set to 0 instead of
        producing ``inf``/``nan``. If False, division follows NumPy semantics
        (zero denominators yield ``inf``/``nan`` and emit a warning).

    Returns
    -------
    EEORasterDataset
        New dataset in float32 (division is a fractional-result op, so the
        quotient is never truncated to an integer dtype). A pixel that is
        nodata in either operand is nodata (NaN) in the output.

    Raises
    ------
    AlignmentError
        If ``other`` is a dataset on a different grid and ``auto_align`` is
        False.

    Notes
    -----
    Reads the full array(s) into memory rather than streaming block-wise.

    Examples
    --------
    >>> ratio = ds_nir.divide(ds_red)
    >>> halved = ds.divide(2)
    """
    ds_nodata = get_nodata(ds)
    src = ds.read()
    other_data, other_nodata = _resolve_operand(ds, other, auto_align=auto_align, method=method)

    # ---- SAFE DIVIDE ----
    if safe:
        if np.isscalar(other_data):
            result = np.zeros_like(src, dtype=np.float32) if other_data == 0 else src / other_data
        else:
            # np.where instead of the in-place out=/where= ufunc form so the
            # expression stays dispatchable to lazy array backends.
            with np.errstate(divide="ignore", invalid="ignore"):
                quotient = np.divide(src, other_data)
            result = np.where(other_data != 0, quotient, np.float32(0))
    else:
        result = src / other_data

    operands = [(src, ds_nodata)]
    if isinstance(other, EEORasterDataset):
        operands.append((other_data, other_nodata))

    data, out_nodata = apply_nodata_contract(result, operands, fractional=True, ds_nodata=ds_nodata)
    return _write_result(ds, data, out_nodata)


@eeo_raster_op
def power(ds: EEORasterDataset, exponent: int | float) -> EEORasterDataset:
    """Raise each pixel to a scalar power.

    Parameters
    ----------
    ds : EEORasterDataset
        Input raster dataset.
    exponent : int or float
        Scalar exponent applied to every pixel.

    Returns
    -------
    EEORasterDataset
        New dataset whose dtype follows NumPy type promotion of the base and
        exponent, with floating results emitted as float32 (so an integer
        raster raised to a fractional exponent becomes float32 rather than
        truncating). Nodata pixels are nodata in the output — NaN for floating
        outputs, the input's integer sentinel for integer outputs.

    Notes
    -----
    Follows NumPy's ``**`` semantics; a negative pixel raised to a
    non-integer exponent yields ``nan`` where it is not masked as nodata.
    Reads the full array into memory rather than streaming block-wise.

    Examples
    --------
    >>> squared = ds.power(2)
    """
    ds_nodata = get_nodata(ds)
    src = ds.read()
    with np.errstate(invalid="ignore", divide="ignore"):
        result = src**exponent

    data, out_nodata = apply_nodata_contract(
        result, [(src, ds_nodata)], fractional=False, ds_nodata=ds_nodata
    )
    return _write_result(ds, data, out_nodata)


# TRANSFORMATIONS
@eeo_raster_op
def sqrt(ds: EEORasterDataset) -> EEORasterDataset:
    """Take the pixel-wise square root.

    Negative pixels are clamped to 0 before the root, so the result never
    contains ``nan`` from negative inputs.

    Parameters
    ----------
    ds : EEORasterDataset
        Input raster dataset.

    Returns
    -------
    EEORasterDataset
        New dataset in float32 (a fractional-result op, so the root is never
        truncated to an integer dtype). Nodata pixels are nodata (NaN) in the
        output.

    Notes
    -----
    Reads the full array into memory rather than streaming block-wise.

    Examples
    --------
    >>> rooted = ds.sqrt()
    """
    ds_nodata = get_nodata(ds)
    src = ds.read()
    result = np.sqrt(np.maximum(src, 0))

    data, out_nodata = apply_nodata_contract(
        result, [(src, ds_nodata)], fractional=True, ds_nodata=ds_nodata
    )
    return _write_result(ds, data, out_nodata)


@eeo_raster_op
def log(ds: EEORasterDataset, base: int | float = np.e) -> EEORasterDataset:
    """Take the pixel-wise logarithm.

    Pixels are clamped to a minimum of ``1e-10`` before the logarithm, so
    zero and negative inputs do not produce ``-inf``/``nan``.

    Parameters
    ----------
    ds : EEORasterDataset
        Input raster dataset.
    base : int or float, default ``numpy.e``
        Logarithm base. Defaults to the natural logarithm.

    Returns
    -------
    EEORasterDataset
        New dataset in float32 (a fractional-result op, so the result is never
        truncated to an integer dtype). Nodata pixels are nodata (NaN) in the
        output.

    Notes
    -----
    Reads the full array into memory rather than streaming block-wise.

    Examples
    --------
    >>> natural = ds.log()
    >>> base10 = ds.log(base=10)
    """
    ds_nodata = get_nodata(ds)
    src = ds.read()
    result = np.log(np.maximum(src, 1e-10)) / np.log(base)

    data, out_nodata = apply_nodata_contract(
        result, [(src, ds_nodata)], fractional=True, ds_nodata=ds_nodata
    )
    return _write_result(ds, data, out_nodata)


@eeo_raster_op
def absolute(ds: EEORasterDataset) -> EEORasterDataset:
    """Take the pixel-wise absolute value.

    Parameters
    ----------
    ds : EEORasterDataset
        Input raster dataset.

    Returns
    -------
    EEORasterDataset
        New dataset in the same dtype NumPy's ``abs`` produces (float64
        narrowed to float32). Nodata pixels are nodata in the output — NaN for
        floating outputs, the input's integer sentinel for integer outputs.

    Notes
    -----
    Reads the full array into memory rather than streaming block-wise. Because
    nodata pixels are masked, a negative nodata sentinel is not turned into its
    magnitude in the output.

    Examples
    --------
    >>> magnitude = ds.absolute()
    """
    ds_nodata = get_nodata(ds)
    src = ds.read()
    result = np.abs(src)

    data, out_nodata = apply_nodata_contract(
        result, [(src, ds_nodata)], fractional=False, ds_nodata=ds_nodata
    )
    return _write_result(ds, data, out_nodata)
