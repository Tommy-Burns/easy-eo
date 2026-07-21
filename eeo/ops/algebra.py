"""Pixel-wise raster algebra operations."""

import numpy as np
import rasterio as rio

from eeo.common import align_raster_to_target
from eeo.core.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_op
from eeo.core.exceptions import AlignmentError


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
        New dataset in the same dtype as ``ds``. Integer inputs are not
        promoted, so fractional results are truncated; cast to a floating
        dtype first if that matters. The nodata value is carried over
        unchanged.

    Raises
    ------
    AlignmentError
        If ``other`` is a dataset on a different grid and ``auto_align`` is
        False.

    Notes
    -----
    Reads the full array(s) into memory rather than streaming block-wise.
    Nodata pixels are not masked: they take part in the arithmetic and are
    corrupted in the output, although the nodata value in the metadata is
    preserved.

    Examples
    --------
    >>> ds = load_array(np.random.rand(64, 64), crs=4326)
    >>> brighter = ds.add(0.1)
    """
    if isinstance(other, EEORasterDataset):
        if ds.get_shape() != other.get_shape() or ds.get_transform() != other.get_transform():
            if auto_align:
                other = align_raster_to_target(other, ds, method=method)
            else:
                raise AlignmentError(
                    "rasters must share the same grid for arithmetic; "
                    f"got shape {other.get_shape()} vs {ds.get_shape()}. "
                    "Pass auto_align=True to resample the other raster onto this grid."
                )
        data = ds.read() + other.read()
    else:
        data = ds.read() + other

    meta = ds.get_metadata()
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(data)
    return EEORasterDataset.from_rasterio(out_ds)


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
        New dataset in the same dtype as ``ds``. Integer inputs are not
        promoted, so fractional or negative results may wrap or truncate;
        cast to a floating dtype first if that matters. The nodata value is
        carried over unchanged.

    Raises
    ------
    AlignmentError
        If ``other`` is a dataset on a different grid and ``auto_align`` is
        False.

    Notes
    -----
    Reads the full array(s) into memory rather than streaming block-wise.
    Nodata pixels are not masked: they take part in the arithmetic and are
    corrupted in the output, although the nodata value in the metadata is
    preserved.

    Examples
    --------
    >>> change = ds_after.subtract(ds_before)
    """
    if isinstance(other, EEORasterDataset):
        if ds.get_shape() != other.get_shape() or ds.get_transform() != other.get_transform():
            if auto_align:
                other = align_raster_to_target(other, ds, method=method)
            else:
                raise AlignmentError(
                    "rasters must share the same grid for arithmetic; "
                    f"got shape {other.get_shape()} vs {ds.get_shape()}. "
                    "Pass auto_align=True to resample the other raster onto this grid."
                )
        data = ds.read() - other.read()
    else:
        data = ds.read() - other
    meta = ds.get_metadata()
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(data)
    return EEORasterDataset.from_rasterio(out_ds)


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
        New dataset in the same dtype as ``ds``. Integer inputs are not
        promoted, so fractional results are truncated; cast to a floating
        dtype first if that matters. The nodata value is carried over
        unchanged.

    Raises
    ------
    AlignmentError
        If ``other`` is a dataset on a different grid and ``auto_align`` is
        False.

    Notes
    -----
    Reads the full array(s) into memory rather than streaming block-wise.
    Nodata pixels are not masked: they take part in the arithmetic and are
    corrupted in the output, although the nodata value in the metadata is
    preserved.

    Examples
    --------
    >>> scaled = ds.multiply(100)
    """
    if isinstance(other, EEORasterDataset):
        if ds.get_shape() != other.get_shape() or ds.get_transform() != other.get_transform():
            if auto_align:
                other = align_raster_to_target(other, ds, method=method)
            else:
                raise AlignmentError(
                    "rasters must share the same grid for arithmetic; "
                    f"got shape {other.get_shape()} vs {ds.get_shape()}. "
                    "Pass auto_align=True to resample the other raster onto this grid."
                )

        data = ds.read() * other.read()
    else:
        data = ds.read() * other
    meta = ds.get_metadata()
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(data)
    return EEORasterDataset.from_rasterio(out_ds)


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
        New dataset. The quotient is computed in float32 but written back in
        the same dtype as ``ds``, so integer inputs truncate the result;
        divide a floating-dtype raster to keep fractional values. The nodata
        value is carried over unchanged.

    Raises
    ------
    AlignmentError
        If ``other`` is a dataset on a different grid and ``auto_align`` is
        False.

    Notes
    -----
    Reads the full array(s) into memory rather than streaming block-wise.
    Nodata pixels are not masked: they take part in the division and are
    corrupted in the output, although the nodata value in the metadata is
    preserved.

    Examples
    --------
    >>> ratio = ds_nir.divide(ds_red)
    >>> halved = ds.divide(2)
    """
    src_data = ds.read()

    # Resolve other operand
    if isinstance(other, EEORasterDataset):
        if ds.get_shape() != other.get_shape() or ds.get_transform() != other.get_transform():
            if auto_align:
                other = align_raster_to_target(other, ds, method=method)
            else:
                raise AlignmentError(
                    "rasters must share the same grid for arithmetic; "
                    f"got shape {other.get_shape()} vs {ds.get_shape()}. "
                    "Pass auto_align=True to resample the other raster onto this grid."
                )
        other_data: np.ndarray | float | int = other.read()
    else:
        other_data = other

    # ---- SAFE DIVIDE ----
    data: np.ndarray
    if safe:
        if np.isscalar(other_data):
            if other_data == 0:
                data = np.zeros_like(src_data, dtype=np.float32)
            else:
                data = src_data / other_data
        else:
            data = np.divide(
                src_data,
                other_data,
                out=np.zeros_like(src_data, dtype=np.float32),
                where=other_data != 0,
            )
    else:
        data = src_data / other_data

    # Write output
    meta = ds.get_metadata()
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(data)

    return EEORasterDataset.from_rasterio(out_ds)


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
        New dataset in the same dtype as ``ds`` (fractional results are
        truncated for integer inputs). The nodata value is carried over
        unchanged.

    Notes
    -----
    Follows NumPy's ``**`` semantics; a negative pixel raised to a
    non-integer exponent yields ``nan``. Reads the full array into memory
    rather than streaming block-wise. Nodata pixels are not masked and are
    corrupted in the output.

    Examples
    --------
    >>> squared = ds.power(2)
    """
    data = ds.read() ** exponent
    meta = ds.get_metadata()
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(data)
    return EEORasterDataset.from_rasterio(out_ds)


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
        New dataset in the same dtype as ``ds``; the root is truncated for
        integer inputs, so use a floating dtype to keep fractional values.
        The nodata value is carried over unchanged.

    Notes
    -----
    Reads the full array into memory rather than streaming block-wise. Nodata
    pixels are not masked; a negative nodata sentinel is clamped to 0 and thus
    corrupted in the output.

    Examples
    --------
    >>> rooted = ds.sqrt()
    """
    data = np.sqrt(np.maximum(ds.read(), 0))
    meta = ds.get_metadata()
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(data)
    return EEORasterDataset.from_rasterio(out_ds)


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
        New dataset in the same dtype as ``ds``; the result is truncated for
        integer inputs, so use a floating dtype to keep fractional values.
        The nodata value is carried over unchanged.

    Notes
    -----
    Reads the full array into memory rather than streaming block-wise. Nodata
    pixels are not masked; because inputs are clamped to ``1e-10``, a nodata
    sentinel is corrupted in the output.

    Examples
    --------
    >>> natural = ds.log()
    >>> base10 = ds.log(base=10)
    """
    data = np.log(np.maximum(ds.read(), 1e-10)) / np.log(base)
    meta = ds.get_metadata()
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(data)
    return EEORasterDataset.from_rasterio(out_ds)


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
        New dataset in the same dtype as ``ds``. The nodata value is carried
        over unchanged.

    Notes
    -----
    Reads the full array into memory rather than streaming block-wise. Nodata
    pixels are not masked; a negative nodata sentinel becomes its magnitude
    and is thus corrupted in the output.

    Examples
    --------
    >>> magnitude = ds.absolute()
    """
    data = np.abs(ds.read())
    meta = ds.get_metadata()
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(data)
    return EEORasterDataset.from_rasterio(out_ds)
