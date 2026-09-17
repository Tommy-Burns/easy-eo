"""Shared helpers used across operations (alignment, resampling)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from rasterio.enums import Resampling

if TYPE_CHECKING:
    # Type-hints only: eeo.core's package init (via load_ops()) imports
    # eeo.ops/eeo.analysis/etc, which import from this module - a real
    # runtime import here would be circular. Neither function below needs
    # EEORasterDataset at runtime; both just call duck-typed methods on it.
    from eeo.core.core import EEORasterDataset


def is_rasterio_backed(ds: EEORasterDataset) -> bool:
    """Return True if ``ds`` is backed by the rasterio adapter.

    Detection is based on the adapter type, not the class of the backend
    object. A rasterio-backed dataset's ``backend`` is usually a
    ``rasterio.io.DatasetReader`` — from a file, or an in-memory result, which
    the library reopens read-only — but a ``rasterio.io.DatasetWriter`` wrapped
    with ``EEORasterDataset.from_rasterio`` is equally valid. Checking
    ``isinstance(backend, DatasetReader)`` would wrongly reject that case.

    Parameters
    ----------
    ds : EEORasterDataset
        Dataset to inspect.

    Returns
    -------
    bool
        True if ``ds`` uses the rasterio adapter, False otherwise.
    """
    from eeo.core.adapters import RasterioAdapter

    return isinstance(ds._adapter, RasterioAdapter)


def require_rasterio(ds: EEORasterDataset, operation: str) -> EEORasterDataset:
    """Return ``ds`` on the rasterio backend, for an op that needs one.

    A lazy dataset is promoted, which for a file-backed one reopens the file
    and reads nothing. A NumPy-backed one is refused rather than promoted
    silently: its pixels are already in memory and promoting copies all of
    them, so that is the caller's decision to make.

    Parameters
    ----------
    ds : EEORasterDataset
        Dataset the operation was called on.
    operation : str
        Name of the operation, used in the error message.

    Returns
    -------
    EEORasterDataset
        ``ds`` itself when it is rasterio-backed, else a promoted copy.

    Raises
    ------
    BackendError
        If ``ds`` uses the NumPy backend.
    """
    from eeo.core.exceptions import BackendError

    if is_rasterio_backed(ds):
        return ds
    if _is_promotable_without_reading(ds):
        return ds.to_rasterio()
    raise BackendError(
        f"{operation} requires a rasterio-backed dataset; this dataset uses the "
        "NumPy backend. Call .to_rasterio() first."
    )


def promote_for_decimated_read(ds: EEORasterDataset) -> EEORasterDataset:
    """Return a dataset that can serve a decimated (``out_shape``) read.

    Only the rasterio backend reads decimated. A lazy dataset reaches it by
    reopening its file, which is what keeps plotting a large lazy raster from
    reading it whole. Anything else is returned unchanged, for the caller to
    fall back on a full read.

    Parameters
    ----------
    ds : EEORasterDataset
        Dataset about to be read for display.

    Returns
    -------
    EEORasterDataset
        ``ds``, or a rasterio-backed promotion of it that cost no read.
    """
    if not is_rasterio_backed(ds) and _is_promotable_without_reading(ds):
        return ds.to_rasterio()
    return ds


def _is_promotable_without_reading(ds: EEORasterDataset) -> bool:
    """Report whether ``ds`` can reach the rasterio backend without a read."""
    from eeo.core.adapters import XarrayAdapter

    return isinstance(ds._adapter, XarrayAdapter) and ds._adapter.source_path is not None


def normalize_resampling_method(value):
    """Normalize a resampling method to a ``rasterio.enums.Resampling`` value."""
    from eeo.core.exceptions import ValidationError

    if isinstance(value, Resampling):
        return value
    if isinstance(value, str):
        name = value.lower().strip()
        try:
            return Resampling[name]
        except KeyError as e:
            valid = ", ".join([r.name for r in Resampling])
            raise ValidationError(
                f"invalid resampling method {value!r}; expected one of: {valid}"
            ) from e
    raise ValidationError(
        f"resampling method must be a str or rasterio.enums.Resampling; got {type(value).__name__}"
    )


# Helper function for raster auto-alignment
def align_raster_to_target(
    ds: EEORasterDataset, target: EEORasterDataset, method: str = "bilinear"
) -> EEORasterDataset:
    """Resample a dataset to match a target raster's shape and transform."""
    if ds.get_shape() == target.get_shape() or ds.get_transform() == target.get_transform():
        return ds
    return ds.resample(size=target.get_shape(), resampling_method=method)


# helper to mask nodata values from an EEORasterDataset
def mask_nodata(ds: EEORasterDataset, array: np.ndarray) -> np.ndarray:
    """Replace ``ds``'s nodata pixels in ``array`` with NaN."""
    nodata = ds.get_metadata().get("nodata", None)
    if nodata is not None:
        array = np.where(array == nodata, np.nan, array)
    return array


def get_nodata(ds: EEORasterDataset):
    """Return ``ds``'s declared nodata value, or None if it declares none."""
    return ds.get_metadata().get("nodata", None)


def resolve_band_index(ds: EEORasterDataset, band: int | str) -> int:
    """Resolve a band specifier to a validated 1-based band index.

    An ``int`` is a 1-based index and is returned unchanged once range-checked.
    A ``str`` is a band name, matched case-insensitively against ``ds``'s
    ``band_names`` after stripping surrounding whitespace. The two lookup
    spaces never overlap: a string is always treated as a name, so a numeric
    string like ``"4"`` matches only a band literally named ``"4"``, never
    band 4.

    Parameters
    ----------
    ds : EEORasterDataset
        Dataset whose bands are being addressed.
    band : int or str
        1-based band index, or a band name declared in ``ds.band_names``.

    Returns
    -------
    int
        The 1-based index of the requested band.

    Raises
    ------
    IndexError
        If an int index is outside the range of available bands.
    ValidationError
        If a name is not found or is ambiguous (declared on more than one
        band), or if ``band`` is neither an int nor a str.
    """
    from eeo.core.exceptions import ValidationError

    count = ds.get_count()

    # bool is a subclass of int; reject it so True/False never means band 1/0.
    if isinstance(band, int) and not isinstance(band, bool):
        if band < 1 or band > count:
            raise IndexError(
                f"band index {band} out of range; dataset has {count} band(s) (valid 1..{count})"
            )
        return band

    if isinstance(band, str):
        wanted = band.strip().casefold()
        names = ds.band_names
        matches = [i for i, name in enumerate(names, start=1) if name and name.casefold() == wanted]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            declared = [n for n in names if n]
            available = ", ".join(repr(n) for n in declared) if declared else "none"
            raise ValidationError(
                f"no band named {band!r}; available band names: {available}. "
                "Set names via the band_names property, or pass a 1-based int index."
            )
        raise ValidationError(
            f"band name {band!r} is ambiguous; it matches bands {matches}. "
            "Rename the duplicates, or pass a 1-based int index."
        )

    raise ValidationError(
        f"band must be a 1-based int index or a band name (str); got {type(band).__name__}"
    )


def _declared_nodata_mask(array, nodata):
    """Boolean mask of ``array`` pixels equal to a declared ``nodata`` value.

    Returns None when ``nodata`` is None (the operand marks no pixels invalid),
    so callers can skip masking. Uses only dispatchable NumPy public-API calls
    so the expression stays valid on lazy array backends.
    """
    if nodata is None:
        return None
    if isinstance(nodata, float) and np.isnan(nodata):
        return np.isnan(array)
    return array == nodata


def resolve_output_dtype(result, *, fractional: bool):
    """Return the output dtype for a pixel-wise result per the dtype policy.

    Fractional-result ops are always float32. Exact arithmetic keeps the
    dtype NumPy promoted the computation to, narrowing float64 to float32.
    """
    if fractional:
        return np.dtype(np.float32)
    dtype = np.dtype(result.dtype)
    if dtype == np.float64:
        return np.dtype(np.float32)
    return dtype


def _combined_nodata_mask(operands):
    """Boolean mask of pixels that are nodata in any operand, or None if none are.

    Nodata is contagious: the masks of every operand that declares one are
    OR-ed together. Returns None when no operand declares a nodata value,
    which is the same condition as the contract producing no output nodata.
    """
    combined = None
    for array, nodata in operands:
        mask = _declared_nodata_mask(array, nodata)
        if mask is None:
            continue
        combined = mask if combined is None else (combined | mask)
    return combined


def resolve_output_nodata(operand_nodatas, *, out_dtype, ds_nodata):
    """Return the nodata value a pixel-wise result should record, or None.

    Decided from declared nodata values and the output dtype alone — no pixel
    data — so a block-wise operation can fix one nodata value for the whole
    output before reading the first block.

    Parameters
    ----------
    operand_nodatas : iterable
        Declared nodata value (or None) of each raster operand.
    out_dtype : numpy.dtype
        Dtype the output will be written in.
    ds_nodata : int, float, or None
        The primary operand's declared nodata, used as the sentinel for
        integer outputs.

    Returns
    -------
    float or int or None
        ``float('nan')`` for floating outputs, the integer sentinel for
        integer outputs, or None when no operand declares nodata.
    """
    declared = [nodata for nodata in operand_nodatas if nodata is not None]
    if not declared:
        # No operand declared nodata: every pixel is valid, nothing to record.
        return None
    if np.issubdtype(out_dtype, np.floating):
        return float("nan")
    sentinel = ds_nodata if ds_nodata is not None else declared[0]
    return np.array(sentinel, dtype=out_dtype).item()


def apply_nodata_mask(result, operands, *, out_dtype, out_nodata):
    """Cast a pixel-wise result to ``out_dtype`` and mark its nodata pixels.

    The dtype and nodata value are supplied rather than derived, so every
    block of a block-wise run lands in the same dtype and uses the same
    marker even when a block happens to contain no nodata pixels at all.

    Parameters
    ----------
    result : array-like
        Values computed over every pixel of the block.
    operands : list of tuple
        ``(array, nodata)`` for each raster operand, over the same pixels as
        ``result``; scalar operands are omitted since they carry no nodata.
    out_dtype : numpy.dtype
        Dtype to cast the result to.
    out_nodata : float, int, or None
        Marker written into nodata pixels, from :func:`resolve_output_nodata`.
        None means no operand declares nodata and nothing is masked.

    Returns
    -------
    array-like
        The masked result in ``out_dtype``.
    """
    result = result.astype(out_dtype)
    if out_nodata is None:
        return result
    combined = _combined_nodata_mask(operands)
    if combined is None:
        return result
    marker = np.array(out_nodata, dtype=out_dtype)
    return np.where(combined, marker, result).astype(out_dtype)
