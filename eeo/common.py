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
    object. A rasterio-backed dataset's ``backend`` may be a
    ``rasterio.io.DatasetReader`` (opened from a file) or a
    ``rasterio.io.DatasetWriter`` (produced in memory by an operation, e.g.
    the result of any algebra op or ``to_rasterio()``); both are valid
    rasterio backends. Checking ``isinstance(backend, DatasetReader)`` misses
    the writer case and wrongly rejects genuinely rasterio-backed datasets.

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


def _output_dtype(result, *, fractional: bool):
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


def apply_nodata_contract(result, operands, *, fractional: bool, ds_nodata):
    """Apply the library nodata & dtype contract to a pixel-wise result.

    Masks the pixels that are nodata in any operand (nodata is contagious),
    casts to the contract's output dtype, and reports the nodata value to
    record in the output metadata.

    Parameters
    ----------
    result : array-like
        Values computed over every pixel. Nodata pixels are overwritten here,
        so computing them first (then masking) is equivalent to masking first
        for element-wise operations.
    operands : list of tuple
        ``(array, nodata)`` for each raster operand; scalar operands are
        omitted since they carry no nodata.
    fractional : bool
        True for operations whose result is inherently fractional (float32
        output regardless of input dtype).
    ds_nodata : int, float, or None
        The primary operand's declared nodata, used as the sentinel for
        integer outputs.

    Returns
    -------
    tuple
        ``(final_array, out_nodata)`` — the masked, correctly typed array and
        the nodata value for the output metadata (``float('nan')`` for
        floating outputs, the integer sentinel for integer outputs, or None
        when no operand declares nodata).
    """
    out_dtype = _output_dtype(result, fractional=fractional)
    is_float_out = np.issubdtype(out_dtype, np.floating)

    combined = None
    declared = []
    for array, nodata in operands:
        if nodata is not None:
            declared.append(nodata)
        mask = _declared_nodata_mask(array, nodata)
        if mask is None:
            continue
        combined = mask if combined is None else (combined | mask)

    result = result.astype(out_dtype)

    if combined is None:
        # No operand declared nodata: nothing to mask, no output nodata.
        return result, None

    if is_float_out:
        marker = np.array(np.nan, dtype=out_dtype)
        out_nodata: float = float("nan")
    else:
        sentinel = ds_nodata if ds_nodata is not None else declared[0]
        marker = np.array(sentinel, dtype=out_dtype)
        out_nodata = marker.item()

    final = np.where(combined, marker, result).astype(out_dtype)
    return final, out_nodata
