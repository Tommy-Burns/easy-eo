"""Spectral analysis helpers built on the algebra primitives.

Rather than predefined indices, this module exposes algebraic primitives:
most vegetation and water indices can be expressed directly using
``normalized_difference`` or raster arithmetic.
"""

import numpy as np
import rasterio as rio

from eeo.common import align_raster_to_target, apply_nodata_contract, get_nodata
from eeo.core.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_op
from eeo.core.exceptions import AlignmentError


@eeo_raster_op
def normalized_difference(
    ds: EEORasterDataset,
    other: EEORasterDataset,
    *,
    auto_align: bool = True,
    method: str = "bilinear",
    return_as_ndarray: bool = False,
) -> np.ndarray | EEORasterDataset:
    """Compute the normalized difference ``(ds - other) / (ds + other)``.

    This is the family of indices that includes NDVI (NIR, Red) and NDWI
    (Green, NIR): ``ds`` is the first band of the pair, ``other`` the second.
    NumPy-backed inputs are promoted to rasterio, and ``other`` is resampled
    onto ``ds``'s grid when ``auto_align`` is True.

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
    return_as_ndarray : bool, default False
        If True, return the raw NumPy array instead of an
        ``EEORasterDataset``.

    Returns
    -------
    EEORasterDataset or numpy.ndarray
        Float32 result in ``[-1, 1]`` — an ``EEORasterDataset`` by default,
        or the raw ``(bands, height, width)`` array when
        ``return_as_ndarray=True``. Pixels where ``ds + other == 0`` are set
        to 0. A pixel that is nodata in either operand is nodata (NaN) in the
        output; the output nodata value is NaN when either input declares
        nodata, otherwise None.

    Raises
    ------
    AlignmentError
        If the two rasters are on different grids and ``auto_align`` is False.

    Notes
    -----
    Reads both rasters fully into memory rather than streaming block-wise.
    Nodata pixels are masked before the ratio; separately, a zero denominator
    (``ds + other == 0``) is guarded by setting those pixels to 0.

    Examples
    --------
    >>> ndvi = ds_nir.normalized_difference(ds_red)
    >>> ndvi_array = ds_nir.normalized_difference(ds_red, return_as_ndarray=True)
    """
    # No-op when the dataset is already rasterio-backed
    ds = ds.to_rasterio()
    if ds.get_shape() != other.get_shape() or ds.get_transform() != other.get_transform():
        if auto_align:
            other = align_raster_to_target(other, ds, method=method)
        else:
            raise AlignmentError(
                "rasters must share the same grid for this operation; "
                f"got shape {other.get_shape()} vs {ds.get_shape()}. "
                "Pass auto_align=True to resample the other raster onto this grid."
            )

    ds_nodata = get_nodata(ds)
    other_nodata = get_nodata(other)
    a_raw = ds.read()
    b_raw = other.read()
    a = a_raw.astype(rio.float32)
    b = b_raw.astype(rio.float32)

    # np.where instead of in-place mask assignment so the expression stays
    # dispatchable to lazy array backends (which reject item assignment). The
    # denominator guard catches both 0/0 (nan) and x/0 (inf) at a+b == 0.
    denom = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        quotient = (a - b) / denom
    nd = np.where(denom != 0, quotient, np.float32(0))

    # Mask nodata last so a masked pixel is NaN regardless of its ratio.
    nd, out_nodata = apply_nodata_contract(
        nd,
        [(a_raw, ds_nodata), (b_raw, other_nodata)],
        fractional=True,
        ds_nodata=ds_nodata,
    )

    if return_as_ndarray:
        return nd

    meta = ds.get_metadata().copy()

    # Ensure correct metadata for writing
    meta.update(
        driver="GTiff",
        dtype="float32",
        nodata=out_nodata,
        height=nd.shape[-2],
        width=nd.shape[-1],
        count=nd.shape[0],
    )

    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(nd)

    return EEORasterDataset.from_rasterio(out_ds)
