"""Spectral analysis helpers built on the algebra primitives.

Rather than predefined indices, this module exposes algebraic primitives:
most vegetation and water indices can be expressed directly using
``normalized_difference`` or raster arithmetic.
"""

import numpy as np
import rasterio as rio

from eeo.common import align_raster_to_target
from eeo.core.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_op


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
        transform differ. If False, a mismatch raises ``ValueError``.
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
        to 0. The nodata value is carried over from ``ds`` unchanged.

    Raises
    ------
    ValueError
        If the two rasters are on different grids and ``auto_align`` is False.

    Notes
    -----
    Reads both rasters fully into memory rather than streaming block-wise.
    Nodata pixels are not masked before the computation; only division by
    zero (``ds + other == 0``) is guarded, by setting those pixels to 0.

    Examples
    --------
    >>> ndvi = ds_nir.normalized_difference(ds_red)
    >>> ndvi_array = ds_nir.normalized_difference(ds_red, return_as_ndarray=True)
    """
    # Ensure reprojection for only rasterio-backend datasets
    backend = ds._adapter.backend
    if not isinstance(backend, rio.DatasetReader):
        ds = ds.to_rasterio()
    if ds.get_shape() != other.get_shape() or ds.get_transform() != other.get_transform():
        if auto_align:
            other = align_raster_to_target(other, ds, method=method)
        else:
            raise ValueError("Rasters must have the same shape and alignment")

    a = ds.read().astype(rio.float32)
    b = other.read().astype(rio.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        nd = (a - b) / (a + b)
        nd[np.isnan(nd)] = 0

    if return_as_ndarray:
        return nd

    meta = ds.get_metadata().copy()

    # Ensure correct metadata for writing
    meta.update(
        driver="GTiff",
        dtype="float32",
        height=nd.shape[-2],
        width=nd.shape[-1],
        count=nd.shape[0],
    )

    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(nd)

    return EEORasterDataset.from_rasterio(out_ds)
