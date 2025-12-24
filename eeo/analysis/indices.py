"""
This library exposes algebraic primitives instead of predefined indices.
Most vegetation and water indices can be expressed directly using
``normalized_difference`` or raster arithmetic.
"""

from typing import Union

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
        return_as_ndarray: bool = False
) -> Union[np.ndarray, EEORasterDataset]:
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
