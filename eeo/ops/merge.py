from typing import Union, Iterable

import numpy as np
import rasterio as rio
from rasterio.merge import merge

from eeo.common import normalize_resampling_method
from eeo.core.core import EEORasterDataset, _save_raster
from eeo.core.decorators import eeo_raster_op

@eeo_raster_op
def mosaic(
        ds: EEORasterDataset,
        others: Union[EEORasterDataset, Iterable[EEORasterDataset]],
        *,
        resampling_method: str = "nearest",
        save_path: str = None,
        auto_reproject: bool = False,
        **kwargs,
) -> Union[EEORasterDataset, None]:
    # Ensure mosaic for only rasterio-backend datasets
    backend = ds._adapter.backend
    if not isinstance(backend, rio.DatasetReader):
        raise TypeError("Mosaicking is only allowed on rasterio backend rasters")

    # normalize resampling
    resampling_method = normalize_resampling_method(resampling_method)

    # normalize inputs to list
    if isinstance(others, EEORasterDataset):
        others = [others]
    else:
        others = list(others)

    if not others:
        raise ValueError("At least one raster must be provided for mosaicking")

    # CRS validation
    src_datasets: list[EEORasterDataset] = [ds]
    target_crs = ds.get_crs()
    for obj in others:
        if obj.get_crs() != target_crs:
            if auto_reproject:
                obj = obj.reproject_raster(target_crs)
            else:
                raise ValueError(
                    "All rasters must have the same CRS for mosaicking. "
                    "Set auto_reproject=True to allow reprojection."
                )

        src_datasets.append(obj)

    # extract datasets and perform mosaics
    datasets = [d.ds for d in src_datasets]
    mosaic_data, out_transform = merge(
        datasets,
        resampling=resampling_method,
        **kwargs
    )

    # modify metadata
    meta = ds.get_metadata().copy()
    meta.update(
        transform=out_transform,
        height=mosaic_data.shape[1],
        width=mosaic_data.shape[2],
        count=mosaic_data.shape[0],
        dtype=mosaic_data.dtype,
    )

    # write to memory file
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(mosaic_data)

    # save or return EEORasterDataset
    if save_path is not None:
        out_ds.save_raster(path=save_path)
        return None

    return EEORasterDataset.from_rasterio(out_ds)



@eeo_raster_op
def stack(
        ds: EEORasterDataset,
        others: Union[EEORasterDataset, Iterable[EEORasterDataset]],
) -> EEORasterDataset:
    # Ensure stack for only rasterio-backend datasets
    backend = ds._adapter.backend
    if not isinstance(backend, rio.DatasetReader):
        raise TypeError("Stacking is only allowed on rasterio backend rasters")

    # normalize inputs
    if isinstance(others, EEORasterDataset):
        others = [others]
    else:
        others = list(others)

    if not others:
        raise ValueError("At least one raster must be provided for stacking")

    # alignment checks
    for item in others:
        if (
            item.get_crs() != ds.get_crs()
            or item.get_transform() != ds.get_transform()
            or item.get_shape() != ds.get_shape()
        ):
            raise ValueError("All rasters must have identical CRS, transform, and shape")

    # read data
    arrays = [ds.read()]
    for obj in others:
        arrays.append(obj.read())

    # stack the arrays
    stacked = np.vstack(arrays)

    # metadata update
    meta = ds.get_metadata().copy()
    meta.update(
        count=stacked.shape[0],
        dtype=stacked.dtype
    )

    # save to memory file
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(stacked)

    return EEORasterDataset.from_rasterio(out_ds)
