from typing import Union, Optional
import pyproj
from pyproj import CRS
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling

from eeo.common import normalize_resampling_method
from eeo.core.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_op


@eeo_raster_op
def reproject_raster(
        ds: EEORasterDataset,
        *,
        target_crs: Union[int, str, CRS],
        resampling_method: Resampling = Resampling.nearest
) -> EEORasterDataset:
    # Ensure reprojection for only rasterio-backend datasets
    backend = ds._adapter.backend
    if not isinstance(backend, rio.DatasetReader):
        raise TypeError("Reprojection is only allowed on rasterio backend rasters")

    # Normalize resampling method
    resampling_method = normalize_resampling_method(resampling_method)
    # Normalise CRS
    if isinstance(target_crs, (int, str)):
        crs = pyproj.CRS.from_user_input(target_crs)
    else:
        crs = target_crs

    if not isinstance(crs, pyproj.CRS):
        raise TypeError("Invalid CRS. Must be int, str, or pyproj.CRS")

    # Get dataset bounds
    left, bottom, right, top = ds.get_bounds()

    # Compute transform and new size
    transform, width, height = calculate_default_transform(
        src_crs=ds.get_crs(),
        dst_crs=crs,
        width=ds.get_width(),
        height=ds.get_shape()[1],
        left=left,
        bottom=bottom,
        right=right,
        top=top
    )

    # Update the metadata
    meta = ds.get_metadata()
    meta.update({
        'crs': crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    # return in-memory dataset
    memfile = rio.io.MemoryFile()
    dataset = memfile.open(**meta)

    for i in range(1, ds.get_count() + 1):
        reproject(
            source=rio.band(ds.ds, i),
            destination=rio.band(dataset, i),
            src_transform=ds.get_transform(),
            src_crs=ds.get_crs(),
            dst_transform=transform,
            dst_crs=crs,
            resampling=resampling_method
        )
    return EEORasterDataset.from_rasterio(dataset)
