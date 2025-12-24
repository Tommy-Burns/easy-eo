from typing import Optional

import rasterio as rio
from rasterio.transform import Affine

from eeo.common import normalize_resampling_method
from eeo.core.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_op


@eeo_raster_op
def resample(ds: EEORasterDataset,
             *,
             size: Optional[tuple[int, int]] = None,
             scale_factor: Optional[float] = None,
             resolution: Optional[tuple[float, float]] = None,
             resampling_method: str = "bilinear",
             plot_kwargs=None,
             show_preview: bool = False
             ) -> EEORasterDataset:
    # Ensure resampling for only rasterio-backend datasets
    backend = ds._adapter.backend
    if not isinstance(backend, rio.DatasetReader):
        raise TypeError("Resampling is only allowed on rasterio backend rasters")

    params = [size, scale_factor, resolution]
    if sum(p is not None for p in params) != 1:
        raise ValueError("Provide exactly one of: size=, scale_factor=, resolution=")

    # Compute new dimensions
    # --- When size is provided ---
    if size is not None:
        new_height, new_width = size

    # --- When scale factor is provided ---
    elif scale_factor is not None:
        new_width = int(ds.get_width() * scale_factor)
        new_height = int(ds.get_height() * scale_factor)

    # --- When size is provided ---
    else:
        xres, yres = resolution
        bounds = ds.get_bounds()

        new_width = int((bounds.right - bounds.left) / abs(xres))
        new_height = int((bounds.top - bounds.bottom) / abs(yres))
    try:
        # Resampling using bilinear interpolation
        resampling_enum = normalize_resampling_method(resampling_method)
        data = ds.read(
            out_shape=(ds.get_count(), new_height, new_width),
            resampling=resampling_enum,
        )

        # Computing scale transform
        scale_x = ds.get_width() / new_width
        scale_y = ds.get_height() / new_height

        transform = ds.get_transform() * Affine.scale(scale_x, scale_y)

        # Save or return EEORasterDataset
        # Update metadata
        meta = ds.get_metadata()
        meta.update(
            transform=transform,
            height=new_height,
            width=new_width,
        )
        # Write to MemoryFile
        memfile = rio.io.MemoryFile()
        with memfile.open(**meta) as mem:
            mem.write(data)

        dataset = memfile.open()

        if show_preview:
            EEORasterDataset.from_rasterio(dataset).plot_raster(**(plot_kwargs or {}))

        return EEORasterDataset.from_rasterio(dataset)
    except Exception as e:
        raise RuntimeError("Could not scale raster data") from e
