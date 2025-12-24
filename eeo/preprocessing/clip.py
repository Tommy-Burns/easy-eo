import os
from typing import Union

import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask
from rasterio.windows import from_bounds

from eeo.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_op


@eeo_raster_op
def clip_raster_with_vector(
    ds: EEORasterDataset,
    vector_file: Union[gpd.GeoDataFrame, str],
    *,
    crop: bool = True,
    pad: bool = False,
    all_touched: bool = False,
    invert: bool = False,
    nodata: Union[int, float, None] = None,
    show_preview: bool = False,
    plot_kwargs: dict | None = None,
) -> EEORasterDataset:
    # Ensure clipping for only rasterio-backend datasets
    backend = ds._adapter.backend
    if not isinstance(backend, rio.DatasetReader):
        raise TypeError("Clipping is only allowed on rasterio backend rasters")

    # Load vector data
    if isinstance(vector_file, gpd.GeoDataFrame):
        gdf = vector_file
    elif isinstance(vector_file, str) and os.path.isfile(vector_file):
        gdf = gpd.read_file(vector_file)
    else:
        raise TypeError(
            "vector_file must be a GeoDataFrame or a valid file path"
        )

    # Reproject vector geometries if needed
    if gdf.crs != ds.get_crs():
        gdf = gdf.to_crs(ds.get_crs())

    shapes = gdf.geometry.values

    # Perform masking
    clipped, clipped_transform = mask(
        ds.ds,
        shapes,
        crop=crop,
        pad=pad,
        all_touched=all_touched,
        invert=invert,
        nodata=nodata,
    )

    # Update metadata
    meta = ds.get_metadata().copy()
    meta.update(
        height=clipped.shape[1],
        width=clipped.shape[2],
        transform=clipped_transform,
        count=clipped.shape[0],
    )

    if nodata is not None:
        meta["nodata"] = nodata

    # Write to MemoryFile
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(clipped)

    # Optional preview
    if show_preview:
        EEORasterDataset.from_rasterio(out_ds).plot_raster(**(plot_kwargs or {}))

    return EEORasterDataset.from_rasterio(out_ds)

from rasterio.windows import from_bounds, Window


@eeo_raster_op
def clip_raster_with_bbox(
        ds,
        bbox: tuple | list,
        plot_kwargs=None,
        show_preview: bool = False
) -> "EEORasterDataset":

    # Ensure rasterio backend
    backend = ds._adapter.backend
    if not isinstance(backend, rio.DatasetReader):
        raise TypeError("Clipping is only allowed on rasterio backend rasters")

    # Validate bbox
    if not (isinstance(bbox, (tuple, list)) and len(bbox) == 4):
        raise ValueError("bbox must be (minx, miny, maxx, maxy)")

    minx, miny, maxx, maxy = bbox

    # Compute window
    window = from_bounds(minx, miny, maxx, maxy, ds.ds.transform)
    window = window.round_offsets().round_lengths()

    # Ensure correct bbox
    if window.width <= 0 or window.height <= 0:
        raise ValueError(
            "Bounding box does not intersect raster extent. "
            f"Raster bounds: {ds.get_bounds()}, bbox: {bbox}"
        )

    transform = rio.windows.transform(window, ds.ds.transform)

    # Read clipped data
    clipped = ds.ds.read(window=window)

    # Update metadata
    meta = ds.get_metadata().copy()
    meta.update(
        height=clipped.shape[1],
        width=clipped.shape[2],
        transform=transform,
    )

    # Write to MemoryFile
    memfile = rio.io.MemoryFile()
    dataset = memfile.open(**meta)
    dataset.write(clipped)

    if show_preview:
        EEORasterDataset.from_rasterio(dataset).plot_raster(**(plot_kwargs or {}))

    return EEORasterDataset.from_rasterio(dataset)

