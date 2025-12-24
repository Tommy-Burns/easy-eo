from typing import Union

import numpy as np
import rasterio as rio
from eeo.common import mask_nodata
from eeo.core.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_op

Coordinate = Union[tuple[float, float], list[float]]


@eeo_raster_op
def extract_value_at_coordinate(
        ds: EEORasterDataset,
        coordinates: Coordinate,
        band_idx: int = 1
) -> Union[int, float]:

    if len(coordinates) != 2:
        raise ValueError(f"Expected 2 coordinates, got {len(coordinates)}")

    backend = ds._adapter.backend
    if not isinstance(backend, rio.DatasetReader):
        ds = ds.to_rasterio()
        backend = ds._adapter.backend

    x, y = coordinates
    row, col = backend.index(x, y)

    return ds.get_band(band_idx)[row, col]



@eeo_raster_op
def get_maximum_pixel(
        ds: EEORasterDataset,
        band_idx: int = 1,
        *,
        return_position_as_pixel_coordinate: bool = False,
) -> dict:
    band = ds.read() if ds.get_count() == 1 else ds.get_band(band_idx)
    band = mask_nodata(ds, band)

    # get max value
    value = float(np.nanmax(band))
    _, row, col = np.unravel_index(np.nanargmax(band), band.shape)

    if return_position_as_pixel_coordinate:
        position = (row, col)
    else:
        transform = ds.get_transform()
        position = transform * (col, row)

    return {"value": value, "position": position}


@eeo_raster_op
def get_minimum_pixel(
        ds: EEORasterDataset,
        band_idx: int = 1,
        *,
        return_position_as_pixel_coordinate: bool = False,
) -> dict:
    band = ds.read() if ds.get_count() == 1 else ds.get_band(band_idx)
    band = mask_nodata(ds, band)

    # get min value
    value = float(np.nanmin(band))
    _, row, col = np.unravel_index(np.nanargmin(band), band.shape)

    if return_position_as_pixel_coordinate:
        position = (row, col)
    else:
        transform = ds.get_transform()
        position = transform * (col, row)

    return {"value": value, "position": position}


@eeo_raster_op
def get_mean_pixel(
        ds: EEORasterDataset,
        band_idx: int = 1,
        *,
        return_position_as_pixel_coordinate: bool = False,
) -> dict:
    band = ds.read() if ds.get_count() == 1 else ds.get_band(band_idx)
    band = mask_nodata(ds, band)

    mean_value = float(np.nanmean(band))
    diff = np.abs(band - mean_value)

    _, row, col = np.unravel_index(np.nanargmin(diff), diff.shape)

    if return_position_as_pixel_coordinate:
        position = (row, col)
    else:
        transform = ds.get_transform()
        position = transform * (col, row)

    return {"value": mean_value, "position": position}


@eeo_raster_op
def get_percentile_pixel(
        ds: EEORasterDataset,
        percentile: float,
        band_idx: int = 1,
        *,
        return_position_as_pixel_coordinate: bool = False,
) -> dict:
    band = ds.read() if ds.get_count() == 1 else ds.get_band(band_idx)
    band = mask_nodata(ds, band)

    perc_value = float(np.nanpercentile(band, percentile))
    diff = np.abs(band - perc_value)

    _, row, col = np.unravel_index(np.nanargmin(diff), band.shape)

    if return_position_as_pixel_coordinate:
        position = (row, col)
    else:
        transform = ds.get_transform()
        position = transform * (col, row)

    return {"value": perc_value, "position": position}

