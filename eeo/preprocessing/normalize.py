from typing import Union

import numpy as np
import rasterio as rio

from eeo.core.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_op


@eeo_raster_op
def standardize(ds: EEORasterDataset) -> EEORasterDataset:
    """Z-Score standardization"""
    data = ds.read()
    mean_value = np.mean(data)
    std_value = np.std(data)
    standardized_data = (data - mean_value) / std_value
    meta = ds.get_metadata()
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(standardized_data)
    return EEORasterDataset.from_rasterio(out_ds)


@eeo_raster_op
def normalize_min_max(ds: EEORasterDataset, *, new_min: Union[float, int] = 0.0,
                      new_max: Union[float, int] = 1.0) -> EEORasterDataset:
    """Normalize raster to new_min, new_max"""
    data = ds.read()
    old_min, old_max = np.min(data), np.max(data)
    normalized_data = (data - old_min) / (old_max - old_min)
    normalized_data = normalized_data * (new_max - new_min) + new_min
    meta = ds.get_metadata()
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(normalized_data)
    return EEORasterDataset.from_rasterio(out_ds)


@eeo_raster_op
def normalize_percentile(ds: EEORasterDataset, *, lower_percentile: Union[float, int] = 0.0,
                         upper_percentile: Union[float, int] = 1.0) -> EEORasterDataset:
    """Normalize raster to lower_percentile, upper_percentile"""
    data = ds.read()
    array_min, array_max = np.nanpercentile(data, (lower_percentile, upper_percentile))
    normalized_data = np.clip((data - array_min) / (array_max - array_min), 0, 1)
    meta = ds.get_metadata()
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(normalized_data)
    return EEORasterDataset.from_rasterio(out_ds)
