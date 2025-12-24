from typing import Union

import numpy as np
import rasterio as rio

from eeo.common import align_raster_to_target
from eeo.core.core import EEORasterDataset
from eeo.core.decorators import eeo_raster_op


# ARITHMETIC AND ALGEBRA
@eeo_raster_op
def add(ds: EEORasterDataset, other: Union[EEORasterDataset, float, int], *, auto_align: bool = True,
        method: str = "bilinear") -> EEORasterDataset:
    """Pixel-wise addition of two rasters"""
    if isinstance(other, EEORasterDataset):
        if ds.get_shape() != other.get_shape() or ds.get_transform() != other.get_transform():
            if auto_align:
                other = align_raster_to_target(other, ds, method=method)
            else:
                raise ValueError("Rasters must have the same shape and alignment for arithmetic operations")
        data = ds.read() + other.read()
    else:
        data = ds.read() + other

    meta = ds.get_metadata()
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(data)
    return EEORasterDataset.from_rasterio(out_ds)


@eeo_raster_op
def subtract(ds: EEORasterDataset, other: Union[EEORasterDataset, float, int], *, auto_align: bool = True,
             method: str = "bilinear") -> EEORasterDataset:
    """Pixel-wise subtraction of two rasters"""
    if isinstance(other, EEORasterDataset):
        if ds.get_shape() != other.get_shape() or ds.get_transform() != other.get_transform():
            if auto_align:
                other = align_raster_to_target(other, ds, method=method)
            else:
                raise ValueError("Rasters must have the same shape and alignment for arithmetic operations")
        data = ds.read() - other.read()
    else:
        data = ds.read() - other
    meta = ds.get_metadata()
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(data)
    return EEORasterDataset.from_rasterio(out_ds)


@eeo_raster_op
def multiply(ds: EEORasterDataset, other: Union[EEORasterDataset, float, int], *, auto_align: bool = True,
             method: str = "bilinear") -> EEORasterDataset:
    """Pixel-wise multiplication of two rasters"""
    if isinstance(other, EEORasterDataset):
        if ds.get_shape() != other.get_shape() or ds.get_transform() != other.get_transform():
            if auto_align:
                other = align_raster_to_target(other, ds, method=method)
            else:
                raise ValueError("Rasters must have the same shape and alignment for arithmetic operations")

        data = ds.read() * other.read()
    else:
        data = ds.read() * other
    meta = ds.get_metadata()
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(data)
    return EEORasterDataset.from_rasterio(out_ds)


@eeo_raster_op
def divide(
    ds: EEORasterDataset,
    other: Union[EEORasterDataset, float, int],
    *,
    auto_align: bool = True,
    method: str = "bilinear",
    safe: bool = True,
) -> EEORasterDataset:
    """Pixel-wise division of two rasters"""

    src_data = ds.read()

    # Resolve other operand
    if isinstance(other, EEORasterDataset):
        if ds.get_shape() != other.get_shape() or ds.get_transform() != other.get_transform():
            if auto_align:
                other = align_raster_to_target(other, ds, method=method)
            else:
                raise ValueError(
                    "Rasters must have the same shape and alignment for arithmetic operations"
                )
        other_data = other.read()
    else:
        other_data = other

    # ---- SAFE DIVIDE ----
    if safe:
        if np.isscalar(other_data):
            if other_data == 0:
                data = np.zeros_like(src_data, dtype=np.float32)
            else:
                data = src_data / other_data
        else:
            data = np.divide(
                src_data,
                other_data,
                out=np.zeros_like(src_data, dtype=np.float32),
                where=other_data != 0,
            )
    else:
        data = src_data / other_data

    # Write output
    meta = ds.get_metadata()
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(data)

    return EEORasterDataset.from_rasterio(out_ds)



@eeo_raster_op
def power(ds: EEORasterDataset, exponent: Union[int, float]) -> EEORasterDataset:
    """Pixel-wise exponentiation (scalar exponent)"""
    data = ds.read() ** exponent
    meta = ds.get_metadata()
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(data)
    return EEORasterDataset.from_rasterio(out_ds)


# TRANSFORMATIONS
@eeo_raster_op
def sqrt(ds: EEORasterDataset) -> EEORasterDataset:
    """Pixel-wise square root of the raster, non-negative"""
    data = np.sqrt(np.maximum(ds.read(), 0))
    meta = ds.get_metadata()
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(data)
    return EEORasterDataset.from_rasterio(out_ds)


@eeo_raster_op
def log(ds: EEORasterDataset, base: Union[int, float] = np.e) -> EEORasterDataset:
    """Pixel-wise log of the raster, safe with zeros"""
    data = np.log(np.maximum(ds.read(), 1e-10)) / np.log(base)
    meta = ds.get_metadata()
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(data)
    return EEORasterDataset.from_rasterio(out_ds)


@eeo_raster_op
def absolute(ds: EEORasterDataset) -> EEORasterDataset:
    """Pixel-wise absolute value of the raster"""
    data = np.abs(ds.read())
    meta = ds.get_metadata()
    memfile = rio.io.MemoryFile()
    out_ds = memfile.open(**meta)
    out_ds.write(data)
    return EEORasterDataset.from_rasterio(out_ds)

