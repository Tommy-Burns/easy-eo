"""Tests for eeo.common"""
import pytest
from rasterio.enums import Resampling

import numpy as np
from affine import Affine
from rasterio.crs import CRS

from eeo import load_array
from eeo.common import normalize_resampling_method, align_raster_to_target, mask_nodata


# RESAMPLING NORMALIZATION
# Accepts Resampling enum
def test_normalize_resampling_enum_passthrough():
    method = normalize_resampling_method(Resampling.bilinear)
    assert method is Resampling.bilinear


# Accepts valid string (case-insensitive)
@pytest.mark.parametrize("value", ["nearest", "Nearest", " BILINEAR "])
def test_normalize_resampling_string(value):
    method = normalize_resampling_method(value)
    assert isinstance(method, Resampling)


# Rejects invalid string
def test_normalize_resampling_invalid_string():
    with pytest.raises(ValueError, match="Invalid resampling method"):
        normalize_resampling_method("invalid_method")


# Rejects invalid type
def test_normalize_resampling_invalid_type():
    with pytest.raises(TypeError):
        normalize_resampling_method(123)


# ALIGN RASTER TO TARGET
# Reject when shape and transform match
def test_align_raster_noop():
    array = np.ones((10, 10))
    transform = Affine.identity()
    crs = CRS.from_epsg(4326)

    ds = load_array(array, transform=transform, crs=crs)
    target = load_array(array, transform=transform, crs=crs)

    result = align_raster_to_target(ds, target)

    assert result is ds

# Resamples when shape differs
def test_align_raster_resamples():
    src = load_array(
        np.ones((10, 10)),
        transform=Affine.identity(),
        crs=CRS.from_epsg(4326),
    )

    target = load_array(
        np.ones((20, 20)),
        transform=Affine.scale(2, 2),  # <-- force mismatch
        crs=CRS.from_epsg(4326),
    )

    result = align_raster_to_target(src, target)

    assert result.get_shape() == target.get_shape()


# NO DATA MASKING
# Masks no data values
def test_mask_nodata_applies_nan():
    array = np.array([[1, -999], [3, 4]], dtype=float)
    nodata = -999

    ds = load_array(array, nodata=nodata)

    masked = mask_nodata(ds, array)

    assert np.isnan(masked[0, 1])
    assert masked[0, 0] == 1

# No data gives unchanged array
def test_mask_nodata_no_nodata():
    array = np.array([[1, 2], [3, 4]], dtype=float)

    ds = load_array(array)

    masked = mask_nodata(ds, array)

    np.testing.assert_array_equal(masked, array)

