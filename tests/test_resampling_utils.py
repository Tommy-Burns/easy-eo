"""Tests for eeo.common"""

import numpy as np
import pytest
from rasterio.enums import Resampling

from eeo import load_array
from eeo.common import align_raster_to_target, mask_nodata, normalize_resampling_method
from eeo.core.exceptions import ValidationError


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
    with pytest.raises(ValidationError, match="invalid resampling method"):
        normalize_resampling_method("invalid_method")


# Rejects invalid type
def test_normalize_resampling_invalid_type():
    with pytest.raises(ValidationError):
        normalize_resampling_method(123)


# ALIGN RASTER TO TARGET
# Returns the input unchanged when shape and transform already match
def test_align_raster_noop(single_band_float32):
    result = align_raster_to_target(single_band_float32, single_band_float32)

    assert result is single_band_float32


# Resamples when shape/resolution differs
def test_align_raster_resamples(shape_mismatch_pair):
    fine, coarse = shape_mismatch_pair

    result = align_raster_to_target(coarse, fine)

    assert result.get_shape() == fine.get_shape()


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
