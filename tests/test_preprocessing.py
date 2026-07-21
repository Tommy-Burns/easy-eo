import numpy as np
import pytest

from eeo.core.exceptions import ValidationError
from eeo.preprocessing import (
    normalize_min_max,
    normalize_percentile,
    resample,
    standardize,
)

# ---------------------------------------------------------------------
# Standardize
# ---------------------------------------------------------------------


def test_standardize_zero_mean_unit_std(single_band_float32):
    standardized = standardize(single_band_float32)
    data = standardized.read()

    np.testing.assert_allclose(np.mean(data), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.std(data), 1.0, atol=1e-6)


# ---------------------------------------------------------------------
# Normalize min–max
# ---------------------------------------------------------------------


def test_normalize_min_max_default(single_band_float32):
    norm = normalize_min_max(single_band_float32)
    data = norm.read()

    assert np.min(data) == 0.0
    assert np.max(data) == 1.0


def test_normalize_min_max_custom_range(single_band_float32):
    norm = normalize_min_max(single_band_float32, new_min=-1, new_max=1)
    data = norm.read()

    assert np.min(data) == -1
    assert np.max(data) == 1


# ---------------------------------------------------------------------
# Normalize percentile
# ---------------------------------------------------------------------


def test_normalize_percentile(single_band_float32):
    norm = normalize_percentile(single_band_float32, lower_percentile=0, upper_percentile=100)
    data = norm.read()

    assert np.min(data) == 0.0
    assert np.max(data) == 1.0


def test_normalize_percentile_default_is_2_98(single_band_float32):
    # Regression test for the (2, 98) default (CODE_STYLE.md convention);
    # a prior (0.0, 1.0) default silently normalized against the 0th-1st
    # percentile range instead, clipping nearly everything to 1.0.
    default_result = normalize_percentile(single_band_float32).read()
    explicit_result = normalize_percentile(
        single_band_float32, lower_percentile=2, upper_percentile=98
    ).read()

    np.testing.assert_array_equal(default_result, explicit_result)


# ---------------------------------------------------------------------
# Resample
# ---------------------------------------------------------------------


def test_resample_with_scale_factor(single_band_float32):
    resampled = resample(single_band_float32, scale_factor=2.0)

    assert resampled.get_shape() == (12, 12)


def test_resample_invalid_params(single_band_float32):
    with pytest.raises(ValidationError):
        resample(single_band_float32, size=(2, 2), scale_factor=2.0)


def test_resample_default_is_nearest(single_band_float32):
    # Regression test for the "nearest" default (CLAUDE.md known issue #3;
    # matches reproject_raster's default). Nearest-neighbor resampling only
    # ever copies existing source values; a prior "bilinear" default would
    # interpolate new in-between values on this gradient raster.
    resampled = resample(single_band_float32, scale_factor=2.0)

    original_values = set(single_band_float32.read().ravel().tolist())
    resampled_values = set(resampled.read().ravel().tolist())

    assert resampled_values <= original_values
