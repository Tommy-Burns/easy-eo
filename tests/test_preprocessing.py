import numpy as np
import pytest

from eeo.preprocessing import (
    resample,
    standardize,
    normalize_min_max,
    normalize_percentile,
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
    norm = normalize_percentile(
        single_band_float32, lower_percentile=0, upper_percentile=100
    )
    data = norm.read()

    assert np.min(data) == 0.0
    assert np.max(data) == 1.0


# ---------------------------------------------------------------------
# Resample
# ---------------------------------------------------------------------

def test_resample_with_scale_factor(single_band_float32):
    resampled = resample(single_band_float32, scale_factor=2.0)

    assert resampled.get_shape() == (12, 12)


def test_resample_invalid_params(single_band_float32):
    with pytest.raises(ValueError):
        resample(single_band_float32, size=(2, 2), scale_factor=2.0)
