import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import box
from affine import Affine
from rasterio.crs import CRS

from eeo.core.core import EEORasterDataset
from eeo.preprocessing import (
    standardize,
    normalize_min_max,
    normalize_percentile,
)
from eeo.preprocessing import resample
from eeo import load_array
from eeo.preprocessing.clip import clip_raster_with_bbox


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def simple_raster():
    array = np.array([[1, 2], [3, 4]], dtype=np.float32)
    return load_array(
        array,
        transform=Affine.translation(0, 2) * Affine.scale(1, -1),
        crs=CRS.from_epsg(4326),
    )


@pytest.fixture
def simple_vector(simple_raster):
    geom = box(0, 0, 2, 2)
    return gpd.GeoDataFrame(
        {"geometry": [geom]},
        crs=simple_raster.get_crs(),
    )


# ---------------------------------------------------------------------
# Standardize
# ---------------------------------------------------------------------

def test_standardize_zero_mean_unit_std(simple_raster):
    standardized = standardize(simple_raster)
    data = standardized.read()

    np.testing.assert_allclose(np.mean(data), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.std(data), 1.0, atol=1e-6)


# ---------------------------------------------------------------------
# Normalize min–max
# ---------------------------------------------------------------------

def test_normalize_min_max_default(simple_raster):
    norm = normalize_min_max(simple_raster)
    data = norm.read()

    assert np.min(data) == 0.0
    assert np.max(data) == 1.0


def test_normalize_min_max_custom_range(simple_raster):
    norm = normalize_min_max(simple_raster, new_min=-1, new_max=1)
    data = norm.read()

    assert np.min(data) == -1
    assert np.max(data) == 1


# ---------------------------------------------------------------------
# Normalize percentile
# ---------------------------------------------------------------------

def test_normalize_percentile(simple_raster):
    norm = normalize_percentile(simple_raster, lower_percentile=0, upper_percentile=100)
    data = norm.read()

    assert np.min(data) == 0.0
    assert np.max(data) == 1.0


# ---------------------------------------------------------------------
# Resample
# ---------------------------------------------------------------------

def test_resample_with_scale_factor(simple_raster):
    resampled = resample(simple_raster, scale_factor=2.0)

    assert resampled.get_shape() == (4, 4)


def test_resample_invalid_params(simple_raster):
    with pytest.raises(ValueError):
        resample(simple_raster, size=(2, 2), scale_factor=2.0)

