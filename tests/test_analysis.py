import numpy as np
import pytest
from affine import Affine

from eeo.core import EEORasterDataset

from test_conf import (
    raster_from_array,
    multiband_raster_from_array,
    multi_band_array, single_band_array,
    dummy_crs, dummy_transform
)


# normalized difference
def test_normalized_difference_basic(raster_from_array):
    ds1 = raster_from_array
    ds2 = raster_from_array

    result = ds1.normalized_difference(ds2)

    # Should return EEORasterDataset
    assert hasattr(result, "read")

    data = result.read()
    assert np.allclose(data, 0.0)


# ndarray returns
def test_normalized_difference_returns_ndarray(raster_from_array):
    ds1 = raster_from_array
    ds2 = raster_from_array

    nd = ds1.normalized_difference(ds2, return_as_ndarray=True)

    assert isinstance(nd, np.ndarray)
    assert nd.shape == ds1.read().shape


# divide-by-zero safety
def test_normalized_difference_divide_by_zero(dummy_transform, dummy_crs):
    array = np.zeros((3, 3), dtype=np.float32)

    from eeo.core import EEORasterDataset
    ds1 = EEORasterDataset.from_array(array, transform=dummy_transform, crs=dummy_crs)
    ds2 = EEORasterDataset.from_array(array, transform=dummy_transform, crs=dummy_crs)

    nd = ds1.normalized_difference(ds2, return_as_ndarray=True)

    assert np.all(nd == 0)



# extract value at coordinate
def test_extract_value_at_coordinate(raster_from_array):
    ds = raster_from_array

    # With identity transform: pixel (1, 1) → world (1, -1)
    value = ds.extract_value_at_coordinate((1, -1))

    assert value == 5


# invalid coordinate length
def test_extract_value_invalid_coordinates(raster_from_array):
    with pytest.raises(ValueError):
        raster_from_array.extract_value_at_coordinate((1,))


# Maximum pixel
def test_get_maximum_pixel_value_and_position(raster_from_array):
    result = raster_from_array.get_maximum_pixel()

    assert result["value"] == 9.0
    assert isinstance(result["position"], tuple)


# Maximum pixel - pixel coords
def test_get_maximum_pixel_pixel_coordinates(raster_from_array):
    result = raster_from_array.get_maximum_pixel(
        return_position_as_pixel_coordinate=True
    )

    assert result["position"] == (2, 2)


# Minimum pixel
def test_get_minimum_pixel_value_and_position(raster_from_array):
    result = raster_from_array.get_minimum_pixel()

    assert result["value"] == 1.0


# Mean pixel
def test_get_mean_pixel(raster_from_array):
    result = raster_from_array.get_mean_pixel()

    assert result["value"] == 5.0

def test_get_mean_pixel_location_pixel_space(raster_from_array):
    result = raster_from_array.get_mean_pixel(
        return_position_as_pixel_coordinate=True
    )

    assert result["position"] == (1, 1)

# Percentile pixel
def test_get_percentile_pixel(raster_from_array):
    result = raster_from_array.get_percentile_pixel(50)

    assert result["value"] == 5.0


def test_get_percentile_pixel_pixel_coordinates(raster_from_array):
    result = raster_from_array.get_percentile_pixel(
        50, return_position_as_pixel_coordinate=True
    )

    assert result["position"] == (1, 1)


# no data handling
def test_nodata_is_masked_in_max():
    array = np.array([
        [1, -9999],
        [3, 4],
    ], dtype=np.float32)

    ds = EEORasterDataset.from_array(
        array=array,
        transform=Affine.identity(),
        crs=4326,
        nodata=-9999,
    )

    result = ds.get_maximum_pixel()

    assert result["value"] == 4.0


# chaining
def test_chainability(raster_from_array):
    result = (
        raster_from_array
        .get_maximum_pixel()
    )

    assert isinstance(result, dict)
    assert "value" in result

