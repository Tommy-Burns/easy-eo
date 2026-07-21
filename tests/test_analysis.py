import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from eeo import load_array
from eeo.core.exceptions import ValidationError


# normalized difference
def test_normalized_difference_basic(raster_3x3):
    result = raster_3x3.normalized_difference(raster_3x3)

    # Should return EEORasterDataset
    assert hasattr(result, "read")

    data = result.read()
    assert np.allclose(data, 0.0)


# ndarray returns
def test_normalized_difference_returns_ndarray(raster_3x3):
    nd = raster_3x3.normalized_difference(raster_3x3, return_as_ndarray=True)

    assert isinstance(nd, np.ndarray)
    assert nd.shape == raster_3x3.read().shape


# divide-by-zero safety
def test_normalized_difference_divide_by_zero():
    zeros = np.zeros((3, 3), dtype=np.float32)
    transform = Affine.translation(0, 3) * Affine.scale(1, -1)
    crs = CRS.from_epsg(4326)

    ds1 = load_array(zeros, transform=transform, crs=crs)
    ds2 = load_array(zeros, transform=transform, crs=crs)

    nd = ds1.normalized_difference(ds2, return_as_ndarray=True)

    assert np.all(nd == 0)


# extract value at coordinate
def test_extract_value_at_coordinate(raster_3x3):
    # unit-pixel grid with top-left origin (0, 3): world (1.5, 1.5) is pixel (1, 1)
    value = raster_3x3.extract_value_at_coordinate((1.5, 1.5))

    assert value == 5


# invalid coordinate length
def test_extract_value_invalid_coordinates(raster_3x3):
    with pytest.raises(ValidationError):
        raster_3x3.extract_value_at_coordinate((1,))


# Maximum pixel
def test_get_maximum_pixel_value_and_position(raster_3x3):
    result = raster_3x3.get_maximum_pixel()

    assert result["value"] == 9.0
    assert isinstance(result["position"], tuple)


# Maximum pixel - pixel coords
def test_get_maximum_pixel_pixel_coordinates(raster_3x3):
    result = raster_3x3.get_maximum_pixel(return_position_as_pixel_coordinate=True)

    assert result["position"] == (2, 2)


# Minimum pixel
def test_get_minimum_pixel_value_and_position(raster_3x3):
    result = raster_3x3.get_minimum_pixel()

    assert result["value"] == 1.0


# Mean pixel
def test_get_mean_pixel(raster_3x3):
    result = raster_3x3.get_mean_pixel()

    assert result["value"] == 5.0


def test_get_mean_pixel_location_pixel_space(raster_3x3):
    result = raster_3x3.get_mean_pixel(return_position_as_pixel_coordinate=True)

    assert result["position"] == (1, 1)


# Percentile pixel
def test_get_percentile_pixel(raster_3x3):
    result = raster_3x3.get_percentile_pixel(50)

    assert result["value"] == 5.0


def test_get_percentile_pixel_pixel_coordinates(raster_3x3):
    result = raster_3x3.get_percentile_pixel(50, return_position_as_pixel_coordinate=True)

    assert result["position"] == (1, 1)


# multi-band: locators operate on the selected band (default: band 1)
def test_get_maximum_pixel_multiband_default_first_band(multiband_uint16):
    result = multiband_uint16.get_maximum_pixel(return_position_as_pixel_coordinate=True)

    assert result["value"] == 1035.0
    assert result["position"] == (5, 5)


def test_get_maximum_pixel_multiband_band_override(multiband_uint16):
    result = multiband_uint16.get_maximum_pixel(
        band_idx=2, return_position_as_pixel_coordinate=True
    )

    assert result["value"] == 2035.0
    assert result["position"] == (5, 5)


def test_get_minimum_pixel_multiband_band_override(multiband_uint16):
    result = multiband_uint16.get_minimum_pixel(
        band_idx=3, return_position_as_pixel_coordinate=True
    )

    assert result["value"] == 3000.0
    assert result["position"] == (0, 0)


def test_get_mean_pixel_multiband_band_override(multiband_uint16):
    result = multiband_uint16.get_mean_pixel(band_idx=2, return_position_as_pixel_coordinate=True)

    # band 2 mean is 2017.5; the first pixel nearest it is 2017 at (2, 5)
    assert result["value"] == 2017.5
    assert result["position"] == (2, 5)


def test_get_percentile_pixel_multiband_band_override(multiband_uint16):
    result = multiband_uint16.get_percentile_pixel(50, band_idx=4)

    assert result["value"] == 4017.5


def test_extract_value_at_coordinate_multiband_band_override(multiband_uint16):
    # 10 m UTM grid, origin (500000, 4200000): world (500015, 4199985) is
    # pixel (1, 1) = gradient value 7
    value = multiband_uint16.extract_value_at_coordinate((500_015.0, 4_199_985.0), band_idx=2)

    assert value == 2007


def test_stats_locators_reject_out_of_range_band(multiband_uint16):
    with pytest.raises(IndexError):
        multiband_uint16.get_maximum_pixel(band_idx=9)


# no data handling
def test_nodata_is_masked_in_stats(raster_with_nodata):
    # nodata (-9999) occupies the top-left 2x2 block of the 0..35 gradient;
    # stats must skip it: max stays 35, min is 2 (0, 1, 6, 7 are nodata)
    assert raster_with_nodata.get_maximum_pixel()["value"] == 35.0
    assert raster_with_nodata.get_minimum_pixel()["value"] == 2.0


# chaining
def test_chainability(raster_3x3):
    result = raster_3x3.get_maximum_pixel()

    assert isinstance(result, dict)
    assert "value" in result
