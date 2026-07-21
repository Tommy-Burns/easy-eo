import math

import geopandas as gpd
import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform
from shapely.geometry import box

from eeo.core.exceptions import ValidationError
from eeo.preprocessing import (
    clip_raster_with_bbox,
    clip_raster_with_vector,
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


# ---------------------------------------------------------------------
# Reproject
# ---------------------------------------------------------------------


def test_reproject_nonsquare_raster_uses_correct_dimensions(nonsquare_float32):
    # A prior version passed the raster's width as the source height when
    # computing the destination grid, distorting the output resolution and
    # shape for any non-square raster (square fixtures hid it).
    target_crs = CRS.from_epsg(4326)
    left, bottom, right, top = nonsquare_float32.get_bounds()
    expected_transform, expected_width, expected_height = calculate_default_transform(
        src_crs=nonsquare_float32.get_crs(),
        dst_crs=target_crs,
        width=nonsquare_float32.get_width(),
        height=nonsquare_float32.get_height(),
        left=left,
        bottom=bottom,
        right=right,
        top=top,
    )

    reprojected = nonsquare_float32.reproject_raster(target_crs=4326)

    assert reprojected.get_width() == expected_width
    assert reprojected.get_height() == expected_height
    assert reprojected.get_transform() == expected_transform
    # the 4x8 source must stay wider than it is tall
    assert reprojected.get_width() > reprojected.get_height()


# ---------------------------------------------------------------------
# Nodata & dtype contract
# ---------------------------------------------------------------------


def test_standardize_outputs_float32_and_excludes_nodata(raster_with_nodata):
    out = standardize(raster_with_nodata)
    data = out.read()

    assert data.dtype == np.float32
    assert math.isnan(out.get_metadata()["nodata"])
    assert np.isnan(data[0][:2, :2]).all()  # nodata block
    # statistics computed over valid pixels only
    np.testing.assert_allclose(np.nanmean(data), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.nanstd(data), 1.0, atol=1e-6)


def test_normalize_min_max_outputs_float32_on_integer_input(multiband_uint16):
    out = normalize_min_max(multiband_uint16)
    data = out.read()

    assert data.dtype == np.float32  # not truncated to uint16 0/1
    assert data.min() == pytest.approx(0.0)
    assert data.max() == pytest.approx(1.0)


def test_normalize_min_max_excludes_nodata(raster_with_nodata):
    out = normalize_min_max(raster_with_nodata)
    data = out.read()[0]

    assert out.read().dtype == np.float32
    assert math.isnan(out.get_metadata()["nodata"])
    assert np.isnan(data[:2, :2]).all()
    # valid range (values 2..35) maps onto [0, 1]
    assert np.nanmin(data) == pytest.approx(0.0)
    assert np.nanmax(data) == pytest.approx(1.0)


def test_normalize_percentile_excludes_nodata(raster_with_nodata):
    out = normalize_percentile(raster_with_nodata, lower_percentile=0, upper_percentile=100)
    data = out.read()[0]

    assert out.read().dtype == np.float32
    assert math.isnan(out.get_metadata()["nodata"])
    assert np.isnan(data[:2, :2]).all()
    assert np.nanmin(data) == pytest.approx(0.0)
    assert np.nanmax(data) == pytest.approx(1.0)


def test_resample_preserves_nodata_and_dtype(raster_with_nodata):
    resampled = resample(raster_with_nodata, scale_factor=0.5)

    assert resampled.read().dtype == np.float32
    assert resampled.get_metadata()["nodata"] == -9999.0
    # nearest resampling copies the sentinel through rather than blending it
    assert np.any(resampled.read() == -9999.0)


def test_reproject_preserves_nodata(raster_with_nodata):
    reprojected = raster_with_nodata.reproject_raster(target_crs=4326)

    assert reprojected.read().dtype == np.float32
    assert reprojected.get_metadata()["nodata"] == -9999.0
    # source nodata and warp-exposed borders are the nodata value, not 0
    assert np.any(reprojected.read() == -9999.0)


def test_clip_bbox_preserves_nodata_and_dtype(raster_with_nodata):
    # a window covering the top-left, which holds the nodata block
    clipped = clip_raster_with_bbox(
        raster_with_nodata, (500_000.0, 4_199_970.0, 500_040.0, 4_200_000.0)
    )

    assert clipped.read().dtype == np.float32
    assert clipped.get_metadata()["nodata"] == -9999.0
    assert np.any(clipped.read() == -9999.0)


def test_clip_vector_preserves_nodata_and_dtype(raster_with_nodata):
    gdf = gpd.GeoDataFrame(
        geometry=[box(500_000.0, 4_199_970.0, 500_040.0, 4_200_000.0)],
        crs=raster_with_nodata.get_crs(),
    )
    clipped = clip_raster_with_vector(raster_with_nodata, gdf)

    assert clipped.read().dtype == np.float32
    assert clipped.get_metadata()["nodata"] == -9999.0
