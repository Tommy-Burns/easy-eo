import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from eeo import load_array
from eeo.viz import (
    plot_band_array,
    plot_composite,
    plot_histogram,
    plot_raster,
    plot_raster_with_histogram,
)
from eeo.viz.plot import (
    _as_list,
    _normalize_bands,
    _percentile_stretch,
)

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
# General-purpose rasters come from conftest.py. Composite tests need a
# module-local fixture: plot_composite feeds bands straight to imshow and
# stretches in place, so it needs float RGB data already in display range.


@pytest.fixture
def rgb_float32_raster():
    """3-band float32 raster with values in [0, 1], NumPy-backed."""
    base = np.linspace(0.0, 1.0, 36, dtype=np.float32).reshape(6, 6)
    array = np.stack([base, 0.5 * base, 0.25 * base])

    return load_array(
        array,
        transform=Affine.translation(0, 6) * Affine.scale(1, -1),
        crs=CRS.from_epsg(4326),
    )


# ---------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------


def test_as_list_single_object():
    obj = 1
    result = _as_list(obj)
    assert result == [1]


def test_as_list_list_passthrough():
    obj = [1, 2, 3]
    result = _as_list(obj)
    assert result is obj


def test_normalize_bands_none(multiband_uint16):
    bands = _normalize_bands(multiband_uint16, None)
    assert bands == [1, 2, 3, 4]


def test_normalize_bands_single_int(multiband_uint16):
    bands = _normalize_bands(multiband_uint16, 2)
    assert bands == [2]


def test_normalize_bands_iterable(multiband_uint16):
    bands = _normalize_bands(multiband_uint16, [1, 3])
    assert bands == [1, 3]


def test_percentile_stretch_basic():
    arr = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    stretched = _percentile_stretch(arr, 0, 100)

    assert np.min(stretched) == 0.0
    assert np.max(stretched) == 1.0


def test_percentile_stretch_constant_array():
    arr = np.ones((5, 5), dtype=np.float32)
    stretched = _percentile_stretch(arr)

    assert np.all(stretched == 0)


# ---------------------------------------------------------------------
# Plotting functions (smoke tests)
# ---------------------------------------------------------------------


def test_plot_band_array_single(single_band_float32):
    plot_band_array(single_band_float32)


def test_plot_band_array_multiple_bands(multiband_uint16):
    plot_band_array(multiband_uint16, bands=[1, 2], stretch=True)


def test_plot_band_array_multiple_datasets(single_band_float32):
    plot_band_array([single_band_float32, single_band_float32])


def test_plot_raster_basic(single_band_float32):
    plot_raster(single_band_float32)


def test_plot_raster_stretch(multiband_uint16):
    plot_raster(multiband_uint16, bands=1, stretch=True)


def test_plot_histogram_basic(single_band_float32):
    plot_histogram(single_band_float32)


def test_plot_histogram_log_scale(single_band_float32):
    plot_histogram(single_band_float32, log=True)


def test_plot_raster_with_histogram(single_band_float32):
    plot_raster_with_histogram(single_band_float32)


def test_plot_raster_with_histogram_multiple_bands(multiband_uint16):
    plot_raster_with_histogram(multiband_uint16, bands=[1, 2])


def test_plot_composite_rgb(rgb_float32_raster):
    plot_composite(rgb_float32_raster, bands=(1, 2, 3))


def test_plot_composite_stretched(rgb_float32_raster):
    plot_composite(rgb_float32_raster, bands=(1, 2, 3), stretch=True)
