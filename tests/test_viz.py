import numpy as np
import pytest
import matplotlib

# Use non-interactive backend for tests
matplotlib.use("Agg")

from affine import Affine
from rasterio.crs import CRS

from eeo.core.core import EEORasterDataset
from eeo import load_array

from eeo.viz.plot import (
    _as_list,
    _normalize_bands,
    _percentile_stretch,
)
from eeo.viz import (
    plot_band_array,
    plot_raster,
    plot_histogram,
    plot_raster_with_histogram,
    plot_composite,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def single_band_raster():
    array = np.array([[1, 2], [3, 4]], dtype=np.float32)
    return load_array(
        array,
        transform=Affine.identity(),
        crs=CRS.from_epsg(4326),
    )


@pytest.fixture
def multi_band_raster():
    array = np.stack(
        [
            [[1, 2], [3, 4]],
            [[10, 20], [30, 40]],
            [[100, 200], [300, 400]],
        ]
    ).astype(np.float32)

    return load_array(
        array,
        transform=Affine.identity(),
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


def test_normalize_bands_none(multi_band_raster):
    bands = _normalize_bands(multi_band_raster, None)
    assert bands == [1, 2, 3]


def test_normalize_bands_single_int(multi_band_raster):
    bands = _normalize_bands(multi_band_raster, 2)
    assert bands == [2]


def test_normalize_bands_iterable(multi_band_raster):
    bands = _normalize_bands(multi_band_raster, [1, 3])
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

def test_plot_band_array_single(single_band_raster):
    plot_band_array(single_band_raster)


def test_plot_band_array_multiple_bands(multi_band_raster):
    plot_band_array(multi_band_raster, bands=[1, 2], stretch=True)


def test_plot_band_array_multiple_datasets(single_band_raster):
    plot_band_array([single_band_raster, single_band_raster])


def test_plot_raster_basic(single_band_raster):
    plot_raster(single_band_raster)


def test_plot_raster_stretch(multi_band_raster):
    plot_raster(multi_band_raster, bands=1, stretch=True)


def test_plot_histogram_basic(single_band_raster):
    plot_histogram(single_band_raster)


def test_plot_histogram_log_scale(single_band_raster):
    plot_histogram(single_band_raster, log=True)


def test_plot_raster_with_histogram(single_band_raster):
    plot_raster_with_histogram(single_band_raster)


def test_plot_raster_with_histogram_multiple_bands(multi_band_raster):
    plot_raster_with_histogram(multi_band_raster, bands=[1, 2])


def test_plot_composite_rgb(multi_band_raster):
    plot_composite(multi_band_raster, bands=(1, 2, 3))


def test_plot_composite_stretched(multi_band_raster):
    plot_composite(multi_band_raster, bands=(1, 2, 3), stretch=True)
