import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from eeo import load_array
from eeo.core.adapters import RasterioAdapter
from eeo.viz import (
    plot_band_array,
    plot_composite,
    plot_histogram,
    plot_raster,
    plot_raster_with_histogram,
)
from eeo.viz.plot import (
    _DISPLAY_OVERSAMPLE,
    _as_list,
    _display_out_shape,
    _normalize_bands,
    _percentile_stretch,
    _read_band_for_display,
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
# Display-resolution (decimated read) helpers
# ---------------------------------------------------------------------


def test_display_out_shape_small_raster_reads_full():
    assert _display_out_shape((6, 6), (8, 8)) is None


def test_display_out_shape_caps_large_raster():
    dpi = plt.rcParams["figure.dpi"]
    out = _display_out_shape((100_000, 50_000), (10, 5))

    assert out is not None
    out_h, out_w = out
    # height-limited: budget is figsize height x dpi x oversample
    assert out_h == round(5 * dpi * _DISPLAY_OVERSAMPLE)
    # native 2:1 aspect ratio preserved
    assert out_w == round(out_h / 2)


def test_read_band_for_display_decimates_and_rescales_transform():
    ds = load_array(
        np.zeros((600, 600), dtype=np.float32),
        transform=Affine.translation(0, 600) * Affine.scale(1, -1),
        crs=CRS.from_epsg(32633),
    ).to_rasterio()

    array, transform = _read_band_for_display(ds, 1, (1, 1))

    expected = round(plt.rcParams["figure.dpi"] * _DISPLAY_OVERSAMPLE)
    assert array.shape == (expected, expected)
    # coarser pixels, unchanged extent
    assert transform.a == pytest.approx(600 / expected)
    assert transform.e == pytest.approx(-600 / expected)
    ds.close()


def test_read_band_for_display_numpy_backend_reads_full():
    ds = load_array(
        np.zeros((600, 600), dtype=np.float32),
        transform=Affine.translation(0, 600) * Affine.scale(1, -1),
        crs=CRS.from_epsg(32633),
    )

    array, transform = _read_band_for_display(ds, 1, (1, 1))

    assert array.shape == (600, 600)
    assert transform == ds.get_transform()


# ---------------------------------------------------------------------
# Large-raster plotting reads reduced arrays (end-to-end)
# ---------------------------------------------------------------------

LARGE_SIDE = 600


@pytest.fixture
def large_rgb_raster():
    """3-band 600x600 float32 raster, rasterio-backed.

    Large relative to the tiny ``figsize=(1, 1)`` display budget used in the
    decimation tests, so every band read must come back reduced.
    """
    ds = load_array(
        np.zeros((3, LARGE_SIDE, LARGE_SIDE), dtype=np.float32),
        transform=Affine.translation(0, LARGE_SIDE) * Affine.scale(1, -1),
        crs=CRS.from_epsg(32633),
    ).to_rasterio()
    yield ds
    ds.close()


def _record_pixel_read_shapes(monkeypatch):
    """Spy on both adapter read paths, recording each result's shape."""
    shapes = []
    real_read = RasterioAdapter.read
    real_read_band = RasterioAdapter.read_band

    def spy_read(self, *args, **kwargs):
        result = real_read(self, *args, **kwargs)
        shapes.append(result.shape)
        return result

    def spy_read_band(self, idx):
        result = real_read_band(self, idx)
        shapes.append(result.shape)
        return result

    monkeypatch.setattr(RasterioAdapter, "read", spy_read)
    monkeypatch.setattr(RasterioAdapter, "read_band", spy_read_band)
    return shapes


@pytest.mark.parametrize(
    "plot_func",
    [
        lambda ds: plot_raster(ds, figsize=(1, 1)),
        lambda ds: plot_band_array(ds, figsize=(1, 1)),
        lambda ds: plot_raster_with_histogram(ds, figsize=(1, 1)),
        lambda ds: plot_composite(ds, bands=(1, 2, 3), figsize=(1, 1)),
    ],
    ids=["plot_raster", "plot_band_array", "plot_raster_with_histogram", "plot_composite"],
)
def test_plotting_large_raster_reads_reduced_arrays(large_rgb_raster, plot_func, monkeypatch):
    shapes = _record_pixel_read_shapes(monkeypatch)

    # The deliberately tiny figure cannot fit all subplot decorations;
    # matplotlib's cosmetic tight_layout warning is irrelevant to the reads
    # under test.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Tight layout not applied", category=UserWarning)
        plot_func(large_rgb_raster)

    budget = round(plt.rcParams["figure.dpi"] * _DISPLAY_OVERSAMPLE)
    assert budget < LARGE_SIDE  # sanity: decimation must actually trigger
    assert shapes  # pixels were read through the adapter
    for shape in shapes:
        height, width = shape[-2:]
        assert height <= budget
        assert width <= budget


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
