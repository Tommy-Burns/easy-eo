import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from affine import Affine
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from rasterio.crs import CRS

from eeo import load_array
from eeo.core.adapters import RasterioAdapter
from eeo.viz import plot as plot_module
from eeo.viz import (
    plot_band_array,
    plot_composite,
    plot_histogram,
    plot_raster,
    plot_raster_with_histogram,
)
from eeo.viz.plot import (
    _DISPLAY_OVERSAMPLE,
    _add_colorbar,
    _as_list,
    _display_out_shape,
    _normalize_bands,
    _percentile_stretch,
    _read_band_for_display,
    _stretch_limits,
    _with_stretch_limits,
)

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
# General-purpose rasters come from conftest.py. Composite tests use a
# module-local float RGB fixture (values already in imshow's [0, 1] display
# range) for the no-stretch path; the stretch path is covered separately
# against the shared uint16 fixture.


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


@pytest.mark.parametrize(
    "func, expected",
    [
        (plot_band_array, True),
        (plot_raster, True),
        (plot_composite, True),
        (plot_raster_with_histogram, False),  # keeps raw histogram
    ],
    ids=["plot_band_array", "plot_raster", "plot_composite", "plot_raster_with_histogram"],
)
def test_stretch_default(func, expected):
    import inspect

    sig = inspect.signature(func)
    assert sig.parameters["stretch"].default is expected
    # The percentile bounds default to the conventional 2-98 everywhere.
    assert sig.parameters["pmin"].default == 2
    assert sig.parameters["pmax"].default == 98


def test_percentile_stretch_basic():
    arr = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    stretched = _percentile_stretch(arr, 0, 100)

    assert np.min(stretched) == 0.0
    assert np.max(stretched) == 1.0


def test_percentile_stretch_constant_array():
    arr = np.ones((5, 5), dtype=np.float32)
    stretched = _percentile_stretch(arr)

    assert np.all(stretched == 0)


def test_stretch_limits_basic():
    arr = np.array([0, 1, 2, 3, 4], dtype=np.float32)

    assert _stretch_limits(arr, 0, 100) == (0.0, 4.0)


def test_stretch_limits_constant_array():
    arr = np.ones((5, 5), dtype=np.float32)

    assert _stretch_limits(arr) is None


def test_stretch_limits_ignores_nan():
    arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0, np.nan], dtype=np.float32)

    assert _stretch_limits(arr, 0, 100) == (0.0, 4.0)


def test_stretch_limits_all_nan_band():
    """An all-nodata band has NaN percentiles; never hand those to Matplotlib."""
    arr = np.full((5, 5), np.nan, dtype=np.float32)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice", category=RuntimeWarning)
        assert _stretch_limits(arr) is None


def test_stretch_limits_match_percentile_stretch():
    """Display-time normalization must render exactly what rescaling did.

    ``Normalize(vmin, vmax)`` applies ``(x - low) / (high - low)`` — the same
    mapping as ``_percentile_stretch``, whose clipping ``Normalize`` leaves to
    the colormap (out-of-range values land on the under/over colours, which
    default to the colormap's own end colours). Proving both halves is what
    licenses the single-band plots to stretch by display limits instead of by
    rewriting the array: same pixels, real values retained.
    """
    # Outliers on both ends so the 2-98 percentiles actually clip.
    arr = np.concatenate([np.linspace(0.0, 1.0, 100), [-5.0, 7.0]]).astype(np.float32)

    limits = _stretch_limits(arr)
    assert limits is not None

    # Same linear mapping, once the clip Matplotlib defers is applied.
    np.testing.assert_allclose(
        np.asarray(Normalize(*limits, clip=True)(arr)),
        _percentile_stretch(arr),
        rtol=1e-6,
    )
    # And the colours actually rendered are identical, clip deferred or not.
    cmap = plt.get_cmap("gray")
    np.testing.assert_array_equal(
        cmap(Normalize(*limits)(arr)),
        cmap(_percentile_stretch(arr)),
    )


def test_with_stretch_limits_leaves_caller_kwargs_untouched():
    arr = np.linspace(0.0, 1.0, 25, dtype=np.float32).reshape(5, 5)
    caller_kwargs: dict = {}

    result = _with_stretch_limits(caller_kwargs, arr, 2, 98)

    assert caller_kwargs == {}  # per-band limits must not leak across subplots
    assert result["vmin"] == pytest.approx(np.nanpercentile(arr, 2))
    assert result["vmax"] == pytest.approx(np.nanpercentile(arr, 98))


def test_with_stretch_limits_respects_explicit_limits():
    arr = np.linspace(0.0, 1.0, 25, dtype=np.float32).reshape(5, 5)

    result = _with_stretch_limits({"vmin": -1.0, "vmax": 1.0}, arr, 2, 98)

    assert result == {"vmin": -1.0, "vmax": 1.0}


def test_with_stretch_limits_constant_array_sets_none():
    arr = np.ones((5, 5), dtype=np.float32)

    assert _with_stretch_limits({}, arr, 2, 98) == {}


# ---------------------------------------------------------------------
# Stretching normalizes at display time (data values survive)
# ---------------------------------------------------------------------


def _capture_imshow(monkeypatch):
    """Spy on ``Axes.imshow``, recording each drawn array and its limits.

    Catches both draw paths: ``plot_band_array`` calls ``imshow`` directly and
    ``rasterio.plot.show`` calls it underneath.
    """
    calls = []
    real_imshow = Axes.imshow

    def spy(self, arr, *args, **kwargs):
        image = real_imshow(self, arr, *args, **kwargs)
        calls.append((np.asarray(arr), image.get_clim()))
        return image

    monkeypatch.setattr(Axes, "imshow", spy)
    return calls


@pytest.fixture
def ndvi_like_raster():
    """Single-band float32 raster on an index-like scale, NumPy-backed."""
    array = np.linspace(-0.4, 0.9, 36, dtype=np.float32).reshape(6, 6)

    return load_array(
        array,
        transform=Affine.translation(0, 6) * Affine.scale(1, -1),
        crs=CRS.from_epsg(4326),
    )


@pytest.mark.parametrize(
    "plot_func",
    [plot_band_array, plot_raster],
    ids=["plot_band_array", "plot_raster"],
)
def test_stretch_sets_display_limits_without_rescaling(ndvi_like_raster, plot_func, monkeypatch):
    """A stretched plot draws the raw band, clipped by vmin/vmax."""
    expected = ndvi_like_raster.get_band(1)
    calls = _capture_imshow(monkeypatch)

    plot_func(ndvi_like_raster, stretch=True)

    drawn, clim = calls[0]
    np.testing.assert_array_equal(drawn, expected)  # values kept, not mapped to [0, 1]
    assert clim == pytest.approx(tuple(np.nanpercentile(expected, (2, 98))))


@pytest.mark.parametrize(
    "plot_func",
    [plot_band_array, plot_raster],
    ids=["plot_band_array", "plot_raster"],
)
def test_explicit_limits_win_over_stretch(ndvi_like_raster, plot_func, monkeypatch):
    """A caller's vmin/vmax must override the stretch, not collide with it."""
    calls = _capture_imshow(monkeypatch)

    plot_func(ndvi_like_raster, stretch=True, vmin=-1.0, vmax=1.0)

    assert calls[0][1] == (-1.0, 1.0)


def test_stretch_keeps_nodata_unpainted_when_range_is_empty(monkeypatch):
    """A degenerate stretch must not paint NaN pixels as real values.

    Rescaling returned ``zeros_like`` for an empty percentile range, turning
    every NaN into a 0 and rendering nodata as the colormap's low end — the one
    case where the old and new stretch draw different pixels. Display limits
    leave NaN as NaN, so nodata stays blank.
    """
    array = np.full((6, 6), np.nan, dtype=np.float32)
    array[0, 0] = 1.0  # single valid pixel: p2 == p98, so no limits apply
    ds = load_array(
        array,
        transform=Affine.translation(0, 6) * Affine.scale(1, -1),
        crs=CRS.from_epsg(4326),
    )
    calls = _capture_imshow(monkeypatch)

    plot_band_array(ds, stretch=True)

    drawn, _ = calls[0]
    assert np.count_nonzero(np.isnan(drawn)) == array.size - 1


def test_stretch_limits_are_computed_per_band(monkeypatch):
    """Bands on different scales each get their own limits.

    Regression guard for reusing one kwargs dict across subplots, which would
    freeze the first band's limits onto every later one.
    """
    band1 = np.linspace(0.0, 1.0, 36, dtype=np.float32).reshape(6, 6)
    band2 = band1 * 10.0 + 100.0
    ds = load_array(
        np.stack([band1, band2]),
        transform=Affine.translation(0, 6) * Affine.scale(1, -1),
        crs=CRS.from_epsg(4326),
    )
    calls = _capture_imshow(monkeypatch)

    plot_band_array(ds, stretch=True)

    assert len(calls) == 2
    assert calls[0][1] == pytest.approx(tuple(np.nanpercentile(band1, (2, 98))))
    assert calls[1][1] == pytest.approx(tuple(np.nanpercentile(band2, (2, 98))))


def test_raster_with_histogram_bins_raw_values_when_stretched(ndvi_like_raster, monkeypatch):
    """The stretch scales the image panel only; the histogram stays in band units."""
    expected = ndvi_like_raster.get_band(1)
    hist_data = []
    real_hist = Axes.hist

    def spy_hist(self, data, *args, **kwargs):
        hist_data.append(np.asarray(data))
        return real_hist(self, data, *args, **kwargs)

    monkeypatch.setattr(Axes, "hist", spy_hist)
    calls = _capture_imshow(monkeypatch)

    plot_raster_with_histogram(ndvi_like_raster, stretch=True)

    np.testing.assert_array_equal(hist_data[0], expected.ravel())
    assert calls[0][1] == pytest.approx(tuple(np.nanpercentile(expected, (2, 98))))


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
    decimation tests, so every band read must come back reduced. The bands
    must not be constant: rasterio >= 1.5 normalizes float bands in
    ``show()`` by their value range, and a zero range divides 0/0 and warns.
    """
    base = np.linspace(0.0, 1.0, LARGE_SIDE * LARGE_SIDE, dtype=np.float32).reshape(
        LARGE_SIDE, LARGE_SIDE
    )
    ds = load_array(
        np.stack([base, 0.5 * base, 0.25 * base]),
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
# Colorbars
# ---------------------------------------------------------------------


def _capture_colorbars(monkeypatch):
    """Spy on ``Figure.colorbar``, recording each colorbar drawn."""
    bars = []
    real_colorbar = Figure.colorbar

    def spy(self, mappable, **kwargs):
        bar = real_colorbar(self, mappable, **kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setattr(Figure, "colorbar", spy)
    return bars


COLORBAR_FUNCS = [plot_band_array, plot_raster, plot_raster_with_histogram]
COLORBAR_IDS = ["plot_band_array", "plot_raster", "plot_raster_with_histogram"]


@pytest.mark.parametrize("plot_func", COLORBAR_FUNCS, ids=COLORBAR_IDS)
def test_colorbar_off_by_default(ndvi_like_raster, plot_func, monkeypatch):
    """Existing calls must render exactly as before, colorbar or not."""
    bars = _capture_colorbars(monkeypatch)

    plot_func(ndvi_like_raster)

    assert bars == []


@pytest.mark.parametrize("plot_func", COLORBAR_FUNCS, ids=COLORBAR_IDS)
def test_colorbar_spans_band_values(ndvi_like_raster, plot_func, monkeypatch):
    """The bar reports index units, not the 0-1 display scale."""
    bars = _capture_colorbars(monkeypatch)

    plot_func(ndvi_like_raster, stretch=True, colorbar=True)

    assert len(bars) == 1
    band = ndvi_like_raster.get_band(1)
    assert bars[0].mappable.get_clim() == pytest.approx(tuple(np.nanpercentile(band, (2, 98))))


@pytest.mark.parametrize("plot_func", COLORBAR_FUNCS, ids=COLORBAR_IDS)
def test_colorbar_labelled_from_band_name(plot_func, monkeypatch):
    """A named band labels its own colorbar; ``colorbar_label`` overrides it."""
    ds = load_array(
        np.linspace(-0.4, 0.9, 36, dtype=np.float32).reshape(6, 6),
        transform=Affine.translation(0, 6) * Affine.scale(1, -1),
        crs=CRS.from_epsg(4326),
        band_names=["NDVI"],
    )
    bars = _capture_colorbars(monkeypatch)

    plot_func(ds, colorbar=True)
    assert bars[-1].ax.get_ylabel() == "NDVI"

    plot_func(ds, colorbar=True, colorbar_label="vegetation index")
    assert bars[-1].ax.get_ylabel() == "vegetation index"


@pytest.mark.parametrize("plot_func", COLORBAR_FUNCS, ids=COLORBAR_IDS)
def test_colorbar_unlabelled_for_unnamed_band(ndvi_like_raster, plot_func, monkeypatch):
    bars = _capture_colorbars(monkeypatch)

    plot_func(ndvi_like_raster, colorbar=True)

    assert bars[0].ax.get_ylabel() == ""


def test_colorbar_extend_marks_clipped_ends(ndvi_like_raster, monkeypatch):
    """Arrowheads must show which ends of the scale the stretch clips."""
    bars = _capture_colorbars(monkeypatch)

    # 2-98 percentiles clip both tails of a ramp.
    plot_raster(ndvi_like_raster, stretch=True, colorbar=True)
    assert bars[-1].extend == "both"

    # Raw display autoscales to the data, so nothing is clipped.
    plot_raster(ndvi_like_raster, stretch=False, colorbar=True)
    assert bars[-1].extend == "neither"


def test_colorbar_extend_one_sided(monkeypatch):
    """Only the clipped end gets an arrowhead."""
    array = np.linspace(0.0, 1.0, 36, dtype=np.float32).reshape(6, 6)
    ds = load_array(
        array,
        transform=Affine.translation(0, 6) * Affine.scale(1, -1),
        crs=CRS.from_epsg(4326),
    )
    bars = _capture_colorbars(monkeypatch)

    plot_band_array(ds, colorbar=True, vmin=float(array.min()), vmax=0.5)

    assert bars[-1].extend == "max"


def test_colorbar_per_band_in_a_grid(monkeypatch):
    """Bands on different scales each get their own bar, not a shared one."""
    band1 = np.linspace(0.0, 1.0, 36, dtype=np.float32).reshape(6, 6)
    band2 = band1 * 10.0 + 100.0
    ds = load_array(
        np.stack([band1, band2]),
        transform=Affine.translation(0, 6) * Affine.scale(1, -1),
        crs=CRS.from_epsg(4326),
    )
    bars = _capture_colorbars(monkeypatch)

    plot_band_array(ds, colorbar=True, stretch=True)

    assert len(bars) == 2
    assert bars[0].mappable.get_clim() == pytest.approx(tuple(np.nanpercentile(band1, (2, 98))))
    assert bars[1].mappable.get_clim() == pytest.approx(tuple(np.nanpercentile(band2, (2, 98))))


def test_add_colorbar_without_a_mappable_draws_nothing(ndvi_like_raster, monkeypatch):
    """No image on the axes means nothing to describe — and no crash."""
    bars = _capture_colorbars(monkeypatch)
    fig, ax = plt.subplots()

    _add_colorbar(fig, ax, None, ndvi_like_raster, 1, None)

    assert bars == []
    plt.close(fig)


def test_colorbar_skipped_when_no_image_drawn(ndvi_like_raster, monkeypatch):
    """`plot_raster` must survive a draw that leaves the axes imageless.

    Reached in practice by passing ``contour=True`` through to
    ``rasterio.plot.show``, which draws contour lines instead of an image;
    stubbed here so the test does not depend on contour rendering. An
    unguarded ``ax.get_images()[-1]`` would raise IndexError in user code.
    """
    monkeypatch.setattr(plot_module.rioplot, "show", lambda *args, **kwargs: None)
    bars = _capture_colorbars(monkeypatch)

    plot_raster(ndvi_like_raster, colorbar=True)

    assert bars == []


def test_colorbar_all_nodata_band(monkeypatch):
    """An all-NaN band has no finite values; the bar falls back to no arrowheads."""
    ds = load_array(
        np.full((6, 6), np.nan, dtype=np.float32),
        transform=Affine.translation(0, 6) * Affine.scale(1, -1),
        crs=CRS.from_epsg(4326),
    )
    bars = _capture_colorbars(monkeypatch)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice", category=RuntimeWarning)
        plot_band_array(ds, colorbar=True, stretch=True)

    assert bars[-1].extend == "neither"


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


def test_plot_composite_stretch_integer_not_truncated(multiband_uint16, monkeypatch):
    """Regression: stretching an integer raster must not render black.

    The stretched channels are floats in ``[0, 1]``; the composite must stay
    floating rather than write them back into the uint16 band dtype, which
    would floor every value below 1 to 0 and produce a black image.
    """
    captured = {}
    monkeypatch.setattr(plt, "imshow", lambda arr, *a, **k: captured.update(arr=np.asarray(arr)))

    plot_composite(multiband_uint16, bands=(3, 2, 1), stretch=True)

    arr = captured["arr"]
    assert np.issubdtype(arr.dtype, np.floating)  # not truncated to the band dtype
    assert arr.min() >= 0.0 and arr.max() <= 1.0
    assert arr.max() > 0.5  # real contrast survives; the image is not black
    assert np.count_nonzero(arr) > arr.size // 2
