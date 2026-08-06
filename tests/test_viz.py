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
from eeo.core.exceptions import ValidationError
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
    _auto_grid,
    _display_out_shape,
    _grid_shape,
    _mask_nodata_for_display,
    _normalize_bands,
    _percentile_stretch,
    _read_band_for_display,
    _resolve_figsize,
    _stretch_limits,
    _valid_values,
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
        # Recorded as passed, not via np.asarray: that would strip the nodata
        # mask the plots rely on.
        calls.append((arr, image.get_clim()))
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
# Subplot grid (nrows / ncols)
# ---------------------------------------------------------------------

GRID_FUNCS = [plot_band_array, plot_raster, plot_histogram]
GRID_IDS = ["plot_band_array", "plot_raster", "plot_histogram"]


def _grid_of(plot_func, ds, monkeypatch, **kwargs):
    """Run a plot and report the (nrows, ncols) it asked ``subplots`` for."""
    captured = {}
    real_subplots = plt.subplots

    def spy(*args, **kw):
        captured["shape"] = args[:2]
        return real_subplots(*args, **kw)

    monkeypatch.setattr(plt, "subplots", spy)
    plot_func(ds, **kwargs)
    return captured["shape"]


def test_grid_shape_derives_rows_from_ncols():
    assert _grid_shape(4, None, 2) == (2, 2)
    assert _grid_shape(5, None, 2) == (3, 2)  # ceil, so nothing is dropped


def test_grid_shape_derives_cols_from_nrows():
    assert _grid_shape(4, 2, None) == (2, 2)
    assert _grid_shape(5, 2, None) == (2, 3)


def test_grid_shape_accepts_an_explicit_pair():
    assert _grid_shape(4, 2, 3) == (2, 3)  # room to spare is fine


def test_grid_shape_rejects_a_grid_that_drops_panels():
    with pytest.raises(ValidationError, match="holds 3 panels, but 4"):
        _grid_shape(4, 1, 3)


@pytest.mark.parametrize("bad", [0, -1, 2.5, "2"], ids=["zero", "negative", "float", "str"])
def test_grid_shape_rejects_non_positive_integers(bad):
    with pytest.raises(ValidationError, match="positive integer"):
        _grid_shape(4, None, bad)


@pytest.mark.parametrize(
    "n_panels, expected",
    [
        (1, (1, 1)),
        (2, (1, 2)),  # small counts stay a single row
        (3, (1, 3)),
        (4, (2, 2)),  # the reported case
        (5, (2, 3)),  # one blank cell
        (6, (2, 3)),
        (8, (2, 4)),
        (9, (3, 3)),
        (12, (3, 4)),
    ],
)
def test_auto_grid_is_near_square_and_landscape(n_panels, expected):
    """Never taller than wide, and never wasting more than one row."""
    rows, cols = _auto_grid(n_panels)

    assert (rows, cols) == expected
    assert rows * cols >= n_panels  # every panel has a cell
    assert rows <= cols  # landscape, matching the default figure sizes


@pytest.mark.parametrize("plot_func", GRID_FUNCS, ids=GRID_IDS)
def test_default_layout_is_near_square(multiband_uint16, plot_func, monkeypatch):
    """A 4-band raster defaults to 2x2, not a 4x1 strip."""
    assert _grid_of(plot_func, multiband_uint16, monkeypatch) == (2, 2)


@pytest.mark.parametrize("plot_func", GRID_FUNCS, ids=GRID_IDS)
def test_default_layout_for_a_list_of_datasets(single_band_float32, plot_func, monkeypatch):
    """Four single-band datasets default to 2x2, not a 1x4 strip."""
    datasets = [single_band_float32] * 4

    assert _grid_of(plot_func, datasets, monkeypatch) == (2, 2)


@pytest.mark.parametrize("plot_func", GRID_FUNCS, ids=GRID_IDS)
def test_single_panel_layout_is_unchanged(single_band_float32, plot_func, monkeypatch):
    assert _grid_of(plot_func, single_band_float32, monkeypatch) == (1, 1)


def test_resolve_figsize_honours_an_explicit_request():
    assert _resolve_figsize((3, 4), (10, 5), 2, 2) == (3, 4)


def test_resolve_figsize_keeps_the_base_for_a_single_row():
    """One row of panels is what the per-function defaults were chosen for."""
    assert _resolve_figsize(None, (10, 5), 1, 3) == (10, 5)


@pytest.mark.parametrize(
    "base, rows, cols, expected",
    [
        ((10, 5), 2, 2, (10, 10)),  # square cells, not 2:1 letterboxes
        ((8, 8), 2, 2, (8, 8)),  # already square: unchanged
        ((10, 5), 2, 4, (10, 5)),  # a wide grid needs no extra height
        ((10, 5), 3, 3, (10, 10)),
    ],
)
def test_resolve_figsize_squares_up_taller_grids(base, rows, cols, expected):
    width, height = _resolve_figsize(None, base, rows, cols)

    assert (width, height) == expected
    # Cell aspect ratio is 1:1 — the point of deriving at all.
    assert (width / cols) / (height / rows) == pytest.approx(1.0)


@pytest.mark.parametrize("plot_func", GRID_FUNCS, ids=GRID_IDS)
def test_default_figsize_grows_with_the_grid(multiband_uint16, plot_func, monkeypatch):
    """A 2x2 of square maps must not be squeezed into the single-row default."""
    captured = {}
    real_subplots = plt.subplots

    def spy(*args, **kwargs):
        captured["figsize"] = kwargs.get("figsize")
        return real_subplots(*args, **kwargs)

    monkeypatch.setattr(plt, "subplots", spy)

    plot_func(multiband_uint16)  # 4 bands -> 2x2

    width, height = captured["figsize"]
    assert height > width / 2  # taller than the (w, w/2) single-row defaults


@pytest.mark.parametrize("plot_func", GRID_FUNCS, ids=GRID_IDS)
def test_explicit_figsize_is_never_overridden(multiband_uint16, plot_func, monkeypatch):
    captured = {}
    real_subplots = plt.subplots

    def spy(*args, **kwargs):
        captured["figsize"] = kwargs.get("figsize")
        return real_subplots(*args, **kwargs)

    monkeypatch.setattr(plt, "subplots", spy)

    plot_func(multiband_uint16, figsize=(7, 3))

    assert captured["figsize"] == (7, 3)


@pytest.mark.parametrize("plot_func", GRID_FUNCS, ids=GRID_IDS)
def test_ncols_reflows_a_multiband_raster(multiband_uint16, plot_func, monkeypatch):
    """The reported case: a 4-band raster as 2x2 instead of a 4x1 strip."""
    assert _grid_of(plot_func, multiband_uint16, monkeypatch, ncols=2) == (2, 2)


@pytest.mark.parametrize("plot_func", GRID_FUNCS, ids=GRID_IDS)
def test_nrows_reflows_a_list_of_datasets(single_band_float32, plot_func, monkeypatch):
    """The other reported case: four datasets as 2x2 instead of a 1x4 strip."""
    datasets = [single_band_float32] * 4

    assert _grid_of(plot_func, datasets, monkeypatch, nrows=2) == (2, 2)


@pytest.mark.parametrize("plot_func", GRID_FUNCS, ids=GRID_IDS)
def test_leftover_cells_are_hidden(multiband_uint16, plot_func, monkeypatch):
    """A 2x3 grid holding 4 panels must not show two empty framed axes."""
    fig_axes = []
    real_subplots = plt.subplots

    def spy(*args, **kw):
        fig, axes = real_subplots(*args, **kw)
        fig_axes.append(axes)
        return fig, axes

    monkeypatch.setattr(plt, "subplots", spy)

    plot_func(multiband_uint16, nrows=2, ncols=3)

    flat = fig_axes[0].ravel()
    assert [ax.axison for ax in flat[4:]] == [False, False]


@pytest.mark.parametrize("plot_func", GRID_FUNCS, ids=GRID_IDS)
def test_grid_too_small_raises(multiband_uint16, plot_func):
    with pytest.raises(ValidationError, match="increase nrows or ncols"):
        plot_func(multiband_uint16, nrows=1, ncols=2)


def test_reflow_fill_order_is_dataset_major(multiband_uint16, monkeypatch):
    """Every band of the first dataset, then every band of the next."""
    second = load_array(
        np.zeros((2, 6, 6), dtype=np.float32) + 5.0,
        transform=Affine.translation(0, 6) * Affine.scale(1, -1),
        crs=CRS.from_epsg(4326),
    )
    titles = []
    real_set_title = Axes.set_title

    def spy(self, label, *args, **kwargs):
        titles.append(label)
        return real_set_title(self, label, *args, **kwargs)

    monkeypatch.setattr(Axes, "set_title", spy)

    plot_band_array([multiband_uint16, second], bands=[1, 2], ncols=2)

    assert titles == ["Band 1", "Band 2", "Band 1", "Band 2"]


def test_semantic_layout_survives_a_two_dimensional_selection(multiband_uint16, monkeypatch):
    """Datasets x bands keeps its meaning when no grid is requested.

    Rows are bands and columns are datasets, so band i of one dataset sits
    beside band i of the other; reflowing that would destroy the comparison.
    """
    shape = _grid_of(
        plot_band_array, [multiband_uint16, multiband_uint16], monkeypatch, bands=[1, 2, 3]
    )

    assert shape == (3, 2)


# ---------------------------------------------------------------------
# Nodata is absent from the display (nodata contract)
# ---------------------------------------------------------------------
# The `raster_with_nodata` fixture is a 6x6 float32 gradient whose top-left
# 2x2 block is the -9999 sentinel: 4 nodata pixels, 32 valid ones.


def test_mask_nodata_for_display_masks_the_sentinel(raster_with_nodata):
    masked = _mask_nodata_for_display(raster_with_nodata, raster_with_nodata.get_band(1))

    assert np.ma.isMaskedArray(masked)
    assert masked.count() == 32  # the 4 sentinel pixels are gone
    assert -9999.0 not in _valid_values(masked)


def test_mask_nodata_for_display_no_nodata_declared(single_band_float32):
    band = single_band_float32.get_band(1)

    assert _mask_nodata_for_display(single_band_float32, band) is band


def test_mask_nodata_for_display_nan_nodata_needs_no_mask():
    """A float raster's nodata is already NaN, which the percentiles ignore."""
    array = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    ds = load_array(
        array,
        transform=Affine.translation(0, 2) * Affine.scale(1, -1),
        crs=CRS.from_epsg(4326),
        nodata=np.nan,
    )

    assert _mask_nodata_for_display(ds, array) is array


def test_valid_values_drops_masked_entries():
    array = np.ma.masked_equal(np.array([1.0, -9999.0, 3.0]), -9999.0)

    np.testing.assert_array_equal(_valid_values(array), [1.0, 3.0])


def test_valid_values_passes_plain_arrays_through():
    array = np.arange(6, dtype=np.float32).reshape(2, 3)

    np.testing.assert_array_equal(_valid_values(array), array.ravel())


def test_stretch_limits_exclude_the_nodata_sentinel(raster_with_nodata):
    """A -9999 fill must not drag the stretch down (CODE_STYLE nodata rule 1)."""
    band = raster_with_nodata.get_band(1)
    valid = band[band != -9999.0]

    limits = _stretch_limits(_mask_nodata_for_display(raster_with_nodata, band))

    assert limits == pytest.approx(tuple(np.nanpercentile(valid, (2, 98))))
    assert limits is not None and limits[0] > -9999.0


def test_stretch_limits_all_nodata_band():
    array = np.ma.masked_equal(np.full((4, 4), -9999.0), -9999.0)

    assert _stretch_limits(array) is None


@pytest.mark.parametrize(
    "plot_func",
    [plot_band_array, plot_raster, plot_raster_with_histogram],
    ids=["plot_band_array", "plot_raster", "plot_raster_with_histogram"],
)
def test_nodata_pixels_are_not_drawn(raster_with_nodata, plot_func, monkeypatch):
    """Nodata renders blank rather than as a colour at the sentinel's value."""
    calls = _capture_imshow(monkeypatch)

    plot_func(raster_with_nodata, stretch=True)

    drawn, clim = calls[0]
    assert np.ma.isMaskedArray(drawn)
    assert np.ma.getmaskarray(drawn).sum() == 4
    assert clim[0] > -9999.0  # the sentinel did not set the low limit


@pytest.mark.parametrize(
    "plot_func",
    [plot_histogram, plot_raster_with_histogram],
    ids=["plot_histogram", "plot_raster_with_histogram"],
)
def test_histograms_exclude_nodata(raster_with_nodata, plot_func, monkeypatch):
    binned = []
    real_hist = Axes.hist

    def spy(self, data, *args, **kwargs):
        binned.append(np.asarray(data))
        return real_hist(self, data, *args, **kwargs)

    monkeypatch.setattr(Axes, "hist", spy)

    plot_func(raster_with_nodata)

    assert binned[0].size == 32  # 36 pixels less the 4 nodata ones
    assert -9999.0 not in binned[0]


def test_colorbar_excludes_nodata(raster_with_nodata, monkeypatch):
    bars = _capture_colorbars(monkeypatch)

    plot_raster(raster_with_nodata, stretch=True, colorbar=True)

    band = raster_with_nodata.get_band(1)
    valid = band[band != -9999.0]
    assert bars[0].mappable.get_clim() == pytest.approx(tuple(np.nanpercentile(valid, (2, 98))))


def test_composite_makes_nodata_transparent(monkeypatch):
    """Nodata in any channel is nodata in the composite (contagion rule)."""
    base = np.linspace(0.0, 1.0, 36, dtype=np.float32).reshape(6, 6)
    array = np.stack([base, 0.5 * base, 0.25 * base])
    array[0, :2, :2] = -9999.0  # nodata in the red channel only
    ds = load_array(
        array,
        transform=Affine.translation(0, 6) * Affine.scale(1, -1),
        crs=CRS.from_epsg(4326),
        nodata=-9999.0,
    )
    captured = {}
    monkeypatch.setattr(plt, "imshow", lambda arr, *a, **k: captured.update(arr=np.asarray(arr)))

    plot_composite(ds, bands=(1, 2, 3), stretch=True)

    composite = captured["arr"]
    assert composite.shape == (6, 6, 4)  # RGBA: an alpha channel was added
    np.testing.assert_array_equal(composite[:2, :2, 3], 0.0)  # nodata transparent
    assert composite[2:, :, 3].min() == 1.0  # valid pixels opaque


def test_composite_stretch_ignores_nodata(monkeypatch):
    """The sentinel must not compress every valid pixel into one shade."""
    base = np.linspace(0.2, 0.8, 36, dtype=np.float32).reshape(6, 6)
    array = np.stack([base, base, base])
    array[:, 0, 0] = -9999.0
    ds = load_array(
        array,
        transform=Affine.translation(0, 6) * Affine.scale(1, -1),
        crs=CRS.from_epsg(4326),
        nodata=-9999.0,
    )
    captured = {}
    monkeypatch.setattr(plt, "imshow", lambda arr, *a, **k: captured.update(arr=np.asarray(arr)))

    plot_composite(ds, bands=(1, 2, 3), stretch=True)

    # Every pixel but [0, 0], which is the sentinel.
    red = captured["arr"][..., 0]
    valid = np.delete(red.ravel(), 0)
    # Masked: the stretch spans the real 0.2-0.8 range, so the channel uses the
    # full [0, 1]. Unmasked, -9999 as the low limit would crush it all near 1.
    assert valid.min() < 0.05 and valid.max() > 0.95


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
