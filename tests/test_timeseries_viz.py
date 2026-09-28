"""Terminal plots for a time series (eeo/viz/timeseries.py).

Three things are worth pinning down. A filmstrip's panels must share one colour
scale, or the figure misleads — a dark date and a bright one render identically
with a scale each. A trajectory must break at a gap rather than draw through it.
And both must read decimated, per the memory model's display rule, which for a
grid means one panel's budget and not the whole figure's.

The Agg backend is forced for the whole suite in conftest.py, so nothing here
needs a display.
"""

import datetime as dt
import warnings

import numpy as np
import pytest
from matplotlib.axes import Axes
from rasterio.crs import CRS
from rasterio.transform import from_origin

import eeo
from eeo.core.adapters import RasterioAdapter
from eeo.core.exceptions import ValidationError
from eeo.viz.plot import _DISPLAY_OVERSAMPLE
from eeo.viz.timeseries import plot_filmstrip, plot_trajectory

UTC = dt.timezone.utc
UTM = CRS.from_epsg(32633)
TRANSFORM = from_origin(500_000.0, 4_200_000.0, 10.0, 10.0)
INSIDE = (500_015.0, 4_199_985.0)
MONTHS = (3, 4, 5, 6, 7)


def scene(month, *, value, nodata=0, names=("red", "nir")):
    """A two-band scene whose pixel (0, 0) can be turned into a gap."""
    bands = [np.full((4, 4), value, dtype="uint16") for _ in names]
    return eeo.load_array(
        np.stack(bands),
        transform=TRANSFORM,
        crs=UTM,
        nodata=nodata,
        band_names=list(names),
        timestamp=dt.datetime(2023, month, 1, tzinfo=UTC),
    )


@pytest.fixture
def series():
    ts = eeo.time_series([scene(month, value=month * 100) for month in MONTHS])
    yield ts
    ts.close()


@pytest.fixture
def gapped():
    """A series whose sampled pixel is nodata at the middle timestep."""
    scenes = []
    for month in MONTHS:
        ds = scene(month, value=month * 100)
        array = ds.to_array().copy()
        if month == 5:
            array[:, 1, 1] = 0  # the declared nodata
        scenes.append(
            eeo.load_array(
                array,
                transform=TRANSFORM,
                crs=UTM,
                nodata=0,
                band_names=["red", "nir"],
                timestamp=dt.datetime(2023, month, 1, tzinfo=UTC),
            )
        )
    ts = eeo.time_series(scenes)
    yield ts
    ts.close()


def drawn_lines(monkeypatch):
    """Record every line a plot draws, as (x values, y values, label)."""
    lines = []
    real_plot = Axes.plot

    def spy(self, *args, **kwargs):
        result = real_plot(self, *args, **kwargs)
        for line in result:
            lines.append((line.get_xdata(), line.get_ydata(), line.get_label()))
        return result

    monkeypatch.setattr(Axes, "plot", spy)
    return lines


def drawn_images(monkeypatch):
    """Record every image a plot draws, as (array, colour limits)."""
    images = []
    real_imshow = Axes.imshow

    def spy(self, array, *args, **kwargs):
        image = real_imshow(self, array, *args, **kwargs)
        images.append((array, image.get_clim()))
        return image

    monkeypatch.setattr(Axes, "imshow", spy)
    return images


# --------------------------------------------------------------------------
# The trajectory
# --------------------------------------------------------------------------
def test_one_line_per_band_over_the_acquisition_dates(series, monkeypatch):
    lines = drawn_lines(monkeypatch)

    series.plot_trajectory(INSIDE)

    assert len(lines) == 2
    assert [label for _, _, label in lines] == ["red", "nir"]
    for x_values, y_values, _ in lines:
        assert len(x_values) == len(MONTHS)
        assert list(y_values) == pytest.approx([300.0, 400.0, 500.0, 600.0, 700.0])


def test_a_single_band_can_be_selected(series, monkeypatch):
    lines = drawn_lines(monkeypatch)

    series.plot_trajectory(INSIDE, bands=["nir"])

    assert len(lines) == 1
    assert lines[0][2] == "nir"


def test_a_gap_is_drawn_as_a_gap_not_as_a_zero(gapped, monkeypatch):
    lines = drawn_lines(monkeypatch)

    gapped.plot_trajectory(INSIDE, bands=["red"])

    _, y_values, _ = lines[0]
    # NaN at the clouded date, so matplotlib breaks the line there; a fill value
    # would have drawn a plunge to zero instead.
    assert np.isnan(y_values[2])
    assert not np.isnan(np.asarray(y_values, dtype=float)[[0, 1, 3, 4]]).any()


def test_every_acquisition_gets_a_marker(series, monkeypatch):
    # Without markers an isolated valid date between two gaps draws nothing at
    # all, since a line needs two ends.
    lines = []
    real_plot = Axes.plot

    def spy(self, *args, **kwargs):
        lines.append(kwargs.get("marker"))
        return real_plot(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", spy)

    series.plot_trajectory(INSIDE)

    assert lines and all(marker is not None for marker in lines)


def test_coordinates_can_be_given_in_another_crs(series, monkeypatch):
    from rasterio.warp import transform as warp_transform

    lon, lat = (value[0] for value in warp_transform(UTM, "EPSG:4326", [INSIDE[0]], [INSIDE[1]]))
    lines = drawn_lines(monkeypatch)

    series.plot_trajectory((lon, lat), crs="EPSG:4326")

    assert list(lines[0][1]) == pytest.approx([300.0, 400.0, 500.0, 600.0, 700.0])


def test_a_point_outside_the_series_is_refused(series):
    with pytest.raises(ValidationError, match="falls outside"):
        series.plot_trajectory((0.0, 0.0))


def test_the_trajectory_can_be_saved(series, tmp_path):
    out = tmp_path / "trajectory.png"

    series.plot_trajectory(INSIDE, save_path=out)

    assert out.exists() and out.stat().st_size > 0


def test_the_trajectory_is_also_a_plain_function(series, monkeypatch):
    lines = drawn_lines(monkeypatch)

    plot_trajectory(series, INSIDE, bands=["red"])

    assert len(lines) == 1


def titles_set(monkeypatch):
    """Record every axes title a plot sets."""
    titles = []
    real_set_title = Axes.set_title

    def spy(self, label, *args, **kwargs):
        titles.append(label)
        return real_set_title(self, label, *args, **kwargs)

    monkeypatch.setattr(Axes, "set_title", spy)
    return titles


@pytest.mark.parametrize(
    ("title", "expected"),
    [(None, ["(500015, 4199985)"]), ("Field 12", ["Field 12"]), ("", [])],
    ids=["default-is-the-point", "given", "empty-means-none"],
)
def test_the_trajectory_title(series, monkeypatch, title, expected):
    titles = titles_set(monkeypatch)

    series.plot_trajectory(INSIDE, title=title)

    assert titles == expected


# --------------------------------------------------------------------------
# The filmstrip
# --------------------------------------------------------------------------
def test_one_panel_per_timestep(series, monkeypatch):
    images = drawn_images(monkeypatch)

    series.plot_filmstrip()

    assert len(images) == len(MONTHS)


def test_panels_are_in_time_order(series, monkeypatch):
    images = drawn_images(monkeypatch)

    series.plot_filmstrip(band="red")

    assert [float(np.ma.median(array)) for array, _ in images] == pytest.approx(
        [300.0, 400.0, 500.0, 600.0, 700.0]
    )


def test_panels_are_titled_with_their_acquisition_date(series, monkeypatch):
    titles = titles_set(monkeypatch)

    series.plot_filmstrip()

    assert titles[: len(MONTHS)] == [
        "2023-03-01",
        "2023-04-01",
        "2023-05-01",
        "2023-06-01",
        "2023-07-01",
    ]


def test_every_panel_shares_one_colour_scale(series, monkeypatch):
    # The claim the filmstrip lives or dies by: with a scale each, a 300 panel
    # and a 700 panel render identically and the figure lies.
    images = drawn_images(monkeypatch)

    series.plot_filmstrip(band="red")

    limits = {clim for _, clim in images}
    assert len(limits) == 1
    low, high = limits.pop()
    assert low < high


def test_a_scale_per_panel_can_be_asked_for(series, monkeypatch):
    images = drawn_images(monkeypatch)

    series.plot_filmstrip(band="red", shared_scale=False)

    # Each panel is constant here, so a per-panel stretch finds no range and
    # falls back to matplotlib's own autoscaling — which is per panel, and so
    # still differs between them.
    assert len({clim for _, clim in images}) == len(MONTHS)


def ramp(month):
    """A single-band scene whose 16 pixels climb from ``month * 100``."""
    values = month * 100 + np.arange(16, dtype="uint16").reshape(1, 4, 4)
    return eeo.load_array(
        values,
        transform=TRANSFORM,
        crs=UTM,
        band_names=["red"],
        timestamp=dt.datetime(2023, month, 1, tzinfo=UTC),
    )


def test_a_scale_per_panel_stretches_each_panel_over_its_own_values(monkeypatch):
    ts = eeo.time_series([ramp(month) for month in MONTHS])
    images = drawn_images(monkeypatch)

    ts.plot_filmstrip(shared_scale=False)

    for (array, clim), month in zip(images, MONTHS, strict=True):
        assert clim == pytest.approx(np.percentile(np.ma.compressed(array), (2, 98)))
        assert month * 100 < clim[0] < clim[1] < month * 100 + 15
    ts.close()


def test_a_shared_scale_with_no_range_leaves_the_scaling_to_matplotlib(monkeypatch):
    # One value in every pixel of every panel: no percentile range to stretch
    # over, so no limits are forced and the panels still draw.
    ts = eeo.time_series([scene(month, value=500) for month in MONTHS])
    images = drawn_images(monkeypatch)

    ts.plot_filmstrip(band="red")

    assert len(images) == len(MONTHS)
    assert {clim for _, clim in images} == {(500.0, 500.0)}
    ts.close()


def test_a_colorbar_is_drawn_only_for_a_shared_scale(series, monkeypatch):
    bars = []
    from matplotlib.figure import Figure

    real_colorbar = Figure.colorbar

    def spy(self, *args, **kwargs):
        bars.append(kwargs.get("label"))
        return real_colorbar(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "colorbar", spy)

    series.plot_filmstrip(band="red")
    assert len(bars) == 1
    assert "red" in bars[0]

    series.plot_filmstrip(band="red", shared_scale=False)
    # Still one: a second bar would have to describe five different scales.
    assert len(bars) == 1


def test_a_band_can_be_named_or_indexed(series, monkeypatch):
    images = drawn_images(monkeypatch)

    series.plot_filmstrip(band=2)
    series.plot_filmstrip(band="nir")

    half = len(MONTHS)
    by_index = [float(np.ma.median(array)) for array, _ in images[:half]]
    by_name = [float(np.ma.median(array)) for array, _ in images[half:]]
    assert by_index == pytest.approx(by_name)


def test_an_unknown_band_is_refused(series):
    with pytest.raises(ValidationError, match="swir"):
        series.plot_filmstrip(band="swir")


def test_the_grid_can_be_chosen(series, monkeypatch):
    shapes = []
    import matplotlib.pyplot as plt

    real_subplots = plt.subplots

    def spy(*args, **kwargs):
        result = real_subplots(*args, **kwargs)
        shapes.append(np.shape(result[1]))
        return result

    monkeypatch.setattr(plt, "subplots", spy)

    series.plot_filmstrip(ncols=5)

    assert shapes[0] == (1, 5)


def test_a_grid_too_small_for_the_series_is_refused(series):
    with pytest.raises(ValidationError):
        series.plot_filmstrip(nrows=1, ncols=2)


def test_a_nodata_pixel_is_masked_rather_than_drawn(gapped, monkeypatch):
    images = drawn_images(monkeypatch)

    gapped.plot_filmstrip(band="red")

    middle, _ = images[2]
    assert np.ma.is_masked(middle[1, 1])


def test_the_filmstrip_can_be_saved(series, tmp_path):
    out = tmp_path / "filmstrip.png"

    series.plot_filmstrip(save_path=out)

    assert out.exists() and out.stat().st_size > 0


def test_the_filmstrip_can_be_titled(series, monkeypatch):
    from matplotlib.figure import Figure

    titles = []
    real_suptitle = Figure.suptitle

    def spy(self, text, *args, **kwargs):
        titles.append(text)
        return real_suptitle(self, text, *args, **kwargs)

    monkeypatch.setattr(Figure, "suptitle", spy)

    series.plot_filmstrip(title="Spring green-up")

    assert titles == ["Spring green-up"]


def test_the_filmstrip_is_also_a_plain_function(series, monkeypatch):
    images = drawn_images(monkeypatch)

    plot_filmstrip(series, band="red")

    assert len(images) == len(MONTHS)


def test_a_binned_series_can_be_filmstripped(series, monkeypatch):
    monthly = series.resample_time("MS").median()
    images = drawn_images(monkeypatch)

    monthly.plot_filmstrip()

    assert len(images) == 5
    monthly.close()


# --------------------------------------------------------------------------
# Decimated reads (memory model rule 4)
# --------------------------------------------------------------------------
LARGE_SIDE = 600


@pytest.fixture
def large_series(tmp_path):
    """Five 600x600 rasterio-backed scenes, large against a tiny figure budget.

    Not constant: a zero value range gives the percentile stretch nothing to
    work with, and this fixture exists to exercise the read path, not that one.
    """
    paths = []
    for month in MONTHS:
        gradient = np.linspace(month, month + 1, LARGE_SIDE * LARGE_SIDE, dtype="float32").reshape(
            LARGE_SIDE, LARGE_SIDE
        )
        path = tmp_path / f"scene_2023{month:02d}01.tif"
        eeo.load_array(
            gradient[np.newaxis],
            transform=from_origin(500_000.0, 4_200_000.0, 10.0, 10.0),
            crs=UTM,
        ).save_raster(path)
        paths.append(path)
    ts = eeo.time_series(tmp_path)
    yield ts
    ts.close()


def record_read_shapes(monkeypatch):
    """Spy on both adapter read paths, recording each result's shape."""
    shapes = []
    real_read = RasterioAdapter.read
    real_read_band = RasterioAdapter.read_band

    def spy_read(self, *args, **kwargs):
        result = real_read(self, *args, **kwargs)
        shapes.append(result.shape)
        return result

    def spy_read_band(self, index):
        result = real_read_band(self, index)
        shapes.append(result.shape)
        return result

    monkeypatch.setattr(RasterioAdapter, "read", spy_read)
    monkeypatch.setattr(RasterioAdapter, "read_band", spy_read_band)
    return shapes


def test_a_filmstrip_of_large_scenes_reads_decimated(large_series, monkeypatch):
    import matplotlib.pyplot as plt

    shapes = record_read_shapes(monkeypatch)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Tight layout not applied", category=UserWarning)
        large_series.plot_filmstrip(figsize=(2, 2))

    budget = round(plt.rcParams["figure.dpi"] * _DISPLAY_OVERSAMPLE)
    assert budget < LARGE_SIDE, "the figure must be small enough to force decimation"
    assert shapes, "no pixels were read through the adapter"
    for shape in shapes:
        height, width = shape[-2:]
        assert height <= budget
        assert width <= budget


def test_the_panel_budget_is_one_panel_not_the_whole_figure(large_series, monkeypatch):
    # A grid of n panels read at figure resolution would pull n times the pixels
    # it can show, which is the thing rule 4 is about.
    import matplotlib.pyplot as plt

    shapes = record_read_shapes(monkeypatch)

    large_series.plot_filmstrip(ncols=5, figsize=(20, 4))

    dpi = plt.rcParams["figure.dpi"]
    panel_budget = round((20 / 5) * dpi * _DISPLAY_OVERSAMPLE)
    figure_budget = round(20 * dpi * _DISPLAY_OVERSAMPLE)
    assert panel_budget < figure_budget  # sanity: the two differ
    for shape in shapes:
        assert shape[-1] <= panel_budget


def test_a_trajectory_of_large_scenes_reads_one_pixel_per_timestep(large_series, monkeypatch):
    shapes = record_read_shapes(monkeypatch)

    large_series.plot_trajectory((500_005.0, 4_199_995.0))

    assert shapes, "no pixels were read through the adapter"
    assert all(int(np.prod(shape)) == 1 for shape in shapes), shapes
