"""Shared synthetic-raster fixtures for the Easy-EO test suite.

Every fixture builds a small, deterministic raster fully in memory
(rasterio ``MemoryFile`` backend via ``to_rasterio()``, or the NumPy
backend via ``load_array``); nothing here touches the filesystem or the
network.

Grid conventions
----------------
Rasterio-backed fixtures use a Sentinel-2-like grid: EPSG:32633 (UTM 33N),
10 m square pixels, origin (500000, 4200000), north-up transform. The
CRS-mismatch partner raster uses EPSG:4326. Pixel values are deterministic
gradients (``0..n-1``) so tests can assert against hand-computed results.
"""

import os
import pathlib
import socket
import warnings
from datetime import datetime, timezone

import matplotlib

# Force the non-interactive Agg backend for the whole suite, before eeo
# (which imports matplotlib.pyplot) is loaded. Viz tests must never need a
# display.
matplotlib.use("Agg")

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS
from rasterio.transform import from_origin

import eeo
from eeo import load_array

UTM_CRS = CRS.from_epsg(32633)
GEO_CRS = CRS.from_epsg(4326)
ORIGIN_X = 500_000.0
ORIGIN_Y = 4_200_000.0
RES = 10.0
NODATA = -9999.0


#: Environment variables naming a downloaded product to test against. The
#: paths are not hardcoded and have no default: a scene lives outside the
#: repository, so the default run must not go looking for one.
SENTINEL2_SCENE_ENV = "EEO_TEST_SENTINEL2_SCENE"
LANDSAT_SCENE_ENV = "EEO_TEST_LANDSAT_SCENE"


def pytest_addoption(parser):
    """Register the opt-in flags for tests the default run must not do."""
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.network (real downloads)",
    )
    parser.addoption(
        "--run-realdata",
        action="store_true",
        default=False,
        help=(
            "run tests marked @pytest.mark.realdata against downloaded products "
            f"named by ${SENTINEL2_SCENE_ENV} and ${LANDSAT_SCENE_ENV}"
        ),
    )


def pytest_collection_modifyitems(config, items):
    """Skip opt-in tests unless their flag is given."""
    optional = {
        "network": ("--run-network", "needs --run-network (real download)"),
        "realdata": ("--run-realdata", "needs --run-realdata (downloaded product)"),
    }
    for marker, (flag, reason) in optional.items():
        if config.getoption(flag):
            continue
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)


def _scene_path(variable):
    """Return the product path named by an environment variable, or skip."""
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"set ${variable} to a downloaded product to run this test")
    path = pathlib.Path(value).expanduser()
    if not path.exists():
        pytest.skip(f"${variable} points at {path}, which does not exist")
    return path


@pytest.fixture(scope="session")
def sentinel2_scene():
    """Path to a real Sentinel-2 L2A product (``.SAFE`` directory or zip)."""
    return _scene_path(SENTINEL2_SCENE_ENV)


@pytest.fixture(scope="session")
def landsat_scene():
    """Path to a real Landsat Collection 2 Level-2 product (directory or tar)."""
    return _scene_path(LANDSAT_SCENE_ENV)


@pytest.fixture(autouse=True)
def _block_network(request, monkeypatch):
    """Fail any unmarked test that opens a network connection.

    The default run must be offline: STAC, sample-data, and remote-raster
    tests all work from recordings, fakes, or local files. This turns a test
    that quietly starts reaching a real service into an immediate, obvious
    failure rather than a slow or flaky one.

    Tests marked ``@pytest.mark.network`` are exempt (they only run under
    ``--run-network``). The guard covers Python-level sockets, which is what
    ``requests`` and ``urllib`` use; GDAL's own HTTP stack does not go through
    them, so a remote raster read is kept out of the default suite by review
    rather than by this fixture.
    """
    if request.node.get_closest_marker("network"):
        return

    def blocked(*args, **kwargs):
        raise RuntimeError(
            "this test tried to open a network connection, which the default "
            "test run forbids; use a recorded response or a local file, or "
            "mark the test with @pytest.mark.network"
        )

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture(autouse=True)
def _silence_agg_show_warning():
    """Filter the UserWarning ``plt.show()`` emits under Agg.

    The plot functions currently call ``plt.show()`` unconditionally; under
    the non-interactive backend every viz test would warn. Remove this
    filter when WP-05.4 reworks the plot functions' display path.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="FigureCanvasAgg is non-interactive",
            category=UserWarning,
        )
        yield


def _north_up(origin_x: float = ORIGIN_X, origin_y: float = ORIGIN_Y, res: float = RES) -> Affine:
    """Return a north-up affine transform with square pixels."""
    return from_origin(origin_x, origin_y, res, res)


def _gradient(shape: tuple[int, int], dtype) -> np.ndarray:
    """Return a deterministic 0..n-1 gradient array of the given shape."""
    return np.arange(np.prod(shape), dtype=dtype).reshape(shape)


@pytest.fixture
def single_band_float32():
    """6x6 single-band float32 raster on the UTM grid, rasterio-backed."""
    ds = load_array(
        _gradient((6, 6), np.float32),
        transform=_north_up(),
        crs=UTM_CRS,
    ).to_rasterio()
    yield ds
    ds.close()


@pytest.fixture
def multiband_uint16():
    """4-band 6x6 uint16 raster, rasterio-backed.

    Band ``i`` (1-based) holds ``i * 1000`` plus the 0..35 gradient, so
    every band is distinct and every value is hand-computable.
    """
    bands = np.stack([i * 1000 + _gradient((6, 6), np.uint16) for i in range(1, 5)]).astype(
        np.uint16
    )
    ds = load_array(bands, transform=_north_up(), crs=UTM_CRS).to_rasterio()
    yield ds
    ds.close()


@pytest.fixture
def nonsquare_float32():
    """4x8 single-band float32 raster on the UTM grid, rasterio-backed.

    Non-square (4 rows, 8 columns) so that height/width mix-ups produce
    observable differences.
    """
    ds = load_array(
        _gradient((4, 8), np.float32),
        transform=_north_up(),
        crs=UTM_CRS,
    ).to_rasterio()
    yield ds
    ds.close()


@pytest.fixture
def raster_with_nodata():
    """6x6 float32 raster with nodata=-9999 in the top-left 2x2 block."""
    array = _gradient((6, 6), np.float32)
    array[:2, :2] = NODATA
    ds = load_array(
        array,
        transform=_north_up(),
        crs=UTM_CRS,
        nodata=NODATA,
    ).to_rasterio()
    yield ds
    ds.close()


@pytest.fixture
def crs_mismatch_pair():
    """Pair of 6x6 float32 rasters with identical values but different CRS.

    The first is on the UTM grid (EPSG:32633), the second on a geographic
    grid (EPSG:4326) near (12E, 42N).
    """
    utm = load_array(
        _gradient((6, 6), np.float32),
        transform=_north_up(),
        crs=UTM_CRS,
    ).to_rasterio()
    geo = load_array(
        _gradient((6, 6), np.float32),
        transform=from_origin(12.0, 42.0, 0.0001, 0.0001),
        crs=GEO_CRS,
    ).to_rasterio()
    yield utm, geo
    utm.close()
    geo.close()


@pytest.fixture
def shape_mismatch_pair():
    """Pair of rasters with the same CRS and bounds but different resolution.

    A 6x6 raster at 10 m and a 3x3 raster at 20 m covering the identical
    extent — the auto-align / resample-to-target case.
    """
    fine = load_array(
        _gradient((6, 6), np.float32),
        transform=_north_up(),
        crs=UTM_CRS,
    ).to_rasterio()
    coarse = load_array(
        _gradient((3, 3), np.float32),
        transform=_north_up(res=20.0),
        crs=UTM_CRS,
    ).to_rasterio()
    yield fine, coarse
    fine.close()
    coarse.close()


@pytest.fixture
def numpy_backed_dataset():
    """6x6 float32 dataset on the NumPy backend (no rasterio promotion)."""
    return load_array(
        _gradient((6, 6), np.float32),
        transform=_north_up(),
        crs=UTM_CRS,
    )


@pytest.fixture
def raster_3x3():
    """3x3 float32 raster with values 1..9, NumPy-backed.

    The odd pixel count gives clean central statistics for the stats ops:
    mean = median = 5 at pixel (1, 1), min 1 at (0, 0), max 9 at (2, 2).
    North-up unit-pixel grid with top-left origin (0, 3) in EPSG:4326, so
    world coordinate (1.5, 1.5) falls in pixel (1, 1).
    """
    return load_array(
        np.arange(1, 10, dtype=np.float32).reshape(3, 3),
        transform=from_origin(0, 3, 1, 1),
        crs=GEO_CRS,
    )


# Five monthly acquisitions over one growing season, for the time-series layer.
# Red dips and near-infrared peaks in midsummer, so an index computed across the
# stack traces a season rather than noise, and every reduction is a round number.
SEASON_MONTHS = (3, 4, 5, 6, 7)
SEASON_RED = (1000, 900, 800, 900, 1000)
SEASON_NIR = (2000, 3000, 4000, 3000, 2000)
SEASON_NODATA = 0
# Pixel (0, 0) is nodata at the third and fourth timesteps only, so a reducer
# that ignores nodata and one that does not give different answers there.
SEASON_GAPS = (2, 3)


@pytest.fixture
def season_stack():
    """Five 4x4 two-band uint16 scenes, one per month of a growing season.

    Rasterio-backed, all on the same UTM grid, bands named ``red`` and ``nir``,
    each carrying its acquisition time (the first of March through July 2023,
    UTC) so they can be handed straight to a time series. Values are uniform
    within a scene and hand-computable across the stack:

    ==========  ====  ====  ====
    timestep    red   nir   NDVI
    ==========  ====  ====  ====
    2023-03-01  1000  2000  1/3
    2023-04-01   900  3000  0.538…
    2023-05-01   800  4000  2/3
    2023-06-01   900  3000  0.538…
    2023-07-01  1000  2000  1/3
    ==========  ====  ====  ====

    So over the stack red has median 900, mean 920, min 800, max 1000, and nir
    has median 3000, mean 2800, min 2000, max 4000.

    Every scene declares ``nodata=0``, and pixel (0, 0) *is* 0 at the third and
    fourth timesteps, so a nodata-aware reduction over that pixel sees only
    three values while a naive one averages in two zeros.
    """
    scenes = []
    for index, month in enumerate(SEASON_MONTHS):
        bands = np.stack(
            [
                np.full((4, 4), SEASON_RED[index], dtype=np.uint16),
                np.full((4, 4), SEASON_NIR[index], dtype=np.uint16),
            ]
        )
        if index in SEASON_GAPS:
            bands[:, 0, 0] = SEASON_NODATA
        scenes.append(
            load_array(
                bands,
                transform=_north_up(),
                crs=UTM_CRS,
                nodata=SEASON_NODATA,
                timestamp=datetime(2023, month, 1, tzinfo=timezone.utc),
                band_names=["red", "nir"],
            ).to_rasterio()
        )
    yield scenes
    for scene in scenes:
        scene.close()


@pytest.fixture
def season_series(season_stack):
    """The :func:`season_stack` scenes as an ``EEOTimeSeries``, oldest first."""
    series = eeo.time_series(season_stack)
    yield series
    series.close()


@pytest.fixture
def season_reference(season_stack):
    """A single-band constant 1000 raster on the season grid, for two-raster ops.

    The partner for a normalized difference across the stack: with it, band 1 of
    each result is ``(red - 1000) / (red + 1000)`` and band 2
    ``(nir - 1000) / (nir + 1000)``, both hand-computable per timestep.
    """
    ds = load_array(
        np.full((4, 4), 1000, dtype=np.uint16),
        transform=_north_up(),
        crs=UTM_CRS,
        nodata=SEASON_NODATA,
    ).to_rasterio()
    yield ds
    ds.close()
