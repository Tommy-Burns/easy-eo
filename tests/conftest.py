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

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from eeo import load_array

UTM_CRS = CRS.from_epsg(32633)
GEO_CRS = CRS.from_epsg(4326)
ORIGIN_X = 500_000.0
ORIGIN_Y = 4_200_000.0
RES = 10.0
NODATA = -9999.0


def _north_up(origin_x: float = ORIGIN_X, origin_y: float = ORIGIN_Y,
              res: float = RES) -> Affine:
    """Return a north-up affine transform with square pixels."""
    return Affine.translation(origin_x, origin_y) * Affine.scale(res, -res)


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
    bands = np.stack(
        [i * 1000 + _gradient((6, 6), np.uint16) for i in range(1, 5)]
    ).astype(np.uint16)
    ds = load_array(bands, transform=_north_up(), crs=UTM_CRS).to_rasterio()
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
        transform=Affine.translation(12.0, 42.0) * Affine.scale(0.0001, -0.0001),
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
        transform=Affine.translation(0, 3) * Affine.scale(1, -1),
        crs=GEO_CRS,
    )
