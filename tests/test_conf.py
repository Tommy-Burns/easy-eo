import numpy as np
import pytest
from affine import Affine
from pyproj import CRS

from eeo import load_array
from eeo.core import EEORasterDataset

@pytest.fixture
def dummy_transform():
    return Affine.translation(0, 0) * Affine.scale(1, -1)


@pytest.fixture
def dummy_crs():
    return 4326  # EPSG:4326


@pytest.fixture
def single_band_array():
    return np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ], dtype=np.float32)


@pytest.fixture
def multi_band_array():
    return np.stack(
        [
            np.ones((3, 3), dtype=np.float32),
            np.full((3, 3), 10, dtype=np.float32),
        ]
    )


@pytest.fixture
def raster_from_array():
    array = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        dtype=np.float32,
    )

    transform = Affine.translation(0, 0) * Affine.scale(1, -1)
    crs = CRS.from_epsg(4326)

    return load_array(array, transform=transform, crs=crs)



@pytest.fixture
def multiband_raster_from_array(multi_band_array, dummy_transform, dummy_crs):
    return EEORasterDataset.from_array(
        array=multi_band_array,
        transform=dummy_transform,
        crs=dummy_crs,
        nodata=None,
    )
