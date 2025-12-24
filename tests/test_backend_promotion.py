import numpy as np
import pytest
import rasterio.io
from affine import Affine
from rasterio.crs import CRS

from eeo import load_array



@pytest.fixture
def numpy_dataset():
    array = np.random.rand(100, 100)
    transform = Affine.translation(0, 0) * Affine.scale(1, -1)
    crs = CRS.from_epsg(4326)
    return load_array(array, transform=transform, crs=crs)


def test_numpy_backend_initial(numpy_dataset):
    ds = numpy_dataset
    backend = ds._adapter.backend
    assert isinstance(backend, np.ndarray)


def test_resample_promotes_to_rasterio(numpy_dataset):
    ds = numpy_dataset

    resampled = ds.resample(scale_factor=2.0)

    # Backend should now be Rasterio
    backend = resampled._adapter.backend
    print(backend.__class__.__name__)
    assert backend is not None
    assert isinstance(backend, rasterio.io.DatasetReader)


def test_resample_preserves_crs(numpy_dataset):
    ds = numpy_dataset
    resampled = ds.resample(scale_factor=2.0)

    assert resampled.get_crs() == ds.get_crs()


def test_resample_preserves_transform(numpy_dataset):
    ds = numpy_dataset
    resampled = ds.resample(scale_factor=2.0)

    assert resampled.get_transform() is not None


def test_resample_output_shape(numpy_dataset):
    ds = numpy_dataset
    resampled = ds.resample(scale_factor=2.0)

    h, w = ds.get_shape()
    new_h, new_w = resampled.get_shape()

    assert new_h == h * 2
    assert new_w == w * 2
