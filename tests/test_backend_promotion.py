import numpy as np
import rasterio.io


def test_numpy_backend_initial(numpy_backed_dataset):
    backend = numpy_backed_dataset._adapter.backend
    assert isinstance(backend, np.ndarray)


def test_resample_promotes_to_rasterio(numpy_backed_dataset):
    resampled = numpy_backed_dataset.resample(scale_factor=2.0)

    # Backend should now be Rasterio
    backend = resampled._adapter.backend
    assert backend is not None
    assert isinstance(backend, rasterio.io.DatasetReader)


def test_resample_preserves_crs(numpy_backed_dataset):
    resampled = numpy_backed_dataset.resample(scale_factor=2.0)

    assert resampled.get_crs() == numpy_backed_dataset.get_crs()


def test_resample_preserves_transform(numpy_backed_dataset):
    resampled = numpy_backed_dataset.resample(scale_factor=2.0)

    assert resampled.get_transform() is not None


def test_resample_output_shape(numpy_backed_dataset):
    resampled = numpy_backed_dataset.resample(scale_factor=2.0)

    h, w = numpy_backed_dataset.get_shape()
    new_h, new_w = resampled.get_shape()

    assert new_h == h * 2
    assert new_w == w * 2
