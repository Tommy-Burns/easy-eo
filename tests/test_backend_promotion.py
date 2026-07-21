import numpy as np
import rasterio.io

from eeo.core.adapters import RasterioAdapter


def test_numpy_backend_initial(numpy_backed_dataset):
    backend = numpy_backed_dataset._adapter.backend
    assert isinstance(backend, np.ndarray)


# op results are DatasetWriter-backed; to_rasterio() must
# recognise them as already-rasterio and return self instead of re-reading
# the full array into a new MemoryFile.
def test_to_rasterio_is_noop_on_op_result(single_band_float32):
    result = single_band_float32.add(1)

    assert isinstance(result.ds, rasterio.io.DatasetWriter)
    assert result.to_rasterio() is result


def test_to_rasterio_is_noop_on_file_backed(tmp_path, single_band_float32):
    path = tmp_path / "plain.tif"
    single_band_float32.save_raster(str(path))

    import eeo

    ds = eeo.load_raster(str(path))
    assert ds.to_rasterio() is ds
    ds.close()


def _forbid_promotion(monkeypatch):
    def fail_from_array(*args, **kwargs):
        raise AssertionError("rasterio-backed dataset was needlessly re-promoted")

    monkeypatch.setattr(RasterioAdapter, "from_array", fail_from_array)


def test_extract_value_does_not_repromote_rasterio_dataset(single_band_float32, monkeypatch):
    _forbid_promotion(monkeypatch)

    # pixel (1, 1) on the 10 m UTM grid holds gradient value 7
    value = single_band_float32.extract_value_at_coordinate((500_015.0, 4_199_985.0))
    assert value == 7.0


def test_normalized_difference_does_not_repromote_rasterio_dataset(
    single_band_float32, monkeypatch
):
    _forbid_promotion(monkeypatch)

    result = single_band_float32.normalized_difference(single_band_float32)
    # (x - x) / (x + x) is 0 everywhere; the 0/0 pixel is guarded to 0
    assert np.all(result.read() == 0)


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
