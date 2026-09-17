"""Reading a cloud-optimized GeoTIFF over HTTP.

The claim being tested is not only that a URL opens, but that opening and
reading one fetch the bytes they need and no more: a COG is useful remotely
precisely because a window costs a few range requests rather than the file.
The server is local (a child process bound to ``127.0.0.1``), so these tests
run offline like every other; what they exercise is GDAL's range-request path,
which is the same code whether the other end is localhost or an object store.
"""

import numpy as np
import pytest
import rasterio as rio
from rasterio.crs import CRS
from rasterio.shutil import copy as rio_copy
from rasterio.transform import from_origin
from rasterio.windows import Window

import eeo
from eeo.core.adapters import RasterioAdapter, XarrayAdapter
from eeo.core.exceptions import BackendError
from eeo.core.remote import is_gdal_path, is_remote
from http_fixtures import serve_directory

SIDE = 1024
BLOCK = 256
TRANSFORM = from_origin(500000.0, 4000000.0, 10.0, 10.0)
CRS_UTM = CRS.from_epsg(32633)


@pytest.fixture(scope="module")
def cog_dir(tmp_path_factory):
    """A directory holding one 2-band COG, and the array it was written from.

    Returns the directory, the array, and the COG's size on disk — the last so
    a test can state what "fetched far less than the file" actually means.
    """
    directory = tmp_path_factory.mktemp("cog")
    array = (np.arange(2 * SIDE * SIDE, dtype="uint16").reshape(2, SIDE, SIDE) % 9000).astype(
        "uint16"
    )
    plain = directory / "plain.tif"
    with rio.open(
        plain,
        "w",
        driver="GTiff",
        height=SIDE,
        width=SIDE,
        count=2,
        dtype="uint16",
        crs=CRS_UTM,
        transform=TRANSFORM,
        nodata=0,
    ) as dst:
        dst.write(array)
        dst.set_band_description(1, "red")
        dst.set_band_description(2, "nir")
    cog = directory / "scene.tif"
    rio_copy(plain, cog, driver="COG", blocksize=BLOCK)
    plain.unlink()
    return directory, array, cog.stat().st_size


@pytest.fixture
def served(cog_dir, tmp_path):
    """The COG directory, served over HTTP for one test."""
    directory = cog_dir[0]
    with serve_directory(directory, tmp_path) as server:
        yield server


@pytest.fixture(params=[None, BLOCK], ids=["rasterio", "lazy"])
def chunks(request):
    """Open the remote raster on each backend in turn.

    Range requests are GDAL's doing, not the backend's, so the byte-level
    claims below are asserted for both. The lazy half needs the extra.
    """
    if request.param is not None:
        pytest.importorskip("dask.array")
        pytest.importorskip("rioxarray")
    return request.param


@pytest.fixture
def expected(cog_dir):
    """The pixels the COG was written from."""
    return cog_dir[1]


# ------------------------------------------------------- path classification


@pytest.mark.parametrize(
    "path",
    [
        "https://example.com/scene.tif",
        "HTTP://example.com/scene.tif",
        "s3://bucket/scene.tif",
        "gs://bucket/scene.tif",
        "/vsicurl/https://example.com/scene.tif",
        "/vsis3/bucket/scene.tif",
    ],
)
def test_remote_paths_are_recognised(path):
    assert is_remote(path)
    assert is_gdal_path(path)


@pytest.mark.parametrize("path", ["scene.tif", "/data/scene.tif", "C:\\data\\scene.tif"])
def test_local_paths_are_not_remote(path):
    assert not is_remote(path)
    assert not is_gdal_path(path)


def test_an_archive_member_is_gdal_s_to_resolve_but_is_not_remote():
    """A local virtual path exists to GDAL and not to os.path.isfile."""
    path = "/vsizip/products.zip/scene.tif"

    assert is_gdal_path(path)
    assert not is_remote(path)


def test_a_missing_local_file_still_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        eeo.load_raster(tmp_path / "absent.tif")


def test_an_unreachable_url_is_a_backend_error():
    # Port 0 never listens, so this fails without leaving the machine.
    with pytest.raises(BackendError, match="could not be opened"):
        eeo.load_raster("http://127.0.0.1:0/nothing.tif")


# --------------------------------------------------------------- opening


def test_a_url_opens_on_the_rasterio_backend(served, expected):
    ds = eeo.load_raster(served.url("scene.tif"))

    assert isinstance(ds._adapter, RasterioAdapter)
    assert ds.get_shape() == (SIDE, SIDE)
    assert ds.band_names == ["red", "nir"]
    np.testing.assert_array_equal(ds.read(), expected)


def test_a_url_opens_on_the_lazy_backend(served, expected):
    pytest.importorskip("dask.array")
    ds = eeo.load_raster(served.url("scene.tif"), chunks=BLOCK)

    assert isinstance(ds._adapter, XarrayAdapter)
    assert ds.get_shape() == (SIDE, SIDE)
    assert ds.band_names == ["red", "nir"]
    assert ds.get_metadata()["nodata"] == 0
    np.testing.assert_array_equal(ds.read(), expected)


def test_opening_fetches_only_the_header(served, cog_dir, chunks):
    """The whole point of a COG: metadata costs a few kilobytes, not a file."""
    file_size = cog_dir[2]

    ds = eeo.load_raster(served.url("scene.tif"), chunks=chunks)
    ds.get_metadata()
    ds.get_bounds()

    assert 0 < served.bytes_served() < file_size // 10, served.requests()


def test_reading_a_window_fetches_less_than_the_file(served, expected, chunks):
    ds = eeo.load_raster(served.url("scene.tif"), chunks=chunks)
    served.reset()

    window = Window(col_off=BLOCK, row_off=BLOCK, width=BLOCK, height=BLOCK)
    block = ds.read(1, window=window)

    np.testing.assert_array_equal(block, expected[0, BLOCK : 2 * BLOCK, BLOCK : 2 * BLOCK])
    windowed = served.bytes_served()
    served.reset()
    ds.read()
    whole = served.bytes_served()
    assert 0 < windowed < whole


def test_every_fetch_is_a_range_request(served, chunks):
    ds = eeo.load_raster(served.url("scene.tif"), chunks=chunks)
    ds.read(1, window=Window(0, 0, BLOCK, BLOCK))

    fetches = [entry for entry in served.requests() if entry["method"] == "GET"]
    assert fetches, served.requests()
    assert all(entry["status"] == 206 for entry in fetches), fetches


# ------------------------------------------------------------- operations


def test_an_operation_runs_against_a_remote_raster(served, expected, chunks):
    ds = eeo.load_raster(served.url("scene.tif"), chunks=chunks)

    ndvi = ds.ndvi("red", nir="nir")

    red, nir = expected[0].astype("float32"), expected[1].astype("float32")
    with np.errstate(divide="ignore", invalid="ignore"):
        reference = (nir - red) / (nir + red)
    reference[(expected[0] == 0) | (expected[1] == 0)] = np.nan
    np.testing.assert_allclose(ndvi.read()[0], reference, equal_nan=True)


def test_saving_a_remote_raster_locally_round_trips(served, expected, tmp_path, chunks):
    ds = eeo.load_raster(served.url("scene.tif"), chunks=chunks)
    out = tmp_path / "local.tif"

    ds.save_raster(out)

    with rio.open(out) as saved:
        np.testing.assert_array_equal(saved.read(), expected)
        assert saved.descriptions == ("red", "nir")
