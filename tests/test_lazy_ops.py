"""Operations on the lazy (xarray) backend.

Two claims are checked for every operation: it produces exactly what the same
call produces on the rasterio backend, and it does not read the lazy raster
into memory to get there. The second is what makes the first worth having, and
it is asserted directly — a dask computation is the only way pixels can leave
the lazy backend, so a run that computes nothing read nothing.
"""

import warnings

import numpy as np
import pytest
import rasterio as rio
from rasterio.crs import CRS
from rasterio.transform import from_origin

import eeo
from eeo.core.adapters import RasterioAdapter, XarrayAdapter
from eeo.core.core import EEORasterDataset
from eeo.core.exceptions import BackendError

pytest.importorskip("dask.array")
pytest.importorskip("rioxarray")

HEIGHT, WIDTH = 120, 160
LARGE_SIDE = 600
TRANSFORM = from_origin(500000.0, 4000000.0, 10.0, 10.0)
CRS_UTM = CRS.from_epsg(32633)


@pytest.fixture
def scene_path(tmp_path):
    """3-band uint16 GeoTIFF with nodata=0 in a corner and named bands."""
    array = np.arange(3 * HEIGHT * WIDTH, dtype=np.uint16).reshape(3, HEIGHT, WIDTH) + 1
    array[:, :4, :4] = 0
    path = tmp_path / "scene.tif"
    with rio.open(
        path,
        "w",
        driver="GTiff",
        height=HEIGHT,
        width=WIDTH,
        count=3,
        dtype="uint16",
        crs=CRS_UTM,
        transform=TRANSFORM,
        nodata=0,
    ) as dst:
        dst.write(array)
        for i, name in enumerate(["red", "nir", "swir16"], start=1):
            dst.set_band_description(i, name)
    return path


@pytest.fixture
def both(scene_path):
    """The same scene opened lazily and with rasterio."""
    return eeo.load_raster(scene_path, chunks=32), eeo.load_raster(scene_path)


class Computes:
    """Count the dask computations a block of code performs."""

    def __enter__(self):
        from dask.callbacks import Callback

        self.graphs = []
        self._callback = Callback(start=lambda dsk: self.graphs.append(dsk))
        self._callback.__enter__()
        return self

    def __exit__(self, *exc):
        return self._callback.__exit__(*exc)

    @property
    def count(self):
        return len(self.graphs)


# --------------------------------------------------------------- promotion


def test_promotion_reopens_the_file_instead_of_reading_it(both):
    lazy, reference = both

    with Computes() as computed:
        promoted = lazy.to_rasterio()

    assert computed.count == 0
    assert isinstance(promoted._adapter, RasterioAdapter)
    np.testing.assert_array_equal(promoted.read(), reference.read())
    assert promoted.get_metadata() == reference.get_metadata()


def test_promotion_carries_names_and_provenance(scene_path):
    lazy = eeo.load_raster(scene_path, chunks=32, attrs={"site": "a"})
    lazy.set_band_name(2, "renamed")

    promoted = lazy.to_rasterio()

    assert promoted.band_names == ["red", "renamed", "swir16"]
    assert promoted.attrs == {"site": "a"}
    assert promoted.path == lazy.path


def test_a_dataarray_without_a_file_is_promoted_by_reading_it(both):
    """The fallback: an array with no file behind it has to be materialised."""
    lazy, reference = both
    detached = EEORasterDataset(adapter=XarrayAdapter(lazy.ds))

    promoted = detached.to_rasterio()

    assert isinstance(promoted._adapter, RasterioAdapter)
    np.testing.assert_array_equal(promoted.read(), reference.read())


# -------------------------------------------------------------- operations

#: Each op as a callable on one dataset, named for the test id.
OPS = {
    "add": lambda ds: ds.add(5),
    "multiply": lambda ds: ds.multiply(2),
    "divide": lambda ds: ds.divide(2),
    "sqrt": lambda ds: ds.sqrt(),
    "ndvi": lambda ds: ds.ndvi("red", nir="nir"),
    "ndbi": lambda ds: ds.ndbi("nir", swir="swir16"),
    "savi": lambda ds: ds.savi("red", nir="nir"),
    "normalize_min_max": lambda ds: ds.normalize_min_max(),
    "normalize_percentile": lambda ds: ds.normalize_percentile(),
    "standardize": lambda ds: ds.standardize(),
    "clip_bbox": lambda ds: ds.clip_raster_with_bbox((500100.0, 3998900.0, 500900.0, 3999900.0)),
    "resample": lambda ds: ds.resample(scale_factor=0.5),
    "reproject": lambda ds: ds.reproject_raster(target_crs=4326),
}


@pytest.mark.parametrize("op", list(OPS), ids=list(OPS))
def test_an_op_gives_the_same_result_on_both_backends(both, op):
    lazy, reference = both

    result = OPS[op](lazy)
    expected = OPS[op](reference)

    np.testing.assert_array_equal(result.read(), expected.read(), strict=True)
    # assert_equal rather than ==: a float result declares NaN nodata, and
    # NaN is not equal to itself.
    np.testing.assert_equal(result.get_metadata(), expected.get_metadata())
    assert result.band_names == expected.band_names


@pytest.mark.parametrize("op", list(OPS), ids=list(OPS))
def test_an_op_does_not_read_the_lazy_raster(both, op):
    """Promotion reopens the file, so the dask array is never computed."""
    lazy, _ = both

    with Computes() as computed:
        OPS[op](lazy).read()

    assert computed.count == 0


#: Statistics and samples, which return plain values rather than datasets.
STATS = {
    "mean": lambda ds: ds.get_mean_pixel(),
    "maximum": lambda ds: ds.get_maximum_pixel(),
    "minimum": lambda ds: ds.get_minimum_pixel(),
    "percentile": lambda ds: ds.get_percentile_pixel(50),
    "extract": lambda ds: ds.extract_value_at_coordinate((500500.0, 3999000.0)),
}


@pytest.mark.parametrize("stat", list(STATS), ids=list(STATS))
def test_a_statistic_matches_and_reads_nothing_lazily(both, stat):
    lazy, reference = both

    with Computes() as computed:
        assert STATS[stat](lazy) == STATS[stat](reference)

    assert computed.count == 0


def test_mosaic_accepts_lazy_inputs(both, tmp_path):
    lazy, reference = both

    with Computes() as computed:
        result = lazy.mosaic([lazy])

    assert computed.count == 0
    np.testing.assert_array_equal(result.read(), reference.mosaic([reference]).read())


def test_stack_accepts_lazy_inputs(both):
    lazy, reference = both

    result = lazy.stack([lazy])

    np.testing.assert_array_equal(result.read(), reference.stack([reference]).read())


def test_a_chain_of_ops_matches_the_rasterio_backend(both):
    lazy, reference = both

    def chain(ds):
        return ds.ndvi("red", nir="nir").add(1).normalize_min_max()

    with Computes() as computed:
        result = chain(lazy).read()

    assert computed.count == 0
    np.testing.assert_array_equal(result, chain(reference).read(), strict=True)


def test_saving_an_op_result_round_trips(both, tmp_path):
    lazy, reference = both
    out = tmp_path / "ndvi.tif"

    lazy.ndvi("red", nir="nir").save_raster(out)

    with rio.open(out) as saved:
        np.testing.assert_array_equal(
            saved.read(), reference.ndvi("red", nir="nir").read(), strict=True
        )


# ------------------------------------------------------------------- plots


@pytest.fixture
def large_scene_path(tmp_path):
    """600x600 single-band raster, larger than the tiny figures' display budget."""
    array = np.linspace(0.0, 1.0, LARGE_SIDE * LARGE_SIDE, dtype="float32").reshape(
        1, LARGE_SIDE, LARGE_SIDE
    )
    path = tmp_path / "large.tif"
    with rio.open(
        path,
        "w",
        driver="GTiff",
        height=LARGE_SIDE,
        width=LARGE_SIDE,
        count=1,
        dtype="float32",
        crs=CRS_UTM,
        transform=TRANSFORM,
    ) as dst:
        dst.write(array)
    return path


def test_plotting_reads_at_display_resolution_not_in_full(large_scene_path, monkeypatch):
    """Rule 4 of the memory model, on the lazy backend: plots decimate."""
    shapes = []
    original = RasterioAdapter.read

    def spy(self, *args, **kwargs):
        array = original(self, *args, **kwargs)
        shapes.append(np.shape(array))
        return array

    monkeypatch.setattr(RasterioAdapter, "read", spy)
    lazy = eeo.load_raster(large_scene_path, chunks=128)

    with Computes() as computed, warnings.catch_warnings():
        # The deliberately tiny figure cannot fit its decorations; that
        # cosmetic warning is irrelevant to the reads under test.
        warnings.filterwarnings("ignore", message="Tight layout not applied", category=UserWarning)
        lazy.plot_raster(figsize=(1, 1))

    assert computed.count == 0, "the lazy raster was read instead of the file"
    assert shapes, "nothing was read at all"
    assert all(max(shape[-2:]) < LARGE_SIDE for shape in shapes), shapes


# --------------------------------------------------- the NumPy backend still refuses


@pytest.mark.parametrize(
    "call",
    [
        lambda ds: ds.mosaic([ds]),
        lambda ds: ds.stack([ds]),
        lambda ds: ds.clip_raster_with_bbox((500100.0, 3998900.0, 500900.0, 3999900.0)),
        lambda ds: ds.reproject_raster(target_crs=4326),
    ],
)
def test_a_numpy_backed_dataset_is_still_refused(numpy_backed_dataset, call):
    """Promoting it would copy pixels that are already in memory."""
    with pytest.raises(BackendError, match="NumPy backend"):
        call(numpy_backed_dataset)
