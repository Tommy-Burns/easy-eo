"""Tests for the lazy, xarray-backed adapter (eeo.core.adapters.XarrayAdapter).

The rasterio backend is the reference: the same file opened both ways must
report the same metadata and return the same pixels.
"""

import importlib
import warnings

import numpy as np
import pytest
import rasterio as rio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.windows import Window

import eeo
from eeo import _optional, load_raster
from eeo.core.adapters import RasterioAdapter, XarrayAdapter
from eeo.core.exceptions import BackendError, MissingDependencyError, ValidationError

# 60 x 80 pixels, so chunks of 20 split every axis unevenly into several pieces.
HEIGHT, WIDTH = 60, 80
TRANSFORM = from_origin(500000.0, 4000000.0, 10.0, 10.0)
CRS_UTM = CRS.from_epsg(32633)


def _write(path, array, *, nodata=None, descriptions=None):
    count, height, width = array.shape
    with rio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=array.dtype,
        crs=CRS_UTM,
        transform=TRANSFORM,
        nodata=nodata,
    ) as dst:
        dst.write(array)
        for i, name in enumerate(descriptions or [], start=1):
            if name:
                dst.set_band_description(i, name)
    return path


@pytest.fixture
def lazy_extra():
    """Skip unless the lazy extra is installed."""
    pytest.importorskip("dask.array")
    pytest.importorskip("rioxarray")


@pytest.fixture
def stack_path(tmp_path):
    """3-band uint16 GeoTIFF with nodata=0 in a corner and two named bands."""
    array = np.arange(3 * HEIGHT * WIDTH, dtype=np.uint16).reshape(3, HEIGHT, WIDTH) + 1
    array[:, :5, :5] = 0
    return _write(tmp_path / "stack.tif", array, nodata=0, descriptions=["red", None, "nir"])


@pytest.fixture
def float_path(tmp_path):
    """Single-band float32 GeoTIFF with NaN nodata and no band description."""
    array = np.linspace(0, 1, HEIGHT * WIDTH, dtype=np.float32).reshape(1, HEIGHT, WIDTH)
    array[0, 10:20, 10:20] = np.nan
    return _write(tmp_path / "float.tif", array, nodata=np.nan)


# --------------------------------------------------------------- opening


@pytest.mark.parametrize("chunks", ["auto", 20, {"y": 20, "x": 30}, {"band": 1}])
def test_chunks_select_the_lazy_backend(lazy_extra, stack_path, chunks):
    import dask.array

    ds = load_raster(stack_path, chunks=chunks)

    assert isinstance(ds._adapter, XarrayAdapter)
    assert isinstance(ds.ds.data, dask.array.Array)
    assert ds.path == stack_path


def test_auto_chunks_match_rioxarrays_own_and_warn_about_nothing(lazy_extra, stack_path):
    """``chunks="auto"`` must mean what rioxarray means by it, quietly.

    rioxarray resolves ``"auto"`` to a dimension-order tuple and hands that to
    ``DataArray.chunk``, which xarray warns about; versions that stopped doing
    it need Python 3.12, so on 3.10 and 3.11 the warning is unavoidable from
    the outside. Easy-EO therefore resolves ``"auto"`` itself, to the same
    sizes — which is what this test pins, in both directions.

    The warning is matched by its message rather than its category on purpose:
    xarray raises it as a ``FutureWarning`` where the version before it used a
    ``DeprecationWarning``, and either would fail the suite's warning gate.
    """
    import rioxarray

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ours = load_raster(stack_path, chunks="auto")

    assert [str(w.message) for w in caught if "chunks as dimension-order" in str(w.message)] == []

    with warnings.catch_warnings():
        # The reference call is the one that warns; that is the point.
        warnings.simplefilter("ignore")
        reference = rioxarray.open_rasterio(stack_path, chunks="auto")

    assert ours.ds.data.chunks == reference.data.chunks


def test_without_chunks_the_backend_is_unchanged(stack_path):
    assert isinstance(load_raster(stack_path)._adapter, RasterioAdapter)


def test_opening_and_reading_metadata_computes_nothing(lazy_extra, stack_path):
    from dask.callbacks import Callback

    computed = []
    with Callback(start=lambda dsk: computed.append(dsk)):
        ds = load_raster(stack_path, chunks=20)
        ds.get_metadata()
        ds.get_bounds()
        ds.band_names  # noqa: B018
        repr(ds)
        ds.describe()

    assert computed == []


def test_dataset_wrappers_apply_on_the_lazy_backend(lazy_extra, stack_path):
    ds = load_raster(stack_path, chunks=20, attrs={"site": "a"}, band_names=["b", "g", "r"])

    assert ds.attrs == {"site": "a"}
    assert ds.band_names == ["b", "g", "r"]


def test_a_file_that_is_not_a_raster_raises_backend_error(lazy_extra, tmp_path):
    bad = tmp_path / "not_a_raster.tif"
    bad.write_text("not a raster")

    with pytest.raises(BackendError, match="could not be opened"):
        load_raster(bad, chunks="auto")


@pytest.mark.parametrize(
    "chunks", [True, False, 0, -5, 1.5, "big", {"z": 10}, {"x": 0}, {"y": "20"}, [20, 20]]
)
def test_invalid_chunks_raise_validation_error(stack_path, chunks):
    """Checked before anything is imported, so it holds without the extra too."""
    with pytest.raises(ValidationError, match="chunks must be"):
        load_raster(stack_path, chunks=chunks)


def test_missing_extra_names_the_lazy_extra(monkeypatch, stack_path):
    real_import = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name.split(".")[0] in {"rioxarray", "dask"}:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(_optional, "_installed_by_conda", lambda: False)

    with pytest.raises(MissingDependencyError) as excinfo:
        load_raster(stack_path, chunks="auto")

    assert "pip install 'easy-eo[lazy]'" in str(excinfo.value)


def test_constructor_rejects_other_dimensions(lazy_extra):
    import xarray as xr

    with pytest.raises(ValidationError, match="dimensions"):
        XarrayAdapter(xr.DataArray(np.zeros((2, 3)), dims=("y", "x")))


def test_reading_an_in_memory_dataarray_hands_back_a_copy(lazy_extra, stack_path):
    """A DataArray that is not dask-backed hands out views of its own buffer.

    The adapter copies in that case, so writing into what ``read()`` returned
    cannot reach back into the dataset it came from.
    """
    loaded = load_raster(stack_path, chunks=20).ds.compute()  # dask -> in memory
    adapter = XarrayAdapter(loaded)

    band = adapter.read(1)
    band[:] = 7

    assert not np.shares_memory(band, loaded.data)
    np.testing.assert_array_equal(adapter.read(1), load_raster(stack_path).read(1))


# -------------------------------------------------------------- metadata


@pytest.mark.parametrize("fixture", ["stack_path", "float_path"])
def test_metadata_matches_the_rasterio_backend(lazy_extra, request, fixture):
    path = request.getfixturevalue(fixture)
    lazy = load_raster(path, chunks=20)
    reference = load_raster(path)

    assert lazy.get_crs() == reference.get_crs()
    assert lazy.get_transform() == reference.get_transform()
    assert lazy.get_bounds() == reference.get_bounds()
    assert lazy.get_shape() == reference.get_shape() == (HEIGHT, WIDTH)
    assert lazy.get_width() == reference.get_width()
    assert lazy.get_height() == reference.get_height()
    assert lazy.get_count() == reference.get_count()
    assert lazy.band_names == reference.band_names

    lazy_meta, reference_meta = lazy.get_metadata(), reference.get_metadata()
    lazy_nodata = lazy_meta.pop("nodata")
    reference_nodata = reference_meta.pop("nodata")
    assert lazy_meta == reference_meta
    assert type(lazy_nodata) is type(reference_nodata)
    np.testing.assert_equal(lazy_nodata, reference_nodata)


def test_band_names_come_from_the_file(lazy_extra, stack_path, float_path):
    assert load_raster(stack_path, chunks=20).band_names == ["red", None, "nir"]
    assert load_raster(float_path, chunks=20).band_names == [None]


def test_a_single_named_band_is_read_as_a_name_not_characters(lazy_extra, tmp_path):
    """rioxarray stores one band's description as a plain string."""
    path = _write(tmp_path / "one.tif", np.ones((1, 4, 4), dtype=np.uint8), descriptions=["ndvi"])

    assert load_raster(path, chunks="auto").band_names == ["ndvi"]


def test_a_raster_without_nodata_reports_none(lazy_extra, tmp_path):
    path = _write(tmp_path / "plain.tif", np.ones((1, 4, 4), dtype=np.uint8))

    assert load_raster(path, chunks="auto").get_metadata()["nodata"] is None


# ------------------------------------------------------------------ reads


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        ((), {}),
        ((2,), {}),
        (([3, 1],), {}),
        ((), {"window": Window(col_off=15, row_off=7, width=30, height=41)}),
        ((1,), {"window": ((0, 60), (79, 80))}),
        (([2, 3],), {"window": Window(0, 0, 80, 60)}),
    ],
)
def test_reads_match_the_rasterio_backend(lazy_extra, stack_path, args, kwargs):
    lazy = load_raster(stack_path, chunks=20).read(*args, **kwargs)
    reference = load_raster(stack_path).read(*args, **kwargs)

    assert isinstance(lazy, np.ndarray)
    assert lazy.dtype == reference.dtype
    np.testing.assert_array_equal(lazy, reference)


def test_nan_nodata_is_read_as_stored(lazy_extra, float_path):
    np.testing.assert_array_equal(
        load_raster(float_path, chunks=20).get_band(1), load_raster(float_path).get_band(1)
    )


def test_get_band_resolves_names(lazy_extra, stack_path):
    ds = load_raster(stack_path, chunks=20)

    np.testing.assert_array_equal(ds.get_band("nir"), load_raster(stack_path).read(3))


def test_a_window_read_computes_only_the_chunks_it_touches(lazy_extra, stack_path):
    from dask.callbacks import Callback

    def tasks_run(read):
        count = []
        with Callback(pretask=lambda key, dsk, state: count.append(key)):
            read()
        return len(count)

    ds = load_raster(stack_path, chunks={"band": 1, "y": 20, "x": 20})
    inside_one_chunk = tasks_run(lambda: ds.read(1, window=Window(2, 2, 10, 10)))
    whole = tasks_run(ds.read)

    assert 0 < inside_one_chunk < whole


def test_the_result_does_not_alias_the_backend(lazy_extra, stack_path):
    ds = load_raster(stack_path, chunks=20)

    band = ds.read(1)
    band[:] = 7

    np.testing.assert_array_equal(ds.read(1), load_raster(stack_path).read(1))


def test_out_of_range_band_raises_index_error(lazy_extra, stack_path):
    ds = load_raster(stack_path, chunks=20)

    with pytest.raises(IndexError, match="out of range"):
        ds.read(4)
    with pytest.raises(IndexError, match="out of range"):
        ds.read([1, 0])


def test_unsupported_read_options_raise_backend_error(lazy_extra, stack_path):
    ds = load_raster(stack_path, chunks=20)

    with pytest.raises(BackendError, match="out_shape"):
        ds.read(1, out_shape=(10, 10))


@pytest.mark.parametrize(
    "window",
    [
        Window(0, 0, 10.5, 10),
        Window(-1, 0, 10, 10),
        Window(75, 0, 10, 10),
        Window(0, 55, 10, 10),
    ],
)
def test_bad_windows_raise_validation_error(lazy_extra, stack_path, window):
    ds = load_raster(stack_path, chunks=20)

    with pytest.raises(ValidationError, match="window"):
        ds.read(1, window=window)


# ------------------------------------------------------------ persistence


@pytest.mark.parametrize("fixture", ["stack_path", "float_path"])
def test_save_round_trips_across_chunks(lazy_extra, request, tmp_path, fixture):
    path = request.getfixturevalue(fixture)
    ds = load_raster(path, chunks={"band": 1, "y": 25, "x": 30})
    out = tmp_path / "out.tif"

    ds.save_raster(out)

    with rio.open(path) as src, rio.open(out) as dst:
        np.testing.assert_array_equal(dst.read(), src.read())
        assert dst.dtypes == src.dtypes
        np.testing.assert_equal(dst.nodata, src.nodata)
        assert dst.crs == src.crs
        assert dst.transform == src.transform
        assert dst.descriptions == src.descriptions


def test_save_writes_renamed_bands(lazy_extra, stack_path, tmp_path):
    ds = load_raster(stack_path, chunks=20)
    ds.set_band_name(2, "green")
    out = tmp_path / "renamed.tif"

    ds.save_raster(out)

    assert eeo.load_raster(out).band_names == ["red", "green", "nir"]


# ----------------------------------------------------- dataset behaviour


def test_to_rasterio_and_to_array_hold_the_same_pixels(lazy_extra, stack_path):
    ds = load_raster(stack_path, chunks=20)
    reference = load_raster(stack_path).read()

    promoted = ds.to_rasterio()

    assert isinstance(promoted._adapter, RasterioAdapter)
    assert promoted.band_names == ["red", None, "nir"]
    np.testing.assert_array_equal(promoted.read(), reference)
    np.testing.assert_array_equal(ds.to_array(), reference)


def test_describe_with_exact_stats_matches_the_rasterio_backend(lazy_extra, stack_path, capsys):
    load_raster(stack_path).describe(stats="exact")
    reference = capsys.readouterr().out
    load_raster(stack_path, chunks=20).describe(stats="exact")

    assert capsys.readouterr().out == reference


def test_close_is_idempotent(lazy_extra, stack_path):
    ds = load_raster(stack_path, chunks=20)

    ds.close()
    ds.close()
