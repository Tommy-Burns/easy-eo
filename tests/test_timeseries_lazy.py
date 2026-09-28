"""The temporal layer on the lazy backend (the ``lazy`` extra).

Two claims, the same pair test_lazy_ops.py makes for single-raster operations:
what a lazy series computes equals what the rasterio-backed series computes, and
it gets there without reading a raster whole. The second is what makes a series
of full scenes workable at all, and it is asserted directly — a dask computation
is the only way pixels can leave the lazy backend, and every windowed read is
counted.
"""

import datetime as dt

import numpy as np
import pytest
import rasterio as rio
from rasterio.crs import CRS
from rasterio.transform import from_origin

import eeo
from eeo.core.adapters import RasterioAdapter, XarrayAdapter

pytest.importorskip("dask.array")
pytest.importorskip("rioxarray")

UTC = dt.timezone.utc
# Large enough that a full read is unmistakable next to a windowed one, and
# past the 1 << 20 pixel block budget so a reduction runs in several strips.
SIDE = 1400
TRANSFORM = from_origin(500_000.0, 4_000_000.0, 10.0, 10.0)
CRS_UTM = CRS.from_epsg(32633)
MONTHS = (3, 4, 5)
FILLS = (100, 300, 200)


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


@pytest.fixture(scope="module")
def scene_paths(tmp_path_factory):
    """Three single-band scenes on one grid, each a uniform fill plus a gradient."""
    directory = tmp_path_factory.mktemp("lazy-series")
    paths = []
    for month, fill in zip(MONTHS, FILLS, strict=True):
        path = directory / f"scene_{month:02d}.tif"
        column = np.arange(SIDE, dtype="uint16")
        array = np.broadcast_to(column, (SIDE, SIDE)).astype("uint16") + fill
        with rio.open(
            path,
            "w",
            driver="GTiff",
            height=SIDE,
            width=SIDE,
            count=1,
            dtype="uint16",
            crs=CRS_UTM,
            transform=TRANSFORM,
            nodata=0,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dst:
            dst.write(array[np.newaxis])
            dst.set_band_description(1, "red")
        paths.append(path)
    return paths


def series(paths, chunks):
    """Build a series over ``paths`` on the backend ``chunks`` selects."""
    return eeo.time_series(
        [
            eeo.load_raster(
                path,
                chunks=chunks,
                timestamp=dt.datetime(2023, month, 1, tzinfo=UTC),
            )
            for month, path in zip(MONTHS, paths, strict=True)
        ]
    )


@pytest.fixture
def lazy(scene_paths):
    ts = series(scene_paths, "auto")
    yield ts
    ts.close()


@pytest.fixture
def eager(scene_paths):
    ts = series(scene_paths, None)
    yield ts
    ts.close()


def test_a_lazy_series_is_lazily_backed(lazy):
    assert all(isinstance(ds._adapter, XarrayAdapter) for ds in lazy)
    assert lazy.band_names == ["red"]


# --------------------------------------------------------------------------
# The two backends agree
# --------------------------------------------------------------------------
@pytest.mark.parametrize("how", ["median", "mean", "min", "max"])
def test_the_reducers_agree_across_backends(lazy, eager, how):
    lazily = getattr(lazy, how)().to_array()
    eagerly = getattr(eager, how)().to_array()

    np.testing.assert_array_equal(lazily, eagerly)


def test_mapping_then_reducing_agrees_across_backends(lazy, eager):
    lazily = lazy.map(eeo.multiply, other=2).median().to_array()
    eagerly = eager.map(eeo.multiply, other=2).median().to_array()

    np.testing.assert_allclose(lazily, eagerly)


def test_extraction_agrees_across_backends(lazy, eager):
    point = (500_005.0, 3_999_995.0)

    np.testing.assert_allclose(
        lazy.extract_at(point).to_numpy(), eager.extract_at(point).to_numpy()
    )


# --------------------------------------------------------------------------
# ...without reading a raster whole
# --------------------------------------------------------------------------
def test_a_reduction_reads_windows_rather_than_whole_rasters(lazy, monkeypatch):
    """A lazy series reduces through windowed reads of the files it came from.

    Promotion reopens the file instead of computing the dask array (WP-17's
    finding), so the reduction streams: every read is one block of one timestep,
    and no dask graph runs at all.
    """
    shapes = []
    original = RasterioAdapter.read

    def spy(self, *args, **kwargs):
        array = original(self, *args, **kwargs)
        shapes.append(np.shape(array))
        return array

    monkeypatch.setattr(RasterioAdapter, "read", spy)

    with Computes() as computed:
        result = lazy.median()

    assert computed.count == 0, "the lazy rasters were computed instead of read by window"
    assert shapes, "nothing was read at all"
    assert all(shape[-2] < SIDE for shape in shapes), (
        f"a read covered every row, so it was not windowed: {sorted(set(shapes))}"
    )
    assert result.get_shape() == (SIDE, SIDE)


def test_a_reduction_runs_in_several_blocks(lazy, monkeypatch):
    # 1400 x 1400 is past the block budget, so a streamed reduction reads each
    # timestep several times over, once per strip — three timesteps, N strips.
    reads = []
    original = RasterioAdapter.read

    def spy(self, *args, **kwargs):
        reads.append(kwargs.get("window"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(RasterioAdapter, "read", spy)

    lazy.median()

    windows = [window for window in reads if window is not None]
    assert len(windows) > len(MONTHS), "the reduction did not stream in blocks"
    assert all(window.height < SIDE for window in windows)


def test_extraction_reads_one_pixel_per_timestep(lazy, monkeypatch):
    shapes = []
    original = RasterioAdapter.read

    def spy(self, *args, **kwargs):
        array = original(self, *args, **kwargs)
        shapes.append(np.shape(array))
        return array

    monkeypatch.setattr(RasterioAdapter, "read", spy)

    with Computes() as computed:
        lazy.extract_at((500_005.0, 3_999_995.0))

    assert computed.count == 0
    assert shapes, "nothing was read at all"
    # One 1x1 window per timestep and band, never a band.
    assert all(np.prod(shape) == 1 for shape in shapes), shapes


def test_saving_a_mapped_lazy_series_keeps_it_lazy(lazy, tmp_path):
    result = lazy.map(eeo.multiply, other=2, save_dir=tmp_path / "doubled")

    assert all(isinstance(ds._adapter, XarrayAdapter) for ds in result)
    # And the reduction of those saved files still agrees with the arithmetic.
    assert result.median().to_array()[0, 0, 0] == pytest.approx((200 + 400 + 600) / 3, abs=1)
    result.close()


def test_a_reduction_of_a_lazy_series_can_stream_to_disk(lazy, tmp_path):
    out = tmp_path / "composite.tif"

    result = lazy.median(save_path=out)

    assert out.exists()
    with rio.open(out) as saved:
        assert saved.shape == (SIDE, SIDE)
        assert saved.dtypes == ("float32",)
    result.close()


def test_a_lazily_backed_but_unchunked_series_stays_lazy_when_saved(scene_paths, tmp_path):
    from eeo.core.core import EEORasterDataset

    # An XarrayAdapter over an in-memory DataArray: on the lazy backend, but
    # with no chunks to carry over, so the saved files are left to dask's own.
    unchunked = eeo.time_series(
        [
            EEORasterDataset(
                XarrayAdapter(eeo.load_raster(path, chunks="auto").ds.compute()),
                timestamp=dt.datetime(2023, month, 1, tzinfo=UTC),
            )
            for month, path in zip(MONTHS, scene_paths, strict=True)
        ]
    )

    result = unchunked.map(eeo.multiply, other=2, save_dir=tmp_path / "doubled")

    assert all(isinstance(ds._adapter, XarrayAdapter) for ds in result)
    assert all(ds._adapter.chunk_sizes is not None for ds in result)
    result.close()
    unchunked.close()
