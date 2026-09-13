"""The pixel-wise ops actually run block-wise, and blocking does not move them.

Task 16.2 routed algebra, the spectral indices, and min-max normalization
through :func:`eeo.core.blockwise.apply_blockwise`. Two things have to hold
afterwards, and the rest of the suite checks neither.

The first is that the ops really are windowed — every existing op test uses
rasters far under the block budget, where the engine resolves a single block
covering the whole raster, so they would pass just as well if nothing were
routed at all. These tests shrink the block shape instead of growing the
raster, which exercises the seams without a multi-megapixel fixture.

The second is that blocking does not change the answer. The values themselves
are pinned by ``test_ops.py``, ``test_indices.py`` and
``test_preprocessing.py``; what is checked here is that those answers are
independent of where the block boundaries fall.
"""

import numpy as np
import pytest
from rasterio.transform import from_origin

from eeo import load_array
from eeo.core import blockwise, streaming

UTM_CRS = 32633


def _force_block_shape(monkeypatch, block_shape):
    """Make every block-shape decision in the library return ``block_shape``.

    Patching the resolver rather than ``DEFAULT_BLOCK_PIXELS`` is deliberate:
    the constant is bound as a default argument when the module is imported,
    so rebinding the module attribute would have no effect at all. And it has
    to be patched in each module that imported the name, not only where it is
    defined: ``blockwise`` cuts the transformation passes and ``streaming``
    cuts the reduction passes, and patching one leaves the other on full-size
    blocks, quietly not exercising what these tests are about.
    """
    for module in (blockwise, streaming):
        monkeypatch.setattr(
            module, "resolve_block_shape", lambda shape, _b=block_shape, **kwargs: _b
        )


@pytest.fixture
def tiny_blocks(monkeypatch):
    """Force 2x2 blocks throughout, so a 6x6 raster is nine seamed windows."""
    _force_block_shape(monkeypatch, (2, 2))


@pytest.fixture
def windows_read(monkeypatch):
    """Record the window of every read the engine issues through a source."""
    seen = []
    original = blockwise.BlockSource.read

    def spy(self, window):
        seen.append(window)
        return original(self, window)

    monkeypatch.setattr(blockwise.BlockSource, "read", spy)
    return seen


def _grid(array, **kwargs):
    """Rasterio-backed dataset on the shared UTM grid."""
    return load_array(
        array, transform=from_origin(500_000.0, 4_200_000.0, 10.0, 10.0), crs=UTM_CRS, **kwargs
    ).to_rasterio()


@pytest.fixture
def scene():
    """6x6 two-band uint16 scene with nodata declared and present."""
    nir = np.arange(100, 136, dtype=np.uint16).reshape(6, 6)
    red = np.arange(1, 37, dtype=np.uint16).reshape(6, 6)
    stacked = np.stack([nir, red])
    stacked[:, 0, 0] = 0  # a declared-nodata pixel in both bands
    ds = _grid(stacked, nodata=0, band_names=["nir", "red"])
    yield ds
    ds.close()


@pytest.fixture
def single():
    """6x6 single-band float32 raster with nodata in the top-left 2x2."""
    array = np.arange(36, dtype=np.float32).reshape(6, 6)
    array[:2, :2] = -9999.0
    ds = _grid(array, nodata=-9999.0)
    yield ds
    ds.close()


#: One entry per routed operation: a name, and a callable running it on the
#: ``scene`` fixture. Parametrising over this is what keeps a newly routed op
#: from quietly escaping both checks below.
ROUTED_OPS = [
    pytest.param(lambda ds: ds.add(5), id="add"),
    pytest.param(lambda ds: ds.subtract(5), id="subtract"),
    pytest.param(lambda ds: ds.multiply(3), id="multiply"),
    pytest.param(lambda ds: ds.divide(3), id="divide"),
    pytest.param(lambda ds: ds.power(2), id="power"),
    pytest.param(lambda ds: ds.sqrt(), id="sqrt"),
    pytest.param(lambda ds: ds.log(), id="log"),
    pytest.param(lambda ds: ds.absolute(), id="absolute"),
    pytest.param(lambda ds: ds.normalize_min_max(), id="normalize_min_max"),
    pytest.param(lambda ds: ds.ndvi("red", nir="nir"), id="ndvi"),
    pytest.param(lambda ds: ds.ndwi("nir", green="red"), id="ndwi"),
    pytest.param(lambda ds: ds.savi("red", nir="nir"), id="savi"),
    pytest.param(lambda ds: ds.evi("red", nir="nir", blue="red"), id="evi"),
    pytest.param(lambda ds: ds.normalized_difference(ds), id="normalized_difference"),
]


class TestOpsAreWindowed:
    @pytest.mark.parametrize("run", ROUTED_OPS)
    def test_the_op_reads_in_windows_not_in_one_gulp(self, scene, run, tiny_blocks, windows_read):
        # A 6x6 raster in 2x2 blocks is nine windows, and every source is read
        # once per window. An op that had not been routed would read no window
        # at all, because it would never reach a BlockSource.
        result = run(scene)
        assert windows_read, "the op read no window: it is not routed through the engine"
        assert len({(w.row_off, w.col_off) for w in windows_read}) == 9
        assert all(w.height <= 2 and w.width <= 2 for w in windows_read)
        result.close()

    @pytest.mark.parametrize("run", ROUTED_OPS)
    def test_the_block_shape_does_not_change_the_answer(self, scene, run, monkeypatch):
        # The raster fits one default block, so this compares the whole-raster
        # run against runs seamed in three different places.
        whole = run(scene).read()
        for block_shape in [(2, 2), (1, 6), (4, 5)]:
            _force_block_shape(monkeypatch, block_shape)
            assert np.array_equal(run(scene).read(), whole, equal_nan=True), block_shape


class TestNodataSurvivesTheSeams:
    def test_a_nodata_pixel_stays_nodata_whatever_block_holds_it(
        self, single, tiny_blocks, windows_read
    ):
        # The nodata square spans four 2x2 blocks, so the mask has to be right
        # in each of them independently.
        result = single.multiply(2)
        assert np.isnan(result.get_metadata()["nodata"])
        assert np.isnan(result.read()[0, :2, :2]).all()
        assert not np.isnan(result.read()[0, 2:, :]).any()

    def test_an_index_masks_nodata_from_either_band(self, scene, tiny_blocks):
        result = scene.ndvi("red", nir="nir")
        assert np.isnan(result.read()[0, 0, 0])
        assert not np.isnan(result.read()[0, 0, 1:]).any()


class TestBranchesThatOnlyRunOnSomeBlocks:
    """Op paths whose correctness is per-block, so seams are the risk."""

    def test_unsafe_divide_lets_numpy_semantics_through_every_block(
        self, single, tiny_blocks, windows_read
    ):
        # safe=False is documented as following NumPy: a zero denominator
        # yields inf and warns. The warning has to survive being raised inside
        # a per-block closure rather than over the whole array.
        with pytest.warns(RuntimeWarning, match="divide by zero"):
            result = single.divide(0, safe=False)
        try:
            assert windows_read, "divide did not go through the engine"
            valid = ~np.isnan(result.read())
            assert np.isinf(result.read()[valid]).all()
        finally:
            result.close()

    def test_safe_divide_by_zero_is_zero_in_every_block(self, single, tiny_blocks):
        result = single.divide(0)
        try:
            # Nodata stays nodata; every valid pixel is the guarded 0.
            assert np.isnan(result.read()[0, :2, :2]).all()
            assert (result.read()[0, 2:, :] == 0).all()
        finally:
            result.close()

    def test_auto_align_resamples_the_operand_before_the_blocks_are_cut(
        self, shape_mismatch_pair, tiny_blocks, windows_read
    ):
        # The operand is on a coarser grid, so it must be resampled onto the
        # target's grid first — the engine reads the two in step window by
        # window and would otherwise be reading mismatched pixels.
        fine, coarse = shape_mismatch_pair
        result = fine.add(coarse)
        try:
            assert result.get_shape() == fine.get_shape()
            assert result.get_transform() == fine.get_transform()
            assert windows_read, "add did not go through the engine"
            # add() aligns with method="bilinear" by default, not resample's
            # own "nearest", so the reference has to ask for the same one.
            aligned = coarse.resample(size=fine.get_shape(), resampling_method="bilinear")
            expected = fine.read() + aligned.read()
            assert np.array_equal(result.read(), expected)
        finally:
            result.close()

    def test_normalized_difference_aligns_its_operand_too(
        self, shape_mismatch_pair, tiny_blocks, windows_read
    ):
        # normalized_difference keeps its own body rather than going through
        # _compute_index, so its alignment branch is a separate path.
        fine, coarse = shape_mismatch_pair
        result = fine.normalized_difference(coarse)
        try:
            assert result.get_shape() == fine.get_shape()
            assert windows_read, "normalized_difference did not go through the engine"
            aligned = coarse.resample(size=fine.get_shape(), resampling_method="bilinear")
            a, b = fine.read().astype(np.float32), aligned.read().astype(np.float32)
            total = a + b
            with np.errstate(divide="ignore", invalid="ignore"):
                expected = np.where(total != 0, (a - b) / total, np.float32(0)).astype(np.float32)
            assert np.array_equal(result.read(), expected, equal_nan=True)
        finally:
            result.close()

    def test_a_grid_mismatch_without_auto_align_still_raises(self, shape_mismatch_pair):
        from eeo.core.exceptions import AlignmentError

        fine, coarse = shape_mismatch_pair
        with pytest.raises(AlignmentError, match="same grid"):
            fine.add(coarse, auto_align=False)
