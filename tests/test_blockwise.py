"""Tests for the block-wise execution engine.

The engine's whole claim is that splitting a raster into blocks changes
nothing about the answer. So most of these tests fix a small raster, run the
same computation at several block shapes — including shapes that divide the
raster unevenly, so the last row and column of blocks are truncated — and
assert every run produces the identical array. A result that depended on where
the seams fell would show up as a disagreement between those runs.

The rest pin what the engine has to decide *before* it can see the data: the
output dtype, the output nodata value, and the block shape itself.
"""

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin

from eeo import load_array
from eeo.core.blockwise import (
    DEFAULT_BLOCK_PIXELS,
    BlockSource,
    apply_blockwise,
    block_windows,
    resolve_block_shape,
)
from eeo.core.exceptions import AlignmentError, ValidationError

UTM_CRS = CRS.from_epsg(32633)

# Block shapes that tile a 6x6 raster differently: the whole raster at once,
# a single row, a single pixel, and two shapes that leave a short final block
# in one or both directions.
UNEVEN_BLOCK_SHAPES = [(6, 6), (1, 6), (1, 1), (4, 4), (5, 2), (3, 7)]


def _sum_blocks(*blocks):
    """Pixel-wise sum of every block, the simplest possible computation."""
    return sum(blocks[1:], start=blocks[0])


class TestResolveBlockShape:
    def test_a_small_raster_is_a_single_block(self):
        # Under the budget there is nothing to gain from splitting, and a
        # single block keeps small rasters free of per-window overhead.
        assert resolve_block_shape((64, 64)) == (64, 64)

    def test_blocks_span_the_full_width(self):
        height, width = 10_980, 10_980
        block_height, block_width = resolve_block_shape((height, width))
        assert block_width == width
        assert block_height < height

    def test_a_block_stays_within_the_pixel_budget(self):
        # A Sentinel-2 10 m band: the block must be a strip small enough to
        # honour the budget, not the 120-megapixel scene.
        block_height, block_width = resolve_block_shape((10_980, 10_980))
        assert block_height * block_width <= DEFAULT_BLOCK_PIXELS

    def test_a_raster_wider_than_the_budget_still_yields_whole_rows(self):
        # A block is never narrower than the raster, so an extremely wide
        # raster degrades to one row per block rather than to zero rows.
        assert resolve_block_shape((10, 5_000_000)) == (1, 5_000_000)

    def test_the_budget_is_adjustable(self):
        assert resolve_block_shape((1000, 100), target_pixels=1000) == (10, 100)


class TestBlockWindows:
    @pytest.mark.parametrize("block_shape", UNEVEN_BLOCK_SHAPES)
    def test_the_windows_partition_the_raster_exactly(self, block_shape):
        # Every pixel covered once: no gap (which would leave the output
        # unwritten) and no overlap (which would compute a pixel twice).
        covered = np.zeros((6, 6), dtype=int)
        for window in block_windows((6, 6), block_shape):
            rows = slice(window.row_off, window.row_off + window.height)
            cols = slice(window.col_off, window.col_off + window.width)
            covered[rows, cols] += 1
        assert np.array_equal(covered, np.ones((6, 6), dtype=int))

    def test_edge_windows_are_truncated_to_the_raster(self):
        windows = list(block_windows((5, 5), (4, 4)))
        assert [(w.height, w.width) for w in windows] == [(4, 4), (4, 1), (1, 4), (1, 1)]

    @pytest.mark.parametrize("block_shape", [(0, 4), (4, 0), (-1, 4)])
    def test_a_non_positive_block_shape_is_rejected(self, block_shape):
        with pytest.raises(ValidationError, match="positive in both dimensions"):
            list(block_windows((6, 6), block_shape))


class TestBlockSource:
    def test_a_scalar_source_yields_its_value_for_any_window(self):
        source = BlockSource.from_scalar(3)
        assert not source.is_raster
        assert source.nodata is None
        assert source.read(next(block_windows((6, 6), (2, 2)))) == 3

    def test_a_dataset_source_reads_only_its_window(self, single_band_float32):
        source = BlockSource.from_dataset(single_band_float32)
        window = next(block_windows((6, 6), (2, 3)))
        block = source.read(window)
        assert block.shape == (1, 2, 3)
        assert np.array_equal(block[0], single_band_float32.read()[0][:2, :3])

    def test_a_band_source_reads_one_band_as_a_2d_block(self, multiband_uint16):
        source = BlockSource.from_dataset(multiband_uint16, band=2)
        block = source.read(next(block_windows((6, 6), (2, 3))))
        assert block.shape == (2, 3)
        assert np.array_equal(block, multiband_uint16.read()[1][:2, :3])

    def test_a_band_source_accepts_a_band_name(self, multiband_uint16):
        # A name works wherever a band index does; this is the public entry
        # point for running your own function block-wise, so it must too.
        multiband_uint16.band_names = ["blue", "green", "red", "nir"]
        by_name = BlockSource.from_dataset(multiband_uint16, band="red")
        window = next(block_windows((6, 6), (2, 3)))

        assert np.array_equal(
            by_name.read(window),
            BlockSource.from_dataset(multiband_uint16, band=3).read(window),
        )

    def test_an_unknown_band_name_is_rejected(self, multiband_uint16):
        with pytest.raises(ValidationError, match="no band named"):
            BlockSource.from_dataset(multiband_uint16, band="swir")

    def test_a_numpy_backed_source_is_promoted_so_windows_are_honoured(self, numpy_backed_dataset):
        # The NumPy adapter's read() ignores a window and would hand back the
        # whole array for every block, silently computing the wrong answer.
        source = BlockSource.from_dataset(numpy_backed_dataset)
        block = source.read(next(block_windows((6, 6), (2, 2))))
        assert block.shape == (1, 2, 2)

    def test_a_dataset_source_carries_its_declared_nodata(self, raster_with_nodata):
        assert BlockSource.from_dataset(raster_with_nodata).nodata == -9999.0


class TestBlockingChangesNothing:
    @pytest.mark.parametrize("block_shape", UNEVEN_BLOCK_SHAPES)
    def test_scalar_arithmetic_matches_the_eager_op(self, single_band_float32, block_shape):
        eager = single_band_float32.multiply(3)
        blocked = apply_blockwise(
            single_band_float32,
            lambda block, factor: block * factor,
            sources=[
                BlockSource.from_dataset(single_band_float32),
                BlockSource.from_scalar(3),
            ],
            block_shape=block_shape,
        )
        assert np.array_equal(blocked.read(), eager.read())
        assert blocked.read().dtype == eager.read().dtype

    @pytest.mark.parametrize("block_shape", UNEVEN_BLOCK_SHAPES)
    def test_two_raster_operands_match_the_eager_op(self, single_band_float32, block_shape):
        eager = single_band_float32.add(single_band_float32)
        blocked = apply_blockwise(
            single_band_float32,
            _sum_blocks,
            sources=[
                BlockSource.from_dataset(single_band_float32),
                BlockSource.from_dataset(single_band_float32),
            ],
            block_shape=block_shape,
        )
        assert np.array_equal(blocked.read(), eager.read())

    @pytest.mark.parametrize("block_shape", UNEVEN_BLOCK_SHAPES)
    def test_nodata_pixels_survive_every_seam(self, raster_with_nodata, block_shape):
        eager = raster_with_nodata.multiply(2)
        blocked = apply_blockwise(
            raster_with_nodata,
            lambda block, factor: block * factor,
            sources=[
                BlockSource.from_dataset(raster_with_nodata),
                BlockSource.from_scalar(2),
            ],
            block_shape=block_shape,
        )
        assert np.array_equal(blocked.read(), eager.read(), equal_nan=True)
        # Both are NaN, which never compares equal to itself.
        assert np.isnan(blocked.get_metadata()["nodata"])
        assert np.isnan(eager.get_metadata()["nodata"])

    @pytest.mark.parametrize("block_shape", UNEVEN_BLOCK_SHAPES)
    def test_a_multiband_raster_keeps_its_bands_aligned(self, multiband_uint16, block_shape):
        blocked = apply_blockwise(
            multiband_uint16,
            lambda block: block + 1,
            sources=[BlockSource.from_dataset(multiband_uint16)],
            block_shape=block_shape,
        )
        assert np.array_equal(blocked.read(), multiband_uint16.read() + 1)

    def test_a_non_square_raster_is_not_transposed(self, nonsquare_float32):
        # A height/width swap in the window arithmetic would either crash or
        # scramble values; a 4x8 raster makes it observable.
        blocked = apply_blockwise(
            nonsquare_float32,
            lambda block: block * 2,
            sources=[BlockSource.from_dataset(nonsquare_float32)],
            block_shape=(3, 3),
        )
        assert blocked.get_shape() == (4, 8)
        assert np.array_equal(blocked.read(), nonsquare_float32.read() * 2)


class TestOutputContract:
    def test_the_georeferencing_is_carried_through(self, single_band_float32):
        blocked = apply_blockwise(
            single_band_float32,
            lambda block: block,
            sources=[BlockSource.from_dataset(single_band_float32)],
            block_shape=(4, 4),
        )
        assert blocked.get_crs() == single_band_float32.get_crs()
        assert blocked.get_transform() == single_band_float32.get_transform()
        assert blocked.get_shape() == single_band_float32.get_shape()

    def test_a_fractional_op_is_float32_whatever_the_input_dtype(self, multiband_uint16):
        blocked = apply_blockwise(
            multiband_uint16,
            lambda block: block / 2,
            sources=[BlockSource.from_dataset(multiband_uint16)],
            fractional=True,
            block_shape=(4, 4),
        )
        assert blocked.read().dtype == np.float32

    def test_exact_arithmetic_keeps_the_promoted_integer_dtype(self, multiband_uint16):
        blocked = apply_blockwise(
            multiband_uint16,
            lambda block: block + 1,
            sources=[BlockSource.from_dataset(multiband_uint16)],
            block_shape=(4, 4),
        )
        assert blocked.read().dtype == np.uint16

    def test_an_op_reducing_the_band_count_sizes_the_output_from_the_result(self, multiband_uint16):
        # The band count comes from what compute returns, not from the input,
        # which is what lets a two-band index write a one-band output.
        blocked = apply_blockwise(
            multiband_uint16,
            lambda nir, red: (nir - red) / (nir + red),
            sources=[
                BlockSource.from_dataset(multiband_uint16, band=4),
                BlockSource.from_dataset(multiband_uint16, band=1),
            ],
            fractional=True,
            block_shape=(4, 4),
        )
        assert blocked.get_count() == 1

    def test_no_declared_nodata_means_no_output_nodata(self, single_band_float32):
        blocked = apply_blockwise(
            single_band_float32,
            lambda block: block + 1,
            sources=[BlockSource.from_dataset(single_band_float32)],
            block_shape=(4, 4),
        )
        assert blocked.get_metadata()["nodata"] is None

    def test_an_integer_output_keeps_the_input_sentinel(self):
        array = np.arange(36, dtype=np.int16).reshape(6, 6)
        array[0, 0] = -32768
        ds = load_array(
            array,
            transform=from_origin(500_000.0, 4_200_000.0, 10.0, 10.0),
            crs=UTM_CRS,
            nodata=-32768,
        ).to_rasterio()
        blocked = apply_blockwise(
            ds,
            lambda block: block + 1,
            sources=[BlockSource.from_dataset(ds)],
            block_shape=(4, 4),
        )
        assert blocked.get_metadata()["nodata"] == -32768
        assert blocked.read()[0, 0, 0] == -32768
        ds.close()

    def test_a_block_free_of_nodata_still_uses_the_whole_output_s_nodata(self, raster_with_nodata):
        # The nodata pixels sit in the top-left 2x2, so with 2x2 blocks only
        # the first block contains any. Deriving the output nodata per block
        # would leave the rest of the output declaring none.
        blocked = apply_blockwise(
            raster_with_nodata,
            lambda block: block + 1,
            sources=[BlockSource.from_dataset(raster_with_nodata)],
            block_shape=(2, 2),
        )
        assert np.isnan(blocked.get_metadata()["nodata"])
        assert np.isnan(blocked.read()[0, :2, :2]).all()
        assert not np.isnan(blocked.read()[0, 2:, :]).any()

    def test_the_output_driver_is_chosen_not_inherited(self, single_band_float32):
        # The source's driver records how it was read, and a GDAL driver need
        # not support creating a dataset, so the output picks its own.
        blocked = apply_blockwise(
            single_band_float32,
            lambda block: block,
            sources=[BlockSource.from_dataset(single_band_float32)],
            block_shape=(4, 4),
            driver="GTiff",
        )
        assert blocked.get_metadata()["driver"] == "GTiff"


class TestSavePath:
    def test_streaming_to_disk_gives_the_same_raster(self, raster_with_nodata, tmp_path):
        path = tmp_path / "streamed.tif"
        in_memory = apply_blockwise(
            raster_with_nodata,
            lambda block: block * 2,
            sources=[BlockSource.from_dataset(raster_with_nodata)],
            block_shape=(2, 2),
        )
        streamed = apply_blockwise(
            raster_with_nodata,
            lambda block: block * 2,
            sources=[BlockSource.from_dataset(raster_with_nodata)],
            block_shape=(2, 2),
            save_path=path,
        )
        assert path.exists()
        assert np.array_equal(streamed.read(), in_memory.read(), equal_nan=True)
        assert streamed.get_transform() == in_memory.get_transform()
        streamed.close()


class TestRejections:
    def test_an_operand_on_another_grid_is_rejected(self, shape_mismatch_pair):
        fine, coarse = shape_mismatch_pair
        with pytest.raises(AlignmentError, match="same grid"):
            apply_blockwise(
                fine,
                _sum_blocks,
                sources=[BlockSource.from_dataset(fine), BlockSource.from_dataset(coarse)],
            )

    def test_no_sources_is_rejected(self, single_band_float32):
        with pytest.raises(ValidationError, match="at least one source"):
            apply_blockwise(single_band_float32, lambda: None, sources=[])

    def test_a_result_narrower_than_its_operands_is_rejected_clearly(self):
        # Reducing four bands to one while still masking against all four
        # broadcasts the mask back up to four. Rasterio would reject the write
        # from inside itself; the engine should explain why first. Needs a
        # declared nodata, since with none there is no mask to broadcast.
        bands = np.stack(
            [i * 1000 + np.arange(36, dtype=np.uint16).reshape(6, 6) for i in range(4)]
        )
        ds = load_array(
            bands.astype(np.uint16),
            transform=from_origin(500_000.0, 4_200_000.0, 10.0, 10.0),
            crs=UTM_CRS,
            nodata=0,
        ).to_rasterio()
        try:
            with pytest.raises(ValidationError, match="masking widened the result"):
                apply_blockwise(
                    ds,
                    lambda block: block[:1],
                    sources=[BlockSource.from_dataset(ds)],
                    block_shape=(4, 4),
                )
        finally:
            ds.close()

    def test_a_failing_computation_propagates(self, single_band_float32):
        def explode(block):
            raise ZeroDivisionError("boom")

        with pytest.raises(ZeroDivisionError, match="boom"):
            apply_blockwise(
                single_band_float32,
                explode,
                sources=[BlockSource.from_dataset(single_band_float32)],
                block_shape=(2, 2),
            )

    def test_a_failure_after_the_output_is_opened_propagates(self, single_band_float32):
        # The destination is opened on the first block, so failing on a later
        # block exercises the path that has a half-written output to close.
        calls = []

        def explode_on_second(block):
            calls.append(1)
            if len(calls) > 1:
                raise ZeroDivisionError("boom")
            return block

        with pytest.raises(ZeroDivisionError, match="boom"):
            apply_blockwise(
                single_band_float32,
                explode_on_second,
                sources=[BlockSource.from_dataset(single_band_float32)],
                block_shape=(2, 2),
            )
