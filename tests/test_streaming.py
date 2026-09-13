"""Streaming reductions, and the ops built on them.

A reduction that streams is only worth having if it gives the same answer as
the whole-array form it replaces. So nearly every test here is differential:
it computes the statistic by streaming, computes it again with NumPy over the
entire array, and demands they agree — usually exactly. No expected numbers are
written down, which is what lets the same assertions run over random data of
several dtypes at several block shapes.

The block shapes matter more than the raster sizes. Each case is run at shapes
that divide the raster unevenly and at one a single pixel wide, so the last row
and column of blocks are truncated and every seam falls somewhere different. A
reduction that lost a partial block, double-counted an overlap, or broke a tie
by block rather than by position would disagree with NumPy at one of them.
"""

import numpy as np
import pytest
from rasterio.transform import from_origin

import eeo
from eeo.core import blockwise, streaming
from eeo.core.exceptions import ValidationError
from eeo.core.streaming import (
    stream_windows,
    valid_mask,
    valid_mean_std,
    valid_min_max,
    valid_percentiles,
)

UTM_CRS = 32633

#: Block shapes to run every differential check at. ``None`` means the
#: library's own choice, which for these small rasters is a single block — the
#: case where streaming and reading whole are trivially the same, and so the
#: one that proves the least.
BLOCK_SHAPES = [None, (2, 2), (1, 6), (7, 3), (5, 1)]

PERCENTILES = [0, 1, 2, 25, 50, 75, 98, 99, 100, 33.3]


def _grid(array, **kwargs):
    return eeo.load_array(
        array, transform=from_origin(500_000.0, 4_200_000.0, 10.0, 10.0), crs=UTM_CRS, **kwargs
    ).to_rasterio()


@pytest.fixture
def blocks(request, monkeypatch):
    """Force a block shape everywhere, or leave the library's own in place."""
    shape = request.param
    if shape is not None:
        for module in (blockwise, streaming):
            monkeypatch.setattr(
                module, "resolve_block_shape", lambda _shape, _b=shape, **kwargs: _b
            )
    return shape


def _reference(array, nodata):
    """The array as the whole-array form sees it: float64, invalid as NaN."""
    reference = array.astype(np.float64)
    if nodata is not None:
        reference = np.where(reference == nodata, np.nan, reference)
    return reference


#: ``(label, array, nodata)`` covering the dtypes and nodata arrangements the
#: reductions have to handle: unsigned and signed integers, floats, a sentinel
#: that is present and one that is declared but never occurs, stray NaNs that
#: no sentinel produced, and heavy ties.
def _cases():
    rng = np.random.default_rng(20260912)
    floats_with_nan = rng.normal(size=(13, 11)).astype(np.float32)
    floats_with_nan[rng.random((13, 11)) < 0.15] = np.nan
    return [
        ("uint16 no nodata", rng.integers(0, 9000, (13, 11)).astype(np.uint16), None),
        ("uint16 nodata present", rng.integers(0, 9000, (13, 11)).astype(np.uint16), 0),
        ("uint16 nodata absent", rng.integers(1, 9000, (13, 11)).astype(np.uint16), 0),
        ("int16 negatives", rng.integers(-400, 400, (13, 11)).astype(np.int16), -32768),
        ("uint8 heavy ties", rng.integers(0, 3, (13, 11)).astype(np.uint8), None),
        ("float32", rng.normal(size=(13, 11)).astype(np.float32), None),
        ("float32 stray NaN", floats_with_nan, None),
        ("float32 sentinel", np.round(rng.normal(size=(13, 11)), 2).astype(np.float32), 0.0),
    ]


CASES = [pytest.param(array, nodata, id=label) for label, array, nodata in _cases()]


class TestValidMask:
    def test_a_declared_sentinel_is_invalid(self):
        block = np.array([[1, 0, 3]], dtype=np.uint16)
        assert valid_mask(block, 0).tolist() == [[True, False, True]]

    def test_no_declared_nodata_leaves_every_integer_pixel_valid(self):
        block = np.array([[1, 0, 3]], dtype=np.uint16)
        assert valid_mask(block, None).all()

    def test_a_stray_nan_is_invalid_even_with_no_sentinel(self):
        # numpy's nan-reductions skip every NaN, not only the ones a sentinel
        # produced, so a streamed statistic has to skip them too or it will
        # disagree with the answer this library used to give.
        block = np.array([[1.0, np.nan, 3.0]], dtype=np.float32)
        assert valid_mask(block, None).tolist() == [[True, False, True]]

    def test_a_nan_sentinel_is_matched_by_nan_not_by_equality(self):
        block = np.array([[1.0, np.nan]], dtype=np.float32)
        assert valid_mask(block, float("nan")).tolist() == [[True, False]]


@pytest.mark.parametrize("blocks", BLOCK_SHAPES, indirect=True)
class TestReductionsMatchNumpy:
    @pytest.mark.parametrize(("array", "nodata"), CASES)
    def test_min_and_max_are_exact(self, array, nodata, blocks):
        ds = _grid(array, nodata=nodata)
        try:
            reference = _reference(array, nodata)
            assert valid_min_max(ds) == (
                float(np.nanmin(reference)),
                float(np.nanmax(reference)),
            )
        finally:
            ds.close()

    @pytest.mark.parametrize(("array", "nodata"), CASES)
    def test_mean_and_std_match_to_rounding(self, array, nodata, blocks):
        ds = _grid(array, nodata=nodata)
        try:
            reference = _reference(array, nodata)
            mean, std = valid_mean_std(ds)
            assert mean == pytest.approx(float(np.nanmean(reference)), rel=1e-12)
            assert std == pytest.approx(float(np.nanstd(reference)), rel=1e-12)
        finally:
            ds.close()

    @pytest.mark.parametrize(("array", "nodata"), CASES)
    def test_percentiles_are_exact(self, array, nodata, blocks):
        # Exact, not approximate: integer rasters are counted one value per
        # bin, so the order statistics come back whole, and the interpolation
        # between them is numpy's own.
        ds = _grid(array, nodata=nodata)
        try:
            reference = _reference(array, nodata)
            expected = np.nanpercentile(reference, PERCENTILES)
            assert valid_percentiles(ds, PERCENTILES) == pytest.approx(list(expected), rel=0, abs=0)
        finally:
            ds.close()

    def test_a_windowed_stream_covers_every_pixel_once(self, blocks):
        array = np.arange(143, dtype=np.uint16).reshape(13, 11)
        ds = _grid(array)
        try:
            rebuilt = np.zeros((13, 11), dtype=np.uint16)
            covered = np.zeros((13, 11), dtype=int)
            for window, block, _valid in stream_windows(ds, 1):
                rows = slice(window.row_off, window.row_off + window.height)
                cols = slice(window.col_off, window.col_off + window.width)
                rebuilt[rows, cols] = block
                covered[rows, cols] += 1
            assert np.array_equal(rebuilt, array)
            assert (covered == 1).all()
        finally:
            ds.close()


class TestEmptyAndDegenerate:
    def test_reductions_over_no_valid_pixel_are_nan(self):
        ds = _grid(np.full((6, 6), -9999.0, dtype=np.float32), nodata=-9999.0)
        try:
            assert all(np.isnan(v) for v in valid_min_max(ds))
            assert all(np.isnan(v) for v in valid_mean_std(ds))
            assert all(np.isnan(v) for v in valid_percentiles(ds, (2, 98)))
        finally:
            ds.close()

    def test_a_constant_band_has_zero_deviation(self):
        ds = _grid(np.full((6, 6), 7.0, dtype=np.float32))
        try:
            assert valid_mean_std(ds) == (7.0, 0.0)
            assert valid_min_max(ds) == (7.0, 7.0)
        finally:
            ds.close()

    def test_an_all_nodata_integer_raster_has_no_percentiles(self):
        # The integer path asks for the range first; there being none is how it
        # learns not to build a histogram at all.
        ds = _grid(np.zeros((6, 6), dtype=np.uint16), nodata=0)
        try:
            assert all(np.isnan(v) for v in valid_percentiles(ds, (2, 50, 98)))
        finally:
            ds.close()

    def test_whole_blocks_of_nodata_do_not_disturb_the_histogram(self, monkeypatch):
        # With 2x2 blocks the nodata half of this raster gives blocks holding
        # no valid pixel at all, which must be skipped rather than counted.
        for module in (blockwise, streaming):
            monkeypatch.setattr(module, "resolve_block_shape", lambda _shape, **kw: (2, 2))
        array = np.arange(36, dtype=np.uint16).reshape(6, 6)
        array[:3, :] = 0
        ds = _grid(array, nodata=0)
        try:
            valid = array[array != 0]
            assert valid_percentiles(ds, PERCENTILES) == pytest.approx(
                list(np.percentile(valid, PERCENTILES)), rel=0, abs=0
            )
        finally:
            ds.close()

    def test_an_integer_range_too_wide_to_count_still_gives_exact_percentiles(self, monkeypatch):
        # Past MAX_HISTOGRAM_BINS the histogram is abandoned for reading the
        # band. The answer must not change, only the memory it took to get it.
        monkeypatch.setattr(streaming, "MAX_HISTOGRAM_BINS", 4)
        array = np.arange(0, 3600, 100, dtype=np.int32).reshape(6, 6)
        ds = _grid(array)
        try:
            assert valid_percentiles(ds, PERCENTILES) == pytest.approx(
                list(np.percentile(array, PERCENTILES)), rel=0, abs=0
            )
        finally:
            ds.close()


def _eager_stat(array, nodata, kind, percentile=None):
    """The value and position the whole-array implementation would report."""
    band = _reference(array, nodata)
    if kind == "max":
        value, flat = np.nanmax(band), np.nanargmax(band)
    elif kind == "min":
        value, flat = np.nanmin(band), np.nanargmin(band)
    elif kind == "mean":
        value = np.nanmean(band)
        flat = np.nanargmin(np.abs(band - value))
    else:
        value = np.nanpercentile(band, percentile)
        flat = np.nanargmin(np.abs(band - value))
    row, col = np.unravel_index(flat, band.shape)
    return float(value), (int(row), int(col))


@pytest.mark.parametrize("blocks", BLOCK_SHAPES, indirect=True)
class TestPixelStatsMatchNumpy:
    """Value *and* position, against the whole-array answer, at every seam."""

    @pytest.mark.parametrize(("array", "nodata"), CASES)
    @pytest.mark.parametrize("kind", ["max", "min", "mean"])
    def test_the_value_and_the_pixel_both_match(self, array, nodata, kind, blocks):
        ds = _grid(array, nodata=nodata)
        try:
            call = {
                "max": ds.get_maximum_pixel,
                "min": ds.get_minimum_pixel,
                "mean": ds.get_mean_pixel,
            }[kind]
            result = call(return_position_as_pixel_coordinate=True)
            value, position = _eager_stat(array, nodata, kind)
            assert result["value"] == pytest.approx(value, rel=1e-12)
            assert result["position"] == position
        finally:
            ds.close()

    @pytest.mark.parametrize(("array", "nodata"), CASES)
    def test_the_percentile_pixel_matches(self, array, nodata, blocks):
        ds = _grid(array, nodata=nodata)
        try:
            result = ds.get_percentile_pixel(50, return_position_as_pixel_coordinate=True)
            value, position = _eager_stat(array, nodata, "percentile", percentile=50)
            assert result["value"] == pytest.approx(value, rel=1e-12)
            assert result["position"] == position
        finally:
            ds.close()

    def test_a_tie_is_broken_by_position_not_by_block(self, blocks):
        # Every pixel is the maximum, so the answer is decided entirely by the
        # tie-break. numpy picks the first in row-major order; so must we,
        # whichever block happens to be read first.
        ds = _grid(np.full((13, 11), 5, dtype=np.uint16))
        try:
            assert ds.get_maximum_pixel(return_position_as_pixel_coordinate=True)["position"] == (
                0,
                0,
            )
            assert ds.get_minimum_pixel(return_position_as_pixel_coordinate=True)["position"] == (
                0,
                0,
            )
        finally:
            ds.close()

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
    def test_the_maximum_of_an_unsigned_band_containing_zero(self, dtype, blocks):
        # Regression: the maximum is found by negating the block and taking a
        # minimum. Negating an unsigned block wraps rather than changing sign,
        # and 0 wraps to 0 — the smallest possible score — so the op returned
        # the *minimum* pixel for any unsigned band containing a zero. The
        # existing fixtures all started at 1000, so nothing caught it.
        array = np.array([[0, 5, 3], [9, 0, 1]], dtype=dtype)
        ds = _grid(array)
        try:
            peak = ds.get_maximum_pixel(return_position_as_pixel_coordinate=True)
            assert peak["value"] == 9
            assert peak["position"] == (1, 0)
            floor = ds.get_minimum_pixel(return_position_as_pixel_coordinate=True)
            assert floor["value"] == 0
            assert floor["position"] == (0, 0)
        finally:
            ds.close()

    def test_world_coordinates_are_the_pixel_position_through_the_transform(self, blocks):
        array = np.arange(143, dtype=np.uint16).reshape(13, 11)
        ds = _grid(array)
        try:
            peak = ds.get_maximum_pixel()
            row, col = _eager_stat(array, None, "max")[1]
            assert peak["position"] == ds.get_transform() * (col, row)
        finally:
            ds.close()


class TestPixelStatsRefusals:
    @pytest.mark.parametrize(
        "call",
        [
            lambda ds: ds.get_maximum_pixel(),
            lambda ds: ds.get_minimum_pixel(),
            lambda ds: ds.get_mean_pixel(),
            lambda ds: ds.get_percentile_pixel(50),
        ],
    )
    def test_a_band_with_no_valid_pixel_is_refused_clearly(self, call):
        # The whole-array form raised "All-NaN slice encountered" from inside
        # numpy here. Saying which band and what was undefined is more use.
        ds = _grid(np.full((6, 6), -9999.0, dtype=np.float32), nodata=-9999.0)
        try:
            with pytest.raises(ValidationError, match="no valid pixels"):
                call(ds)
        finally:
            ds.close()


class TestPointSamplingReadsOnePixel:
    def test_sampling_does_not_read_the_band(self, monkeypatch):
        # The op used to read the whole band to index one pixel out of it,
        # which on a 10 m Sentinel-2 band is 241 MB for one number.
        reads = []
        original = eeo.core.core.EEORasterDataset.read

        def spy(self, *args, **kwargs):
            array = original(self, *args, **kwargs)
            reads.append(np.shape(array))
            return array

        # Built before the spy goes on: promoting a NumPy-backed array to
        # rasterio reads it once, and that read is not the one under test.
        ds = _grid(np.arange(143, dtype=np.uint16).reshape(13, 11))
        monkeypatch.setattr(eeo.core.core.EEORasterDataset, "read", spy)
        try:
            value = ds.extract_value_at_coordinate((500_005.0, 4_199_995.0))
            assert value == 0
            assert reads == [(1, 1)], f"expected one 1x1 read, got {reads}"
        finally:
            ds.close()

    def test_the_sampled_value_is_still_the_right_pixel(self):
        array = np.arange(143, dtype=np.uint16).reshape(13, 11)
        ds = _grid(array)
        try:
            # Centre of pixel (3, 4) on a 10 m grid from (500000, 4200000).
            x = 500_000.0 + 4 * 10.0 + 5.0
            y = 4_200_000.0 - 3 * 10.0 - 5.0
            assert ds.extract_value_at_coordinate((x, y)) == array[3, 4]
        finally:
            ds.close()


@pytest.mark.parametrize("blocks", BLOCK_SHAPES, indirect=True)
class TestNormalizationMatchesNumpy:
    @pytest.mark.parametrize(("array", "nodata"), CASES)
    def test_standardize_matches_the_whole_array_form(self, array, nodata, blocks):
        ds = _grid(array, nodata=nodata)
        try:
            reference = _reference(array, nodata)
            with np.errstate(divide="ignore", invalid="ignore"):
                expected = ((reference - np.nanmean(reference)) / np.nanstd(reference)).astype(
                    np.float32
                )
            assert ds.standardize().read()[0] == pytest.approx(expected, nan_ok=True, rel=1e-5)
        finally:
            ds.close()

    @pytest.mark.parametrize(("array", "nodata"), CASES)
    def test_normalize_percentile_matches_the_whole_array_form(self, array, nodata, blocks):
        ds = _grid(array, nodata=nodata)
        try:
            reference = _reference(array, nodata)
            low, high = np.nanpercentile(reference, (2, 98))
            with np.errstate(divide="ignore", invalid="ignore"):
                expected = np.clip((reference - low) / (high - low), 0, 1).astype(np.float32)
            assert ds.normalize_percentile().read()[0] == pytest.approx(
                expected, nan_ok=True, rel=1e-6
            )
        finally:
            ds.close()

    @pytest.mark.parametrize(("array", "nodata"), CASES)
    def test_normalize_min_max_matches_the_whole_array_form(self, array, nodata, blocks):
        ds = _grid(array, nodata=nodata)
        try:
            reference = _reference(array, nodata)
            low, high = np.nanmin(reference), np.nanmax(reference)
            with np.errstate(divide="ignore", invalid="ignore"):
                expected = ((reference - low) / (high - low)).astype(np.float32)
            assert ds.normalize_min_max().read()[0] == pytest.approx(
                expected, nan_ok=True, rel=1e-6
            )
        finally:
            ds.close()


class TestNormalizePercentileValidation:
    """The ``Raises`` section used to promise an error NumPy never raised."""

    @pytest.mark.parametrize(
        ("lower", "upper"),
        [(-1, 98), (2, 101), (0, 100.5)],
    )
    def test_a_percentile_outside_0_to_100_is_refused(self, lower, upper):
        ds = _grid(np.arange(36, dtype=np.float32).reshape(6, 6))
        try:
            with pytest.raises(ValidationError, match=r"range \[0, 100\]"):
                ds.normalize_percentile(lower_percentile=lower, upper_percentile=upper)
        finally:
            ds.close()

    @pytest.mark.parametrize(("lower", "upper"), [(98, 2), (50, 50)])
    def test_an_inverted_or_empty_range_is_refused(self, lower, upper):
        # numpy accepted these and returned the thresholds in the given order,
        # so the stretch silently came out inverted or divided by zero.
        ds = _grid(np.arange(36, dtype=np.float32).reshape(6, 6))
        try:
            with pytest.raises(ValidationError, match="must be below"):
                ds.normalize_percentile(lower_percentile=lower, upper_percentile=upper)
        finally:
            ds.close()


class TestPointSamplingNodata:
    def test_a_declared_sentinel_is_reported_as_nan(self):
        ds = _grid(np.full((6, 6), -9999.0, dtype=np.float32), nodata=-9999.0)
        try:
            assert np.isnan(ds.extract_value_at_coordinate((500_005.0, 4_199_995.0)))
        finally:
            ds.close()

    def test_a_stray_nan_is_reported_as_nan_even_with_no_sentinel(self):
        # A float band can hold NaN without declaring any nodata at all; that
        # pixel is still not a measurement.
        array = np.arange(36, dtype=np.float32).reshape(6, 6)
        array[0, 0] = np.nan
        ds = _grid(array)
        try:
            assert np.isnan(ds.extract_value_at_coordinate((500_005.0, 4_199_995.0)))
        finally:
            ds.close()
