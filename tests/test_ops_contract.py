"""Nodata & dtype contract tests for the ``eeo/ops`` algebra operations.

Covers the library-wide contract (see ``CODE_STYLE.md`` "Nodata & Dtype
Contract"): fractional-result ops output float32, exact arithmetic follows
NumPy promotion with floats narrowed to float32, and nodata is masked before
compute and contagious across operands (NaN for float outputs, the integer
sentinel for integer outputs).
"""

import math

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from eeo import load_array

UTM = CRS.from_epsg(32633)
GRID = Affine.translation(500_000.0, 4_200_000.0) * Affine.scale(10.0, -10.0)


def _rio(array, *, nodata=None):
    """Build a rasterio-backed dataset from an array on the shared grid."""
    return load_array(array, transform=GRID, crs=UTM, nodata=nodata).to_rasterio()


def _nodata(ds):
    return ds.get_metadata()["nodata"]


# ---------------------------------------------------------------------
# Dtype policy
# ---------------------------------------------------------------------


def test_integer_arithmetic_stays_integer(multiband_uint16):
    """int raster + int scalar keeps the integer dtype and is exact."""
    result = multiband_uint16.add(1)
    assert result.read().dtype == np.uint16
    # band 1 holds 1000 + gradient(0..35); +1 shifts each value by one.
    np.testing.assert_array_equal(result.read()[0], multiband_uint16.read()[0] + 1)


@pytest.mark.parametrize("op,expected", [("add", 1000.5), ("multiply", 500.0)])
def test_integer_plus_float_scalar_promotes_to_float32(multiband_uint16, op, expected):
    """int raster combined with a float scalar promotes to float32, no truncation."""
    result = getattr(multiband_uint16, op)(0.5)
    assert result.read().dtype == np.float32
    # band 1, pixel (0, 0) == 1000 -> add:1000.5, multiply:500.0
    assert result.read()[0][0, 0] == pytest.approx(expected)


def test_divide_outputs_float32_without_truncating(multiband_uint16):
    """Division is fractional: an odd quotient keeps its .5, not truncated to int."""
    result = multiband_uint16.divide(2)
    assert result.read().dtype == np.float32
    # band 1, pixel (0, 1) == 1001 -> 500.5
    assert result.read()[0][0, 1] == pytest.approx(500.5)


def test_sqrt_and_log_output_float32(multiband_uint16):
    """sqrt and log are fractional-result ops: float32 regardless of input dtype."""
    assert multiband_uint16.sqrt().read().dtype == np.float32
    assert multiband_uint16.log().read().dtype == np.float32


def test_power_fractional_exponent_promotes_to_float32(multiband_uint16):
    """An integer raster raised to a fractional exponent promotes to float32."""
    assert multiband_uint16.power(0.5).read().dtype == np.float32


def test_power_integer_exponent_stays_integer(multiband_uint16):
    """An integer raster raised to an integer exponent keeps its integer dtype."""
    assert multiband_uint16.power(2).read().dtype == np.uint16


def test_float_ops_stay_float32(single_band_float32):
    """Every op on a float32 raster returns float32."""
    for result in (
        single_band_float32.add(1),
        single_band_float32.subtract(1),
        single_band_float32.multiply(2),
        single_band_float32.divide(2),
        single_band_float32.power(2),
        single_band_float32.sqrt(),
        single_band_float32.log(),
        single_band_float32.absolute(),
    ):
        assert result.read().dtype == np.float32


# ---------------------------------------------------------------------
# Nodata policy
# ---------------------------------------------------------------------


def test_no_nodata_declared_produces_no_nodata(single_band_float32):
    """A raster with nodata=None produces output with no nodata."""
    result = single_band_float32.add(1)
    assert _nodata(result) is None
    assert not np.isnan(result.read()).any()


def test_scalar_op_masks_float_nodata_to_nan(raster_with_nodata):
    """A float raster's nodata region becomes NaN and metadata nodata is NaN."""
    result = raster_with_nodata.add(1)
    out = result.read()[0]

    assert np.isnan(out[:2, :2]).all()  # the nodata block
    assert math.isnan(_nodata(result))
    # a valid pixel is unaffected: gradient index (3, 3) == 21 -> +1 == 22
    assert out[3, 3] == pytest.approx(22.0)


def test_nodata_is_contagious_across_operands():
    """A pixel that is nodata in either operand is nodata in the output."""
    a = np.arange(36, dtype=np.float32).reshape(6, 6)
    a[:2, :2] = -9999.0
    b = np.arange(36, dtype=np.float32).reshape(6, 6)
    b[4:, 4:] = -9999.0
    ds_a = _rio(a, nodata=-9999.0)
    ds_b = _rio(b, nodata=-9999.0)

    out = ds_a.add(ds_b).read()[0]

    assert np.isnan(out[:2, :2]).all()  # nodata contributed by a
    assert np.isnan(out[4:, 4:]).all()  # nodata contributed by b
    assert out[3, 3] == pytest.approx(2 * 21.0)  # valid in both operands


def test_integer_nodata_sentinel_is_preserved():
    """An integer raster keeps its integer nodata sentinel, not NaN."""
    array = np.arange(1, 37, dtype=np.uint16).reshape(6, 6)
    array[:2, :2] = 0
    ds = _rio(array, nodata=0)

    result = ds.add(1)
    out = result.read()[0]

    assert result.read().dtype == np.uint16
    assert _nodata(result) == 0
    assert (out[:2, :2] == 0).all()  # sentinel preserved, not incremented
    assert out[3, 3] == array[3, 3] + 1  # valid pixel incremented


def test_divide_masks_nodata_to_nan(raster_with_nodata):
    """Division masks the nodata region to NaN in its float32 output."""
    out = raster_with_nodata.divide(2).read()[0]
    assert np.isnan(out[:2, :2]).all()
    assert out[3, 3] == pytest.approx(21.0 / 2)


def test_sqrt_masks_nodata_instead_of_clamping(raster_with_nodata):
    """The negative nodata sentinel is masked to NaN, not clamped to 0 then rooted."""
    out = raster_with_nodata.sqrt().read()[0]
    assert np.isnan(out[:2, :2]).all()
    assert out[3, 3] == pytest.approx(math.sqrt(21.0))


def test_absolute_masks_nodata_instead_of_taking_magnitude(raster_with_nodata):
    """The negative nodata sentinel is masked to NaN, not turned into its magnitude."""
    out = raster_with_nodata.absolute().read()[0]
    assert np.isnan(out[:2, :2]).all()
    assert not (out[:2, :2] == 9999.0).any()
    assert out[3, 3] == pytest.approx(21.0)
