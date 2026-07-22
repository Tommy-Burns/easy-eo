"""Value-correctness and contract tests for the spectral index library."""

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from eeo import load_array
from eeo.core.core import EEORasterDataset
from eeo.core.exceptions import AlignmentError, ValidationError

UTM_CRS = CRS.from_epsg(32633)

# A north-up 2x2 grid shared by every band raster below.
_TRANSFORM = Affine.translation(500_000.0, 4_200_000.0) * Affine.scale(10.0, -10.0)

# Deterministic band values chosen so every index is hand-computable and no
# denominator is zero.
NIR = np.array([[0.4, 0.6], [0.8, 0.5]], dtype=np.float32)
RED = np.array([[0.2, 0.1], [0.3, 0.25]], dtype=np.float32)
BLUE = np.array([[0.1, 0.05], [0.2, 0.15]], dtype=np.float32)
GREEN = np.array([[0.3, 0.35], [0.25, 0.4]], dtype=np.float32)
SWIR = np.array([[0.15, 0.2], [0.35, 0.3]], dtype=np.float32)


def _band(array, nodata=None):
    """Build a rasterio-backed single-band raster on the shared grid."""
    return load_array(array, transform=_TRANSFORM, crs=UTM_CRS, nodata=nodata).to_rasterio()


def _nd(a, b):
    """Reference normalized difference ``(a - b) / (a + b)``."""
    return (a - b) / (a + b)


# ---------------------------------------------------------------------------
# Value correctness (hand-computed reference arrays)
# ---------------------------------------------------------------------------
def test_ndvi_matches_reference():
    result = _band(NIR).ndvi(_band(RED), return_as_ndarray=True)
    assert np.allclose(result, _nd(NIR, RED))


def test_ndwi_matches_reference():
    # McFeeters water NDWI: (Green - NIR) / (Green + NIR), called on Green.
    result = _band(GREEN).ndwi(_band(NIR), return_as_ndarray=True)
    assert np.allclose(result, _nd(GREEN, NIR))


def test_ndmi_matches_reference():
    result = _band(NIR).ndmi(_band(SWIR), return_as_ndarray=True)
    assert np.allclose(result, _nd(NIR, SWIR))


def test_ndbi_matches_reference():
    # Built-up NDBI: (SWIR1 - NIR) / (SWIR1 + NIR), called on SWIR1.
    result = _band(SWIR).ndbi(_band(NIR), return_as_ndarray=True)
    assert np.allclose(result, _nd(SWIR, NIR))


def test_evi_matches_reference():
    result = _band(NIR).evi(_band(RED), _band(BLUE), return_as_ndarray=True)
    expected = 2.5 * (NIR - RED) / (NIR + 6.0 * RED - 7.5 * BLUE + 1.0)
    assert np.allclose(result, expected)


def test_savi_matches_reference():
    result = _band(NIR).savi(_band(RED), soil_factor=0.5, return_as_ndarray=True)
    expected = (1.0 + 0.5) * (NIR - RED) / (NIR + RED + 0.5)
    assert np.allclose(result, expected)


def test_savi_with_zero_soil_factor_equals_ndvi():
    savi = _band(NIR).savi(_band(RED), soil_factor=0.0, return_as_ndarray=True)
    ndvi = _band(NIR).ndvi(_band(RED), return_as_ndarray=True)
    assert np.allclose(savi, ndvi)


# ---------------------------------------------------------------------------
# Band-index selection vs separate datasets are equivalent
# ---------------------------------------------------------------------------
def test_band_index_selection_matches_separate_datasets():
    scene = _band(np.stack([NIR, RED]))  # 2-band stack: band 1 NIR, band 2 Red
    from_indices = scene.ndvi(red=2, nir=1, return_as_ndarray=True)
    from_datasets = _band(NIR).ndvi(_band(RED), return_as_ndarray=True)
    assert np.allclose(from_indices, from_datasets)


def test_evi_band_index_selection():
    scene = _band(np.stack([NIR, RED, BLUE]))
    result = scene.evi(red=2, blue=3, nir=1, return_as_ndarray=True)
    expected = 2.5 * (NIR - RED) / (NIR + 6.0 * RED - 7.5 * BLUE + 1.0)
    assert np.allclose(result, expected)


# ---------------------------------------------------------------------------
# Output type, dtype, and chainability
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "call",
    [
        lambda: _band(NIR).ndvi(_band(RED)),
        lambda: _band(GREEN).ndwi(_band(NIR)),
        lambda: _band(NIR).ndmi(_band(SWIR)),
        lambda: _band(SWIR).ndbi(_band(NIR)),
        lambda: _band(NIR).evi(_band(RED), _band(BLUE)),
        lambda: _band(NIR).savi(_band(RED)),
    ],
)
def test_index_returns_single_band_float32_dataset(call):
    result = call()
    assert isinstance(result, EEORasterDataset)
    meta = result.get_metadata()
    assert meta["dtype"] == "float32"
    assert result.get_count() == 1
    # Chainable: the result is itself a dataset an op can consume.
    doubled = result.multiply(2)
    assert isinstance(doubled, EEORasterDataset)


def test_index_result_is_chainable_end_to_end():
    result = _band(NIR).ndvi(_band(RED)).multiply(100).add(1)
    assert isinstance(result, EEORasterDataset)


# ---------------------------------------------------------------------------
# Nodata propagation
# ---------------------------------------------------------------------------
def test_nodata_is_contagious_and_output_nodata_is_nan():
    nir = NIR.copy()
    nir[0, 0] = -9999.0
    result = _band(nir, nodata=-9999.0).ndvi(_band(RED))
    meta = result.get_metadata()
    assert meta["nodata"] is not None and np.isnan(meta["nodata"])

    data = result.read()[0]
    assert np.isnan(data[0, 0])  # masked pixel
    assert not np.isnan(data[1, 1])  # valid pixel survives


def test_nodata_in_second_band_propagates():
    red = RED.copy()
    red[1, 1] = -1.0
    result = _band(NIR).ndvi(_band(red, nodata=-1.0), return_as_ndarray=True)
    assert np.isnan(result[1, 1])
    assert not np.isnan(result[0, 0])


def test_no_declared_nodata_yields_none_nodata():
    result = _band(NIR).ndvi(_band(RED))
    assert result.get_metadata()["nodata"] is None


# ---------------------------------------------------------------------------
# Numerical safety: zero denominator guarded to 0
# ---------------------------------------------------------------------------
def test_zero_denominator_maps_to_zero():
    zeros = np.zeros((2, 2), dtype=np.float32)
    result = _band(zeros).ndvi(_band(zeros), return_as_ndarray=True)
    assert np.all(result == 0)
    assert not np.any(np.isnan(result))


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
def test_mismatched_grid_raises_without_auto_align():
    coarse = load_array(
        np.ones((1, 1), dtype=np.float32),
        transform=Affine.translation(500_000.0, 4_200_000.0) * Affine.scale(20.0, -20.0),
        crs=UTM_CRS,
    ).to_rasterio()
    with pytest.raises(AlignmentError):
        _band(NIR).ndvi(coarse, auto_align=False)


def test_mismatched_grid_auto_aligns_by_default():
    coarse = load_array(
        np.full((1, 1), 0.2, dtype=np.float32),
        transform=Affine.translation(500_000.0, 4_200_000.0) * Affine.scale(20.0, -20.0),
        crs=UTM_CRS,
    ).to_rasterio()
    result = _band(NIR).ndvi(coarse)  # auto_align=True by default
    assert result.get_shape() == (2, 2)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_invalid_band_spec_type_raises_validation_error():
    with pytest.raises(ValidationError):
        _band(NIR).ndvi(2.5)  # float is neither a dataset nor an int index


def test_out_of_range_band_index_raises_index_error():
    with pytest.raises(IndexError):
        _band(NIR).ndvi(red=5, nir=1)  # single-band raster has no band 5


# ---------------------------------------------------------------------------
# Provenance preservation (timestamp / attrs survive the operation)
# ---------------------------------------------------------------------------
def test_index_preserves_provenance():
    from datetime import datetime

    nir = load_array(
        NIR,
        transform=_TRANSFORM,
        crs=UTM_CRS,
        timestamp=datetime(2024, 6, 1),
        attrs={"tile": "T33"},
    ).to_rasterio()
    result = nir.ndvi(_band(RED))
    assert result.timestamp == datetime(2024, 6, 1)
    assert result.attrs == {"tile": "T33"}
