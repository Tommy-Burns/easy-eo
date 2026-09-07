"""Tests for the mask_clouds operation (eeo/preprocessing/masking.py).

The decoders are tested elsewhere; what matters here is everything around
them — that the right decoder is chosen from the band's name, that the mask
reaches every band, that the nodata contract is honoured, and that the cases
where the operation cannot know what the user meant are refused rather than
guessed at.
"""

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from eeo import load_array
from eeo.core.exceptions import AlignmentError, ValidationError
from eeo.preprocessing.quality import QAConfidence, SCLClass

UTM = CRS.from_epsg(32633)
TRANSFORM = Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_200_000.0)

# One pixel of each: clear vegetation, high-probability cloud, cloud shadow,
# water, thin cirrus, and no-data.
SCL_ROW = [
    SCLClass.VEGETATION,
    SCLClass.CLOUD_HIGH_PROBABILITY,
    SCLClass.CLOUD_SHADOWS,
    SCLClass.WATER,
    SCLClass.THIN_CIRRUS,
    SCLClass.NO_DATA,
]
# Which of those the default class set masks.
SCL_MASKED = [False, True, True, False, True, True]

# Landsat: clear-with-lows, high-conf cloud, mid-conf cloud, high-conf shadow,
# high-conf snow, fill. Values are USGS's own (LSDS-1619 Table 6-3).
QA_ROW = [21824, 22280, 22080, 23888, 30048, 1]
QA_MASKED = [False, True, True, True, False, True]


def _dataset(bands, names, *, nodata=None, dtype="uint16"):
    """Build a rasterio-backed dataset from a list of 1-pixel-tall bands.

    Each entry of ``bands`` is one band's row of pixels, so the result is a
    ``(len(bands), 1, width)`` raster — one row keeps the expected values
    short enough to read as a list in each assertion.
    """
    array = np.asarray(bands, dtype=dtype).reshape(len(bands), 1, -1)
    ds = load_array(array, transform=TRANSFORM, crs=UTM, nodata=nodata).to_rasterio()
    ds.band_names = names
    return ds


@pytest.fixture
def sentinel2_like():
    """Two reflectance bands and an SCL band, on one 1x6 grid."""
    red = np.full(6, 1000)
    nir = np.full(6, 3000)
    return _dataset([red, nir, list(SCL_ROW)], ["red", "nir", "scl"])


@pytest.fixture
def landsat_like():
    """Two reflectance bands and a QA_PIXEL band, with a declared nodata."""
    red = np.full(6, 1000)
    nir = np.full(6, 3000)
    ds = _dataset([red, nir, QA_ROW], ["red", "nir08", "qa_pixel"], nodata=0)
    ds.attrs = {"mission": "Landsat 9"}
    return ds


class TestDecoderSelection:
    """Which decoder runs is settled by the band's name, not by an argument."""

    def test_an_scl_band_is_decoded_as_scene_classes(self, sentinel2_like):
        out = sentinel2_like.mask_clouds(nodata=0).read()
        assert (out[0][0] == 0).tolist() == SCL_MASKED

    def test_a_qa_pixel_band_is_decoded_as_bits(self, landsat_like):
        out = landsat_like.mask_clouds().read()
        assert (out[0][0] == 0).tolist() == QA_MASKED

    def test_mid_confidence_cloud_is_masked_through_the_op(self, landsat_like):
        # The threshold has to survive the trip through the operation, not
        # only work when qa_pixel_mask is called directly.
        assert landsat_like.mask_clouds().read()[0][0][2] == 0
        kept = landsat_like.mask_clouds(min_cloud_confidence=None).read()[0]
        assert kept[0][2] == 1000

    def test_the_band_can_be_named_explicitly(self, sentinel2_like):
        by_name = sentinel2_like.mask_clouds(mask_band="scl", nodata=0).read()
        by_index = sentinel2_like.mask_clouds(mask_band=3, nodata=0).read()
        assert np.array_equal(by_name, by_index)

    def test_naming_a_band_that_is_not_a_quality_layer_is_refused(self, sentinel2_like):
        with pytest.raises(ValidationError, match="not a quality band"):
            sentinel2_like.mask_clouds(mask_band="red", nodata=0)

    def test_a_dataset_without_a_quality_band_is_refused(self):
        ds = _dataset([np.full(6, 1000)], ["red"])
        with pytest.raises(ValidationError, match="no quality band"):
            ds.mask_clouds(nodata=0)

    def test_two_quality_bands_are_refused_rather_than_ordered(self):
        # Choosing by band order would make the mask depend on load order.
        ds = _dataset([list(SCL_ROW), QA_ROW], ["scl", "qa_pixel"])
        with pytest.raises(ValidationError, match="more than one quality band"):
            ds.mask_clouds(nodata=0)


class TestSensorSpecificArguments:
    def test_classes_on_a_landsat_band_is_refused(self, landsat_like):
        with pytest.raises(ValidationError, match="use flags="):
            landsat_like.mask_clouds(classes=[3])

    def test_flags_on_a_sentinel2_band_is_refused(self, sentinel2_like):
        with pytest.raises(ValidationError, match="use classes="):
            sentinel2_like.mask_clouds(flags=["cloud"], nodata=0)

    def test_classes_selects_which_scene_classes_to_mask(self, sentinel2_like):
        out = sentinel2_like.mask_clouds(classes=["water"], nodata=0).read()
        assert (out[0][0] == 0).tolist() == [False, False, False, True, False, False]

    def test_flags_selects_which_bits_to_mask(self, landsat_like):
        out = landsat_like.mask_clouds(flags=["snow"], min_cloud_confidence=None).read()
        assert (out[0][0] == 0).tolist() == [False, False, False, False, True, False]

    def test_a_high_threshold_keeps_mid_confidence_cloud(self, landsat_like):
        out = landsat_like.mask_clouds(min_cloud_confidence=QAConfidence.HIGH).read()
        assert out[0][0][2] == 1000


class TestMission:
    def test_the_mission_is_read_from_attrs(self, landsat_like):
        assert landsat_like.attrs["mission"] == "Landsat 9"
        landsat_like.mask_clouds()  # does not raise

    def test_an_explicit_mission_overrides_attrs(self, landsat_like):
        # Landsat 7 has no cirrus bit; the default drops it rather than raising.
        landsat_like.mask_clouds(mission=7)

    def test_a_missing_mission_is_refused_rather_than_assumed(self):
        ds = _dataset([np.full(6, 1000), QA_ROW], ["red", "qa_pixel"], nodata=0)
        ds.attrs = {}
        with pytest.raises(ValidationError, match="which Landsat took the scene"):
            ds.mask_clouds()

    def test_naming_cirrus_on_a_sensor_without_it_is_refused(self, landsat_like):
        with pytest.raises(ValidationError, match="Unused"):
            landsat_like.mask_clouds(mission=7, flags=["cirrus"])

    def test_sentinel2_needs_no_mission(self, sentinel2_like):
        sentinel2_like.attrs = {}
        sentinel2_like.mask_clouds(nodata=0)


class TestNodataContract:
    def test_the_declared_nodata_is_used_and_recorded(self, landsat_like):
        out = landsat_like.mask_clouds()
        assert out.get_metadata()["nodata"] == 0

    def test_an_explicit_nodata_overrides_the_declared_one(self, landsat_like):
        out = landsat_like.mask_clouds(nodata=9999)
        assert out.get_metadata()["nodata"] == 9999
        assert out.read()[0][0][1] == 9999

    def test_a_float_raster_without_nodata_gets_nan(self):
        ds = _dataset([np.full(6, 0.5), list(SCL_ROW)], ["red", "scl"], dtype="float32")
        out = ds.mask_clouds()
        assert np.isnan(out.get_metadata()["nodata"])
        assert np.isnan(out.read()[0][0]).tolist() == SCL_MASKED

    def test_an_integer_raster_without_nodata_is_refused(self, sentinel2_like):
        # There is no value a masked pixel could hold, and an integer array
        # cannot hold NaN; guessing a sentinel could delete real data.
        with pytest.raises(ValidationError, match="declares no nodata value"):
            sentinel2_like.mask_clouds()

    def test_a_fractional_nodata_on_an_integer_raster_is_refused(self, sentinel2_like):
        with pytest.raises(ValidationError, match="not a whole number"):
            sentinel2_like.mask_clouds(nodata=0.5)

    def test_a_nodata_outside_the_dtype_is_refused(self, sentinel2_like):
        with pytest.raises(ValidationError, match="does not fit"):
            sentinel2_like.mask_clouds(nodata=-1)

    def test_the_dtype_is_unchanged(self, landsat_like):
        assert landsat_like.mask_clouds().get_metadata()["dtype"] == "uint16"


class TestResultShape:
    def test_every_band_gets_the_same_mask(self, sentinel2_like):
        out = sentinel2_like.mask_clouds(nodata=0).read()
        assert ((out[0] == 0) == (out[1][0] == 0)).all()

    def test_the_quality_band_is_masked_too(self, sentinel2_like):
        # Documented behaviour: the result cannot be used as its own mask a
        # second time.
        out = sentinel2_like.mask_clouds(nodata=0).read()
        assert (out[2][0] == 0).tolist() == SCL_MASKED

    def test_shape_band_count_and_names_are_preserved(self, sentinel2_like):
        out = sentinel2_like.mask_clouds(nodata=0)
        assert out.get_shape() == sentinel2_like.get_shape()
        assert out.get_count() == sentinel2_like.get_count()
        assert out.band_names == ["red", "nir", "scl"]

    def test_georeferencing_and_attrs_are_preserved(self, landsat_like):
        out = landsat_like.mask_clouds()
        assert out.get_transform() == landsat_like.get_transform()
        assert out.get_crs() == landsat_like.get_crs()
        assert out.attrs["mission"] == "Landsat 9"

    def test_the_input_is_not_modified(self, sentinel2_like):
        before = sentinel2_like.read().copy()
        sentinel2_like.mask_clouds(nodata=0)
        assert np.array_equal(sentinel2_like.read(), before)

    def test_the_result_chains(self, sentinel2_like):
        # The point of a chainable op: the masked result feeds an index, and
        # the masked pixels are excluded from it.
        ndvi = sentinel2_like.mask_clouds(nodata=0).ndvi("red", nir="nir")
        assert np.isnan(ndvi.read()[0][0]).tolist() == SCL_MASKED


class TestSeparateMaskDataset:
    def test_a_separate_mask_is_applied(self):
        data = _dataset([np.full(6, 1000), np.full(6, 3000)], ["red", "nir"])
        mask = _dataset([list(SCL_ROW)], ["scl"])
        out = data.mask_clouds(mask=mask, nodata=0).read()
        assert (out[0][0] == 0).tolist() == SCL_MASKED
        assert out.shape == (2, 1, 6)

    def test_a_misaligned_mask_is_refused(self):
        data = _dataset([np.full(6, 1000)], ["red"])
        wrong = load_array(
            np.asarray([list(SCL_ROW) + [0, 0]], dtype="uint16").reshape(1, 1, 8),
            transform=TRANSFORM,
            crs=UTM,
        ).to_rasterio()
        wrong.band_names = ["scl"]
        with pytest.raises(AlignmentError, match="same pixel grid"):
            data.mask_clouds(mask=wrong, nodata=0)

    def test_an_unnamed_single_band_mask_must_state_its_encoding(self):
        data = _dataset([np.full(6, 1000)], ["red"])
        mask = load_array(
            np.asarray(list(SCL_ROW), dtype="uint16").reshape(1, 1, 6),
            transform=TRANSFORM,
            crs=UTM,
        ).to_rasterio()
        with pytest.raises(ValidationError, match="mask_band='scl'"):
            data.mask_clouds(mask=mask, nodata=0)

    def test_an_unnamed_mask_band_can_be_named_by_the_caller(self):
        data = _dataset([np.full(6, 1000)], ["red"])
        mask = load_array(
            np.asarray(list(SCL_ROW), dtype="uint16").reshape(1, 1, 6),
            transform=TRANSFORM,
            crs=UTM,
        ).to_rasterio()
        out = data.mask_clouds(mask=mask, mask_band="scl", nodata=0).read()
        assert (out[0][0] == 0).tolist() == SCL_MASKED


def test_the_op_is_bound_as_a_method():
    ds = _dataset([list(SCL_ROW)], ["scl"])
    assert callable(ds.mask_clouds)


def test_masking_nothing_leaves_every_pixel(sentinel2_like):
    out = sentinel2_like.mask_clouds(classes=["snow_ice"], nodata=0).read()
    assert not (out[0][0] == 0).any()


class TestClearFraction:
    """How much of a raster still holds a measurement."""

    def test_counts_the_pixels_that_are_not_nodata(self):
        # Two of six pixels are nodata.
        ds = _dataset([[0, 0, 1, 2, 3, 4]], ["red"], nodata=0)
        assert ds.clear_fraction() == pytest.approx(4 / 6)

    def test_a_fully_clear_raster_is_one(self):
        ds = _dataset([[1, 2, 3, 4, 5, 6]], ["red"], nodata=0)
        assert ds.clear_fraction() == 1.0

    def test_a_fully_masked_raster_is_zero(self):
        ds = _dataset([[0, 0, 0, 0, 0, 0]], ["red"], nodata=0)
        assert ds.clear_fraction() == 0.0

    def test_a_raster_declaring_no_nodata_is_one(self):
        # Nothing is marked absent, so every pixel is a measurement — the
        # same reading the rest of the library takes.
        ds = _dataset([[0, 0, 1, 2, 3, 4]], ["red"])
        assert ds.clear_fraction() == 1.0

    def test_nan_nodata_is_recognised(self):
        ds = _dataset([[np.nan, 1.0, 2.0, 3.0, 4.0, 5.0]], ["red"], nodata=np.nan, dtype="float32")
        assert ds.clear_fraction() == pytest.approx(5 / 6)

    def test_a_pixel_missing_from_any_band_is_not_clear(self):
        # Nodata is contagious, so the default is the intersection of the
        # bands rather than any one of them.
        ds = _dataset([[0, 1, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1]], ["red", "nir"], nodata=0)
        assert ds.clear_fraction() == pytest.approx(4 / 6)

    def test_one_band_can_be_measured_on_its_own(self):
        ds = _dataset([[0, 1, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1]], ["red", "nir"], nodata=0)
        assert ds.clear_fraction(band="red") == pytest.approx(5 / 6)
        assert ds.clear_fraction(band=2) == pytest.approx(5 / 6)

    def test_an_unknown_band_name_is_refused(self):
        ds = _dataset([[1, 1, 1, 1, 1, 1]], ["red"], nodata=0)
        with pytest.raises(ValidationError, match="no band named"):
            ds.clear_fraction(band="green")

    def test_an_out_of_range_index_is_refused(self):
        ds = _dataset([[1, 1, 1, 1, 1, 1]], ["red"], nodata=0)
        with pytest.raises(IndexError):
            ds.clear_fraction(band=7)

    def test_it_reads_after_masking(self, sentinel2_like):
        # The composition the helper exists for. SCL_MASKED marks four of the
        # six fixture pixels for masking.
        assert sentinel2_like.mask_clouds(nodata=0).clear_fraction() == pytest.approx(
            SCL_MASKED.count(False) / len(SCL_MASKED)
        )

    def test_masking_never_raises_the_clear_fraction(self, landsat_like):
        before = landsat_like.clear_fraction()
        after = landsat_like.mask_clouds().clear_fraction()
        assert after <= before

    def test_the_result_is_a_plain_float(self, sentinel2_like):
        value = sentinel2_like.mask_clouds(nodata=0).clear_fraction()
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0
