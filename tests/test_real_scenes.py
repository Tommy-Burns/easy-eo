"""Quality-layer decoding checked against real downloaded products.

The synthetic tests in ``test_quality_masks.py`` and ``test_qa_pixel.py`` pin
the tables to what ESA and USGS publish. These check the same code against
what the agencies actually ship, which is a different question: a table can be
transcribed correctly and still be read against the wrong band, the wrong
dtype, or a grid the loader resampled on the way in.

The checks are invariants rather than fixed numbers, because they must hold
for *any* scene the maintainer happens to have, not just for one. The
strongest is that each single-bit Landsat flag must equal its own confidence
field reading High: that one statement exercises all eight bit positions and
all four two-bit field offsets against data neither we nor the test author
wrote.

Opt in with ``--run-realdata``, pointing the environment variables at
downloaded products::

    EEO_TEST_SENTINEL2_SCENE=~/Downloads/sen/S2A_....SAFE.zip \\
    EEO_TEST_LANDSAT_SCENE=~/Downloads/ls/LC09_....tar \\
    pytest tests/test_real_scenes.py --run-realdata

Nothing here downloads anything; the products must already be on disk.
"""

import numpy as np
import pytest

import eeo
from eeo.core.exceptions import ValidationError
from eeo.preprocessing.quality import (
    QAConfidence,
    QAConfidenceField,
    QAPixelFlag,
    SCLClass,
    qa_pixel_confidence,
    qa_pixel_flag,
    qa_pixel_mask,
    scl_mask,
)

pytestmark = pytest.mark.realdata

# Flag / confidence-field pairs that describe the same phenomenon. USGS
# defines each flag as "its confidence is High", so the two must agree
# everywhere, on every mission that has them.
FLAG_AND_FIELD = (
    (QAPixelFlag.CLOUD, QAConfidenceField.CLOUD),
    (QAPixelFlag.CLOUD_SHADOW, QAConfidenceField.CLOUD_SHADOW),
    (QAPixelFlag.SNOW, QAConfidenceField.SNOW_ICE),
    (QAPixelFlag.CIRRUS, QAConfidenceField.CIRRUS),
)

# The three fields for which USGS documents the value 2 as Reserved. A real
# product must therefore never contain it there.
FIELDS_RESERVING_TWO = (
    QAConfidenceField.CLOUD_SHADOW,
    QAConfidenceField.SNOW_ICE,
    QAConfidenceField.CIRRUS,
)


@pytest.fixture(scope="module")
def scl(sentinel2_scene):
    """The ``SCL`` band of the real Sentinel-2 product, read once."""
    return eeo.load_sentinel2(str(sentinel2_scene), bands=["scl"]).read()[0]


@pytest.fixture(scope="module")
def landsat(landsat_scene):
    """The ``QA_PIXEL`` band of the real Landsat product, and its mission.

    The mission is taken from the loaded dataset's own ``attrs`` rather than
    re-parsed from the product, because that is the route a chainable masking
    op has to it. It is recorded there as prose — ``"Landsat 9"`` — so the
    number a decoder needs has to be dug out of it.
    """
    ds = eeo.load_landsat(str(landsat_scene), bands=["qa_pixel"])
    mission = int(ds.attrs["mission"].rsplit(maxsplit=1)[-1])
    return ds.read()[0], mission


class TestRealSentinel2SCL:
    def test_every_value_present_is_a_documented_class(self, scl):
        # An undocumented class number would mean the enumeration is
        # incomplete, which no synthetic test can discover.
        present = set(np.unique(scl).tolist())
        assert present <= {member.value for member in SCLClass}

    def test_the_band_is_an_integer_class_map(self, scl):
        # Class numbers must arrive unmodified; a resampled SCL would show
        # fractional values and mean nothing.
        assert np.issubdtype(scl.dtype, np.integer)

    def test_the_scene_is_classified_rather_than_empty(self, scl):
        # Guards against silently reading the wrong asset: a scene of one
        # single class is a band we misidentified, not a classification.
        assert len(np.unique(scl)) > 1

    def test_the_mask_is_exactly_the_union_of_its_classes(self, scl):
        # np.isin over many classes must agree with the classes counted one
        # at a time, on a real class distribution rather than a crafted one.
        classes = [SCLClass.CLOUD_SHADOWS, SCLClass.CLOUD_HIGH_PROBABILITY]
        mask = scl_mask(scl, classes=classes)
        assert int(mask.sum()) == sum(int((scl == c).sum()) for c in classes)

    def test_masking_more_classes_never_masks_fewer_pixels(self, scl):
        few = scl_mask(scl, classes=[SCLClass.CLOUD_HIGH_PROBABILITY])
        many = scl_mask(scl, classes=[SCLClass.CLOUD_HIGH_PROBABILITY, SCLClass.THIN_CIRRUS])
        assert int(many.sum()) >= int(few.sum())
        assert (many | few == many).all()

    @pytest.mark.parametrize(("older", "current"), [("bare_soil", 5), ("dark_features", 2)])
    def test_a_renamed_class_resolves_to_the_same_real_pixels(self, scl, older, current):
        assert int(scl_mask(scl, classes=[older]).sum()) == int((scl == current).sum())

    def test_a_float_copy_decodes_identically(self, scl):
        # Stacking SCL beside float reflectance widens it; the decode must
        # survive that on real values, not just on hand-built ones.
        assert np.array_equal(scl_mask(scl.astype("float32")), scl_mask(scl))


class TestRealLandsatQAPixel:
    def test_the_band_is_an_integer_bit_field(self, landsat):
        qa, _ = landsat
        assert np.issubdtype(qa.dtype, np.integer)

    @pytest.mark.parametrize(("flag", "field"), FLAG_AND_FIELD)
    def test_each_flag_equals_its_confidence_field_reading_high(self, landsat, flag, field):
        # The strongest available check: it pins every bit position and every
        # two-bit field offset simultaneously, against agency-produced data.
        qa, mission = landsat
        if flag not in eeo.qa_pixel_flags(mission):
            pytest.skip(f"Landsat {mission} leaves {flag.name.lower()} Unused")
        by_flag = qa_pixel_flag(qa, flag, mission=mission)
        by_confidence = qa_pixel_confidence(qa, field, mission=mission) == QAConfidence.HIGH
        assert np.array_equal(by_flag, by_confidence)

    def test_clear_is_the_absence_of_cloud_and_dilated_cloud(self, landsat):
        # USGS derives bit 6 from bits 1 and 3. Fill pixels are exempt: every
        # bit is zero there, so 'clear' reads 0 where the derivation says 1.
        qa, mission = landsat
        clear = qa_pixel_flag(qa, QAPixelFlag.CLEAR, mission=mission)
        cloud = qa_pixel_flag(qa, QAPixelFlag.CLOUD, mission=mission)
        dilated = qa_pixel_flag(qa, QAPixelFlag.DILATED_CLOUD, mission=mission)
        fill = qa_pixel_flag(qa, QAPixelFlag.FILL, mission=mission)
        assert ((clear == ~(cloud | dilated)) | fill).all()

    @pytest.mark.parametrize("field", FIELDS_RESERVING_TWO)
    def test_a_reserved_confidence_value_never_appears(self, landsat, field):
        # If 2 turned up here, treating these fields as a three-level scale
        # would be defensible after all. It does not.
        qa, mission = landsat
        if field not in eeo.qa_pixel_confidence_fields(mission):
            pytest.skip(f"Landsat {mission} leaves {field.name.lower()} confidence Unused")
        assert 2 not in np.unique(qa_pixel_confidence(qa, field, mission=mission)).tolist()

    def test_cloud_confidence_does_use_the_medium_value(self, landsat):
        # The other half of the asymmetry: 2 means Medium here, and real
        # scenes contain it. Without this, the parametrized test above would
        # pass just as well if our field offsets were all wrong.
        qa, mission = landsat
        present = np.unique(qa_pixel_confidence(qa, QAConfidenceField.CLOUD, mission=mission))
        assert QAConfidence.MEDIUM in present.tolist()

    def test_medium_confidence_cloud_carries_no_flag_of_its_own(self, landsat):
        # The premise of the default threshold, on real data: these pixels
        # exist and the single-bit flags do not catch them.
        qa, mission = landsat
        medium = qa_pixel_confidence(qa, QAConfidenceField.CLOUD, mission=mission) == (
            QAConfidence.MEDIUM
        )
        assert medium.any(), "scene has no medium-confidence cloud to test with"
        assert not qa_pixel_flag(qa, QAPixelFlag.CLOUD, mission=mission)[medium].any()

    def test_the_default_mask_catches_what_the_flags_miss(self, landsat):
        qa, mission = landsat
        default = qa_pixel_mask(qa, mission=mission)
        flags_only = qa_pixel_mask(qa, mission=mission, min_cloud_confidence=None)
        assert (default | flags_only == default).all(), "the default must never mask less"
        assert int(default.sum()) > int(flags_only.sum()), "no mid-confidence cloud was caught"

    def test_a_high_threshold_reproduces_the_flags(self, landsat):
        qa, mission = landsat
        by_threshold = qa_pixel_mask(qa, mission=mission, min_cloud_confidence=QAConfidence.HIGH)
        by_flag = qa_pixel_mask(qa, mission=mission, min_cloud_confidence=None)
        assert np.array_equal(by_threshold, by_flag)

    def test_fill_marks_the_scene_corners_and_nothing_masks_more(self, landsat):
        # A north-up grid over a rotated WRS-2 footprint always has fill, and
        # the default mask must be a superset of it.
        qa, mission = landsat
        fill = qa_pixel_flag(qa, QAPixelFlag.FILL, mission=mission)
        assert fill.any(), "a Landsat scene on a north-up grid must have fill pixels"
        assert (qa_pixel_mask(qa, mission=mission) | fill).sum() == int(
            qa_pixel_mask(qa, mission=mission).sum()
        )

    def test_a_float_copy_decodes_identically(self, landsat):
        qa, mission = landsat
        assert np.array_equal(
            qa_pixel_mask(qa.astype("float32"), mission=mission),
            qa_pixel_mask(qa, mission=mission),
        )

    def test_reading_a_cirrus_bit_the_sensor_lacks_still_raises(self, landsat):
        qa, mission = landsat
        if mission in (8, 9):
            pytest.skip("this mission has a cirrus band")
        with pytest.raises(ValidationError, match="Unused"):
            qa_pixel_flag(qa, QAPixelFlag.CIRRUS, mission=mission)


class TestMaskCloudsOnRealScenes:
    """The whole point of WP-26, end to end on agency data."""

    def test_landsat_masks_and_records_its_nodata(self, landsat_scene):
        ds = eeo.load_landsat(str(landsat_scene), bands=["red", "nir08", "qa_pixel"])
        declared = ds.get_metadata()["nodata"]
        assert declared is not None, "Landsat C2 declares 0 as fill"
        out = ds.mask_clouds()
        assert out.get_metadata()["nodata"] == declared
        assert out.get_metadata()["dtype"] == ds.get_metadata()["dtype"]
        assert out.band_names == ds.band_names
        assert out.get_transform() == ds.get_transform()

    def test_landsat_masking_reaches_every_band(self, landsat_scene):
        ds = eeo.load_landsat(str(landsat_scene), bands=["red", "nir08", "qa_pixel"])
        out = ds.mask_clouds().read()
        fill = out[0] == 0
        assert ((out[1] == 0) == fill).all()
        assert fill.any() and not fill.all()

    def test_landsat_masking_removes_more_than_the_scene_already_lacked(self, landsat_scene):
        ds = eeo.load_landsat(str(landsat_scene), bands=["red", "qa_pixel"])
        before = ds.read()[0]
        after = ds.mask_clouds().read()[0]
        assert int((after == 0).sum()) > int((before == 0).sum()), "nothing was masked"

    def test_an_index_over_a_masked_scene_excludes_the_masked_pixels(self, landsat_scene):
        # WP-26's "Done when": ndvi over the masked result excludes the
        # flagged pixels. Asserted on a real cloudy scene.
        ds = eeo.load_landsat(str(landsat_scene), bands=["red", "nir08", "qa_pixel"])
        masked = ds.mask_clouds()
        nodata_pixels = masked.read()[0] == 0
        ndvi = masked.ndvi("red", nir="nir08").read()[0]
        assert np.isnan(ndvi)[nodata_pixels].all(), "a masked pixel reached the index"
        assert np.isfinite(ndvi).any(), "the whole scene was masked"

    def test_sentinel2_carries_the_fill_value_esa_declares(self, sentinel2_scene):
        # The JP2s declare no nodata; the manifest does. A real product is the
        # only place that distinction shows up — the fixture cannot prove that
        # ESA really states it this way.
        ds = eeo.load_sentinel2(str(sentinel2_scene), bands=["red", "scl"])
        assert ds.get_metadata()["nodata"] == 0

    def test_sentinel2_masks_without_being_told_the_fill_value(self, sentinel2_scene):
        ds = eeo.load_sentinel2(str(sentinel2_scene), bands=["red", "scl"])
        out = ds.mask_clouds()
        assert out.get_metadata()["nodata"] == 0
        assert out.band_names == ds.band_names

    def test_both_missions_agree_on_the_fill_value(self, sentinel2_scene, landsat_scene):
        # The contract WP-25 states: a workflow does not care which route or
        # which mission the data came from.
        s2 = eeo.load_sentinel2(str(sentinel2_scene), bands=["red"])
        ls = eeo.load_landsat(str(landsat_scene), bands=["red"])
        assert s2.get_metadata()["nodata"] == ls.get_metadata()["nodata"] == 0

    def test_sentinel2_masks_exactly_the_flagged_classes(self, sentinel2_scene):
        ds = eeo.load_sentinel2(str(sentinel2_scene), bands=["red", "scl"])
        scl = ds.read()[1]
        expected = scl_mask(scl)
        after = ds.mask_clouds(nodata=0).read()[0]
        # Every flagged pixel is nodata. The converse need not hold: a real
        # reflectance band already contains zeros of its own.
        assert (after[expected] == 0).all()

    def test_the_quality_band_is_upsampled_by_nearest_neighbour(self, sentinel2_scene):
        # SCL is a 20 m band; asking for it beside 10 m bands resamples it.
        # An interpolating method would invent class numbers, which is the
        # one thing that silently destroys a mask.
        ds = eeo.load_sentinel2(str(sentinel2_scene), bands=["red", "scl"])
        upsampled = ds.read()[1]
        assert ds.attrs["resolution"] == 10, "expected the 10 m grid"
        assert set(np.unique(upsampled).tolist()) <= {m.value for m in SCLClass}


class TestClearFractionOnRealScenes:
    """The number 20.2's composite will select scenes on."""

    def test_landsat_clear_fraction_falls_after_masking(self, landsat_scene):
        ds = eeo.load_landsat(str(landsat_scene), bands=["red", "qa_pixel"])
        before = ds.clear_fraction()
        after = ds.mask_clouds().clear_fraction()
        assert 0.0 < after < before < 1.0, "a real scene has both fill and cloud"

    def test_landsat_clear_fraction_matches_the_mask_it_was_told_about(self, landsat_scene):
        # The fraction must be the complement of what mask_clouds actually
        # wrote, not an independently drifting number.
        ds = eeo.load_landsat(str(landsat_scene), bands=["red", "qa_pixel"])
        masked = ds.mask_clouds()
        nodata_pixels = (masked.read() == 0).any(axis=0)
        assert masked.clear_fraction() == pytest.approx(1.0 - nodata_pixels.mean())

    def test_scene_edge_fill_dominates_the_whole_scene_figure(self, landsat_scene):
        # Why the docstring warns against reading a whole-scene clear fraction
        # as cloudiness: a north-up grid over a rotated WRS-2 footprint is
        # heavily fill before any cloud is masked.
        ds = eeo.load_landsat(str(landsat_scene), bands=["red"])
        assert ds.clear_fraction() < 0.9

    def test_sentinel2_clear_fraction_is_high_on_a_clear_tile(self, sentinel2_scene):
        ds = eeo.load_sentinel2(str(sentinel2_scene), bands=["red", "scl"])
        assert ds.mask_clouds().clear_fraction() > 0.9
