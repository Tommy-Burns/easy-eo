"""Tests for decoding the Landsat QA_PIXEL band (eeo/preprocessing/quality.py).

The strongest available check is that USGS publishes not just the bit index
but a table of real pixel values and what each one means. Those tables are
transcribed below and every field of every documented value is decoded and
compared, which tests the bit positions, the two-bit field offsets and the
per-mission gate all at once against the mission's own documentation rather
than against our reading of it.

Sources
-------
LSDS-1619 Landsat 8-9 Collection 2 Level-2 Science Product Guide v6,
Table 6-2 (bit index) and Table 6-3 (value interpretations).
LSDS-1618 Landsat 4-7 Collection 2 Level-2 Science Product Guide v4,
Table 5-5 (bit index) and Table 5-6 (value interpretations).
"""

import numpy as np
import pytest

from eeo.core.exceptions import ValidationError
from eeo.preprocessing.quality import (
    QA_PIXEL_CLOUDY,
    QA_PIXEL_DEFAULT_MASKED,
    QA_PIXEL_DEFAULT_MIN_CLOUD_CONFIDENCE,
    QA_PIXEL_NODATA,
    QAConfidence,
    QAConfidenceField,
    QAPixelFlag,
    _resolve_qa_confidence_field,
    confidence_has_medium,
    qa_pixel_confidence,
    qa_pixel_confidence_fields,
    qa_pixel_flag,
    qa_pixel_flags,
    qa_pixel_mask,
    resolve_qa_pixel_flags,
)

# Confidence words as the USGS tables spell them, to the raw two-bit value.
CONF = {"None": 0, "Low": 1, "Mid": 2, "High": 3}

# Table 6-3, verbatim. Flags in bit order (fill, dilated, cirrus, cloud,
# cloud shadow, snow, clear, water), then the four confidence fields
# (cloud, cloud shadow, snow/ice, cirrus).
L89_VALUES = {
    1: ("YNNNNNNN", "None", "None", "None", "None", "Fill"),
    21824: ("NNNNNNYN", "Low", "Low", "Low", "Low", "Clear with lows set"),
    21826: ("NYNNNNYN", "Low", "Low", "Low", "Low", "Dilated cloud over land"),
    21888: ("NNNNNNNY", "Low", "Low", "Low", "Low", "Water with lows set"),
    21890: ("NYNNNNNY", "Low", "Low", "Low", "Low", "Dilated cloud over water"),
    22080: ("NNNNNNYN", "Mid", "Low", "Low", "Low", "Mid conf cloud"),
    22144: ("NNNNNNNY", "Mid", "Low", "Low", "Low", "Mid conf cloud over water"),
    22280: ("NNNYNNNN", "High", "Low", "Low", "Low", "High conf Cloud"),
    23888: ("NNNNYNYN", "Low", "High", "Low", "Low", "High conf cloud shadow"),
    23952: ("NNNNYNNY", "Low", "High", "Low", "Low", "Water with cloud shadow"),
    24088: ("NNNYYNNN", "Mid", "High", "Low", "Low", "Mid conf cloud with shadow"),
    24216: ("NNNYYNNY", "Mid", "High", "Low", "Low", "Mid conf cloud, shadow, water"),
    24344: ("NNNYYNNN", "High", "High", "Low", "Low", "High conf cloud with shadow"),
    24472: ("NNNYYNNY", "High", "High", "Low", "Low", "High conf cloud, shadow, water"),
    30048: ("NNNNNYYN", "Low", "Low", "High", "Low", "High conf snow/ice"),
    54596: ("NNYNNNYN", "Low", "Low", "Low", "High", "High conf Cirrus"),
    54852: ("NNYNNNYN", "Mid", "Low", "Low", "High", "Cirrus, mid cloud"),
    55052: ("NNYYNNNN", "High", "Low", "Low", "High", "Cirrus, high cloud"),
}

# Table 5-6, verbatim. The cirrus flag is "N/A" on these sensors and the
# cirrus confidence field is Unused, so neither is decodable here; the flag
# string carries "-" in the cirrus position and cirrus confidence is omitted.
L47_VALUES = {
    1: ("YN-NNNNN", "None", "None", "None", "Fill"),
    5440: ("NN-NNNYN", "Low", "Low", "Low", "Clear with lows set"),
    5442: ("NY-NNNYN", "Low", "Low", "Low", "Dilated cloud over land"),
    5504: ("NN-NNNNY", "Low", "Low", "Low", "Water with lows set"),
    5506: ("NY-NNNNY", "Low", "Low", "Low", "Dilated cloud over water"),
    5696: ("NN-NNNYN", "Mid", "Low", "Low", "Mid conf cloud"),
    5760: ("NN-NNNNY", "Mid", "Low", "Low", "Mid conf cloud over water"),
    5896: ("NN-YNNNN", "High", "Low", "Low", "High conf Cloud"),
    7440: ("NN-NYNNN", "Low", "High", "Low", "High conf cloud shadow"),
    7568: ("NN-NYNNY", "Low", "High", "Low", "Water with cloud shadow"),
    7696: ("NN-NYNNN", "Mid", "High", "Low", "Mid conf cloud with shadow"),
    7824: ("NN-NYNNY", "Mid", "High", "Low", "Mid conf cloud, shadow, water"),
    7960: ("NN-YYNNN", "High", "High", "Low", "High conf cloud with shadow"),
    8088: ("NN-YYNNY", "High", "High", "Low", "High conf cloud, shadow, water"),
    13664: ("NN-NNYYN", "Low", "Low", "High", "High conf snow/ice"),
}

# The flag string's positions, in the order the USGS tables print them.
FLAG_ORDER = (
    QAPixelFlag.FILL,
    QAPixelFlag.DILATED_CLOUD,
    QAPixelFlag.CIRRUS,
    QAPixelFlag.CLOUD,
    QAPixelFlag.CLOUD_SHADOW,
    QAPixelFlag.SNOW,
    QAPixelFlag.CLEAR,
    QAPixelFlag.WATER,
)


class TestBitIndexAgainstUSGSValues:
    """Every documented pixel value, decoded field by field."""

    @pytest.mark.parametrize(("value", "row"), sorted(L89_VALUES.items()))
    def test_landsat_8_9_flags(self, value, row):
        qa = np.array([value], dtype="uint16")
        expected = dict(zip(FLAG_ORDER, row[0], strict=True))
        for flag, want in expected.items():
            got = bool(qa_pixel_flag(qa, flag, mission=9)[0])
            assert got is (want == "Y"), f"{value}: {flag.name} should be {want}"

    @pytest.mark.parametrize(("value", "row"), sorted(L89_VALUES.items()))
    def test_landsat_8_9_confidence_fields(self, value, row):
        qa = np.array([value], dtype="uint16")
        fields = (
            QAConfidenceField.CLOUD,
            QAConfidenceField.CLOUD_SHADOW,
            QAConfidenceField.SNOW_ICE,
            QAConfidenceField.CIRRUS,
        )
        for field, want in zip(fields, row[1:5], strict=True):
            got = int(qa_pixel_confidence(qa, field, mission=9)[0])
            assert got == CONF[want], f"{value}: {field.name} should be {want}"

    @pytest.mark.parametrize(("value", "row"), sorted(L47_VALUES.items()))
    def test_landsat_4_7_flags(self, value, row):
        qa = np.array([value], dtype="uint16")
        for flag, want in zip(FLAG_ORDER, row[0], strict=True):
            if want == "-":  # cirrus: no such bit on TM or ETM+
                continue
            got = bool(qa_pixel_flag(qa, flag, mission=7)[0])
            assert got is (want == "Y"), f"{value}: {flag.name} should be {want}"

    @pytest.mark.parametrize(("value", "row"), sorted(L47_VALUES.items()))
    def test_landsat_4_7_confidence_fields(self, value, row):
        qa = np.array([value], dtype="uint16")
        fields = (
            QAConfidenceField.CLOUD,
            QAConfidenceField.CLOUD_SHADOW,
            QAConfidenceField.SNOW_ICE,
        )
        for field, want in zip(fields, row[1:4], strict=True):
            got = int(qa_pixel_confidence(qa, field, mission=7)[0])
            assert got == CONF[want], f"{value}: {field.name} should be {want}"

    def test_landsat_4_7_never_sets_the_bits_it_documents_as_unused(self):
        # Bit 2 and bits 14-15 are Unused on TM and ETM+; no documented value
        # may have them set, which is the check that our gate is not hiding a
        # real signal.
        unused = (1 << 2) | (1 << 14) | (1 << 15)
        for value in L47_VALUES:
            assert value & unused == 0, f"{value} sets a bit documented as Unused"


class TestBitPositions:
    """The enumerations are bit positions, transcribed from the bit index."""

    def test_flag_bit_positions(self):
        assert {f.name: f.value for f in QAPixelFlag} == {
            "FILL": 0,
            "DILATED_CLOUD": 1,
            "CIRRUS": 2,
            "CLOUD": 3,
            "CLOUD_SHADOW": 4,
            "SNOW": 5,
            "CLEAR": 6,
            "WATER": 7,
        }

    def test_confidence_fields_start_at_their_low_bit(self):
        assert {f.name: f.value for f in QAConfidenceField} == {
            "CLOUD": 8,
            "CLOUD_SHADOW": 10,
            "SNOW_ICE": 12,
            "CIRRUS": 14,
        }

    def test_confidence_levels(self):
        assert [c.value for c in QAConfidence] == [0, 1, 2, 3]

    def test_only_cloud_confidence_defines_medium(self):
        # USGS documents 10 as Reserved for the other three fields, so a
        # three-level reading of them would invent a level.
        assert confidence_has_medium(QAConfidenceField.CLOUD)
        for field in (
            QAConfidenceField.CLOUD_SHADOW,
            QAConfidenceField.SNOW_ICE,
            QAConfidenceField.CIRRUS,
        ):
            assert not confidence_has_medium(field)


class TestPerMissionGate:
    """Bit 2 and bits 14-15 exist on OLI only."""

    @pytest.mark.parametrize("mission", [8, 9])
    def test_oli_has_every_flag_and_field(self, mission):
        assert qa_pixel_flags(mission) == set(QAPixelFlag)
        assert qa_pixel_confidence_fields(mission) == set(QAConfidenceField)

    @pytest.mark.parametrize("mission", [4, 5, 7])
    def test_tm_and_etm_lack_cirrus_only(self, mission):
        assert qa_pixel_flags(mission) == set(QAPixelFlag) - {QAPixelFlag.CIRRUS}
        assert qa_pixel_confidence_fields(mission) == set(QAConfidenceField) - {
            QAConfidenceField.CIRRUS
        }

    @pytest.mark.parametrize("mission", [4, 5, 7])
    def test_reading_cirrus_on_tm_raises_rather_than_reporting_a_constant(self, mission):
        qa = np.array([0], dtype="uint16")
        with pytest.raises(ValidationError, match="Unused"):
            qa_pixel_flag(qa, "cirrus", mission=mission)
        with pytest.raises(ValidationError, match="Unused"):
            qa_pixel_confidence(qa, "cirrus", mission=mission)

    @pytest.mark.parametrize("mission", [1, 3, 6, 10, 0])
    def test_an_undocumented_mission_is_refused(self, mission):
        with pytest.raises(ValidationError, match="no QA_PIXEL bit index"):
            qa_pixel_flags(mission)

    def test_mission_must_be_a_number(self):
        with pytest.raises(ValidationError, match="mission must be"):
            qa_pixel_flags("nine")


class TestResolveFlags:
    """Members, bit positions and names must all name the same flags."""

    def test_accepts_members_bits_and_names_alike(self):
        assert (
            resolve_qa_pixel_flags([QAPixelFlag.CLOUD])
            == resolve_qa_pixel_flags([3])
            == resolve_qa_pixel_flags(["cloud"])
            == resolve_qa_pixel_flags([" Cloud "])
        )

    def test_rejects_a_bare_string(self):
        with pytest.raises(ValidationError, match="sequence of QA_PIXEL flags"):
            resolve_qa_pixel_flags("cloud")

    def test_rejects_an_empty_sequence(self):
        with pytest.raises(ValidationError, match="at least one QA_PIXEL flag"):
            resolve_qa_pixel_flags([])

    @pytest.mark.parametrize("bit", [8, 10, 12, 14])
    def test_a_confidence_field_is_not_a_flag(self, bit):
        # Reading one bit of a two-bit field answers half a question.
        with pytest.raises(ValidationError, match="paired confidence fields"):
            resolve_qa_pixel_flags([bit])

    def test_rejects_an_unknown_name(self):
        with pytest.raises(ValidationError, match="no QA_PIXEL flag named"):
            resolve_qa_pixel_flags(["cloudy"])

    @pytest.mark.parametrize("value", [None, 3.0, True, ["cloud"]])
    def test_rejects_a_value_that_is_not_a_flag(self, value):
        with pytest.raises(ValidationError, match="must be a QAPixelFlag"):
            resolve_qa_pixel_flags([value])


class TestDefaultFlagSets:
    def test_cloudy_set_is_dilated_cirrus_cloud_and_shadow(self):
        # Asserted as bit positions, so the set is pinned to the bit index
        # rather than to the names we chose for it.
        assert {flag.value for flag in QA_PIXEL_CLOUDY} == {1, 2, 3, 4}

    def test_nodata_set_is_the_fill_bit(self):
        assert {flag.value for flag in QA_PIXEL_NODATA} == {0}

    def test_default_is_the_union(self):
        assert QA_PIXEL_DEFAULT_MASKED == QA_PIXEL_CLOUDY | QA_PIXEL_NODATA

    def test_defaults_keep_the_surface_and_derived_flags(self):
        # Snow and water are surfaces; clear is derived from the cloud bits
        # and would double-count.
        assert not (QA_PIXEL_DEFAULT_MASKED & {QAPixelFlag.SNOW, QAPixelFlag.WATER})
        assert QAPixelFlag.CLEAR not in QA_PIXEL_DEFAULT_MASKED

    def test_sets_are_immutable(self):
        assert isinstance(QA_PIXEL_CLOUDY, frozenset)
        assert isinstance(QA_PIXEL_NODATA, frozenset)
        assert isinstance(QA_PIXEL_DEFAULT_MASKED, frozenset)


class TestQAPixelMask:
    def test_default_masks_every_documented_cloudy_value(self):
        # Every Table 6-3 row whose description names cloud, shadow or cirrus,
        # excluding the snow/ice row. Mid-confidence cloud is in here, which is
        # what the confidence threshold exists to catch.
        cloudy = [
            value
            for value, row in L89_VALUES.items()
            if ("conf" in row[-1] or "irrus" in row[-1]) and "snow" not in row[-1]
        ]
        qa = np.array(cloudy, dtype="uint16")
        assert qa_pixel_mask(qa, mission=9).all()

    def test_mid_confidence_cloud_carries_no_flag_of_its_own(self):
        # The premise of the confidence threshold, asserted directly against
        # USGS's "Mid conf cloud" rows: the cloud bit fires at High only, so
        # a flags-only mask lets medium-confidence cloud through.
        mid = np.array([22080, 22144], dtype="uint16")
        assert not qa_pixel_mask(mid, mission=9, min_cloud_confidence=None).any()
        assert qa_pixel_mask(mid, mission=9).all()

    def test_default_threshold_is_medium(self):
        assert QA_PIXEL_DEFAULT_MIN_CLOUD_CONFIDENCE == QAConfidence.MEDIUM

    def test_a_high_threshold_reproduces_the_flag(self):
        qa = np.array(sorted(L89_VALUES), dtype="uint16")
        by_threshold = qa_pixel_mask(qa, mission=9, min_cloud_confidence=QAConfidence.HIGH)
        by_flag = qa_pixel_mask(qa, mission=9, min_cloud_confidence=None)
        assert np.array_equal(by_threshold, by_flag)

    def test_thresholds_are_monotonic(self):
        qa = np.array(sorted(L89_VALUES), dtype="uint16")
        counts = [
            int(qa_pixel_mask(qa, mission=9, min_cloud_confidence=level).sum())
            for level in (QAConfidence.HIGH, QAConfidence.MEDIUM, QAConfidence.LOW)
        ]
        assert counts == sorted(counts), "a lower threshold must never mask fewer pixels"

    @pytest.mark.parametrize("value", [4, -1, "mid", 2.0, True])
    def test_an_invalid_threshold_is_refused(self, value):
        qa = np.array([0], dtype="uint16")
        with pytest.raises(ValidationError):
            qa_pixel_mask(qa, mission=9, min_cloud_confidence=value)

    def test_threshold_accepts_a_level_name_or_number(self):
        qa = np.array(sorted(L89_VALUES), dtype="uint16")
        expected = qa_pixel_mask(qa, mission=9, min_cloud_confidence=QAConfidence.MEDIUM)
        assert np.array_equal(qa_pixel_mask(qa, mission=9, min_cloud_confidence=2), expected)
        assert np.array_equal(qa_pixel_mask(qa, mission=9, min_cloud_confidence="medium"), expected)

    def test_default_keeps_clear_water_and_snow(self):
        # 21824 clear, 21888 water, 30048 high-confidence snow/ice: all three
        # sit at Low cloud confidence, so the threshold does not touch them.
        clear = np.array([21824, 21888, 30048], dtype="uint16")
        assert not qa_pixel_mask(clear, mission=9).any()

    def test_fill_is_masked(self):
        assert qa_pixel_mask(np.array([1], dtype="uint16"), mission=9)[0]

    def test_default_drops_cirrus_on_a_sensor_without_it(self):
        # The default must work on every mission rather than raising.
        qa = np.array([5896, 5440], dtype="uint16")
        assert qa_pixel_mask(qa, mission=7).tolist() == [True, False]

    def test_the_threshold_applies_on_every_mission(self):
        # Cloud confidence is bits 8-9 on all five missions; 5696 is Table
        # 5-6's "Mid conf cloud" for TM and ETM+.
        mid = np.array([5696], dtype="uint16")
        assert not qa_pixel_mask(mid, mission=7, min_cloud_confidence=None)[0]
        assert qa_pixel_mask(mid, mission=7)[0]

    def test_naming_cirrus_explicitly_on_tm_still_raises(self):
        # Silently dropping a flag the caller asked for by name would hide a
        # misunderstanding rather than correct it.
        qa = np.array([0], dtype="uint16")
        with pytest.raises(ValidationError, match="Unused"):
            qa_pixel_mask(qa, mission=7, flags=["cirrus"])

    def test_selecting_one_flag(self):
        qa = np.array([0, 1 << 5, 1 << 3], dtype="uint16")
        got = qa_pixel_mask(qa, mission=9, flags=["snow"], min_cloud_confidence=None)
        assert got.tolist() == [False, True, False]

    def test_several_flags_are_combined_with_or(self):
        qa = np.array([1 << 3, 1 << 4, 1 << 7], dtype="uint16")
        got = qa_pixel_mask(
            qa, mission=9, flags=["cloud", "cloud_shadow"], min_cloud_confidence=None
        )
        assert got.tolist() == [True, True, False]

    def test_returns_a_boolean_array_of_the_input_shape(self):
        mask = qa_pixel_mask(np.zeros((2, 5, 7), dtype="uint16"), mission=9)
        assert mask.dtype == np.bool_
        assert mask.shape == (2, 5, 7)

    def test_does_not_modify_the_input(self):
        qa = np.array(sorted(L89_VALUES), dtype="uint16")
        before = qa.copy()
        qa_pixel_mask(qa, mission=9)
        assert np.array_equal(qa, before)


class TestFloatBitField:
    """A QA band stacked beside float reflectance arrives as float."""

    def test_a_float_band_is_read_back_as_a_bit_field(self):
        qa = np.array(sorted(L89_VALUES), dtype="float32")
        expected = qa_pixel_mask(np.array(sorted(L89_VALUES), dtype="uint16"), mission=9)
        assert np.array_equal(qa_pixel_mask(qa, mission=9), expected)

    def test_every_documented_value_survives_a_float32_round_trip(self):
        # float32 has a 24-bit mantissa, so every uint16 is exact; this is
        # what makes reading a widened band lossless rather than lucky.
        values = np.array(sorted(L89_VALUES) + sorted(L47_VALUES), dtype="uint16")
        assert np.array_equal(values.astype("float32").astype("uint16"), values)

    def test_a_float_confidence_field_decodes_the_same(self):
        qa_int = np.array([24344], dtype="uint16")
        qa_float = qa_int.astype("float64")
        for field in QAConfidenceField:
            assert qa_pixel_confidence(qa_float, field, mission=9) == qa_pixel_confidence(
                qa_int, field, mission=9
            )


class TestEveryBitInIsolation:
    """One bit set at a time, which the value table cannot show.

    Table 6-3's values each set several bits at once, so a decoder that read
    two flags from one bit, or shifted a confidence field by one, could still
    satisfy every documented value if the errors happened to cancel. Setting
    exactly one bit and asserting exactly one flag rules that out.
    """

    @pytest.mark.parametrize("flag", sorted(QAPixelFlag, key=lambda f: f.value))
    def test_a_lone_flag_bit_sets_only_its_own_flag(self, flag):
        qa = np.array([1 << flag.value], dtype="uint16")
        for other in QAPixelFlag:
            expected = other is flag
            assert bool(qa_pixel_flag(qa, other, mission=9)[0]) is expected, (
                f"bit {flag.value} should set only {flag.name}, but "
                f"{other.name} read {not expected}"
            )

    @pytest.mark.parametrize("flag", sorted(QAPixelFlag, key=lambda f: f.value))
    def test_a_lone_flag_bit_disturbs_no_confidence_field(self, flag):
        # Bits 0-7 are the flags; none of them may leak into bits 8-15.
        qa = np.array([1 << flag.value], dtype="uint16")
        for field in QAConfidenceField:
            assert int(qa_pixel_confidence(qa, field, mission=9)[0]) == 0

    @pytest.mark.parametrize("field", sorted(QAConfidenceField, key=lambda f: f.value))
    @pytest.mark.parametrize("level", [1, 2, 3])
    def test_each_confidence_field_reads_its_own_two_bits(self, field, level):
        qa = np.array([level << field.value], dtype="uint16")
        assert int(qa_pixel_confidence(qa, field, mission=9)[0]) == level
        for other in QAConfidenceField:
            if other is not field:
                assert int(qa_pixel_confidence(qa, other, mission=9)[0]) == 0

    @pytest.mark.parametrize("field", sorted(QAConfidenceField, key=lambda f: f.value))
    def test_a_confidence_field_never_leaks_into_the_flag_bits(self, field):
        # The product sets a flag bit and its confidence field together, but
        # they are independent bits: writing bits 8-15 alone must leave every
        # flag in bits 0-7 clear. A shift in the wrong direction would show up
        # here as a flag appearing from nowhere.
        qa = np.array([3 << field.value], dtype="uint16")
        for flag in QAPixelFlag:
            assert not qa_pixel_flag(qa, flag, mission=9)[0], (
                f"{field.name} confidence at High leaked into flag {flag.name}"
            )

    def test_the_sixteen_bits_account_for_every_value(self):
        # Nothing in a uint16 QA_PIXEL falls outside the documented layout:
        # eight flag bits plus four two-bit fields is exactly 16.
        covered = 0
        for flag in QAPixelFlag:
            covered |= 1 << flag.value
        for field in QAConfidenceField:
            covered |= 3 << field.value
        assert covered == 0xFFFF


class TestResolveConfidenceField:
    """Naming a confidence field, and every way of naming one badly.

    The flag resolver's error paths are covered above; this is its twin, and
    the messages are what a caller who mixed up bits and fields actually
    sees.
    """

    def test_accepts_members_low_bits_and_names_alike(self):
        assert (
            _resolve_qa_confidence_field(QAConfidenceField.CLOUD_SHADOW)
            == _resolve_qa_confidence_field(10)
            == _resolve_qa_confidence_field("cloud_shadow")
            == _resolve_qa_confidence_field(" Cloud_Shadow ")
        )

    def test_rejects_an_unknown_name_and_lists_the_real_ones(self):
        with pytest.raises(ValidationError, match="no QA_PIXEL confidence field named") as e:
            _resolve_qa_confidence_field("cloudiness")
        assert "snow_ice" in str(e.value)

    @pytest.mark.parametrize("bit", [0, 3, 9, 11, 15, 16])
    def test_rejects_a_bit_that_does_not_start_a_field(self, bit):
        # 9, 11 and 15 are the *high* bits of real fields: reading from there
        # would take one bit of one field and one of the next.
        with pytest.raises(ValidationError, match="does not start a QA_PIXEL confidence field"):
            _resolve_qa_confidence_field(bit)

    @pytest.mark.parametrize("value", [None, 8.0, True, ["cloud"], object()])
    def test_rejects_a_value_that_is_not_a_field_at_all(self, value):
        with pytest.raises(ValidationError, match="must be a QAConfidenceField"):
            _resolve_qa_confidence_field(value)

    def test_a_numpy_integer_low_bit_is_accepted(self):
        assert _resolve_qa_confidence_field(np.uint8(12)) is QAConfidenceField.SNOW_ICE

    def test_the_error_reaches_the_public_reader(self):
        qa = np.array([0], dtype="uint16")
        with pytest.raises(ValidationError, match="does not start a QA_PIXEL confidence field"):
            qa_pixel_confidence(qa, 9, mission=9)


class TestBitFieldConversion:
    """Reading bits out of whatever the caller happens to hold."""

    def test_a_plain_python_list_is_read_as_a_bit_field(self):
        # No dtype and no .astype, so neither of the fast paths applies.
        assert qa_pixel_flag([1, 1 << 3, 0], "fill", mission=9).tolist() == [True, False, False]

    def test_a_list_decodes_a_confidence_field_too(self):
        assert qa_pixel_confidence([3 << 8, 0], "cloud", mission=9).tolist() == [3, 0]

    def test_a_list_and_an_array_agree(self):
        values = sorted(L89_VALUES)
        assert np.array_equal(
            qa_pixel_mask(values, mission=9),
            qa_pixel_mask(np.array(values, dtype="uint16"), mission=9),
        )
