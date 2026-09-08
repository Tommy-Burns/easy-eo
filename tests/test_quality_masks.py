"""Tests for decoding product quality layers (eeo/preprocessing/quality.py).

A mask is a claim about which pixels are real, and a wrong one is invisible:
it does not raise, it just quietly deletes ground or admits cloud. The checks
here are therefore mostly about the class numbers themselves — that the
enumeration matches ESA's published numbering exactly, and that the default
sets contain what the documentation says they contain — plus the decoder
against hand-built arrays covering every class.
"""

import numpy as np
import pytest

from eeo.core.exceptions import ValidationError
from eeo.preprocessing.quality import (
    SCL_CLOUDY,
    SCL_DEFAULT_MASKED,
    SCL_NODATA,
    SCLClass,
    resolve_scl_classes,
    scl_mask,
)

# The Scene Classification table as Copernicus publishes it, at
# https://sentiwiki.copernicus.eu/web/s2-processing. This is the ground truth
# the enumeration is checked against; it is spelled out separately on purpose,
# so that a typo in the enum cannot be a typo in its own test. Class 11 is
# "SNOW or ICE" upstream, which is not a Python identifier.
ESA_CLASSES = {
    0: "NO_DATA",
    1: "SATURATED_OR_DEFECTIVE",
    2: "CAST_SHADOWS",
    3: "CLOUD_SHADOWS",
    4: "VEGETATION",
    5: "NOT_VEGETATED",
    6: "WATER",
    7: "UNCLASSIFIED",
    8: "CLOUD_MEDIUM_PROBABILITY",
    9: "CLOUD_HIGH_PROBABILITY",
    10: "THIN_CIRRUS",
    11: "SNOW_ICE",
}

# Names ESA moved away from, which existing scripts and tutorials still use.
RENAMED = {"dark_features": 2, "dark_area": 2, "bare_soil": 5, "cloud_shadow": 3}


class TestSCLClass:
    """The enumeration must match ESA's numbering exactly."""

    def test_has_exactly_the_twelve_classes(self):
        assert {member.value: member.name for member in SCLClass} == ESA_CLASSES

    @pytest.mark.parametrize(("number", "name"), sorted(ESA_CLASSES.items()))
    def test_each_class_number_maps_to_its_name(self, number, name):
        assert SCLClass(number).name == name
        assert SCLClass[name].value == number

    def test_members_are_usable_as_plain_integers(self):
        # IntEnum, so a member must be interchangeable with the raw class
        # number a user reads off the ESA table.
        assert SCLClass.CLOUD_HIGH_PROBABILITY == 9
        assert np.uint8(9) == SCLClass.CLOUD_HIGH_PROBABILITY


class TestDefaultClassSets:
    """The documented defaults, asserted as the numbers they are."""

    def test_cloudy_set_is_shadows_both_probabilities_and_cirrus(self):
        assert {int(c) for c in SCL_CLOUDY} == {3, 8, 9, 10}

    def test_nodata_set_is_no_data_and_saturated(self):
        assert {int(c) for c in SCL_NODATA} == {0, 1}

    def test_default_masked_is_the_union_of_the_two(self):
        assert SCL_DEFAULT_MASKED == SCL_CLOUDY | SCL_NODATA
        assert {int(c) for c in SCL_DEFAULT_MASKED} == {0, 1, 3, 8, 9, 10}

    def test_defaults_keep_every_surface_class(self):
        # Vegetation, soil, water and snow are surfaces, not obstructions;
        # masking one by default would silently delete a legitimate analysis.
        surfaces = {
            SCLClass.VEGETATION,
            SCLClass.NOT_VEGETATED,
            SCLClass.WATER,
            SCLClass.SNOW_ICE,
        }
        assert not (SCL_DEFAULT_MASKED & surfaces)

    def test_sets_are_immutable(self):
        # Exported module state: a caller must not be able to edit the default
        # for every other caller in the process.
        assert isinstance(SCL_CLOUDY, frozenset)
        assert isinstance(SCL_NODATA, frozenset)
        assert isinstance(SCL_DEFAULT_MASKED, frozenset)


class TestResolveSCLClasses:
    """Members, numbers and names must all name the same classes."""

    def test_accepts_members_numbers_and_names_alike(self):
        assert (
            resolve_scl_classes([SCLClass.CLOUD_SHADOWS])
            == resolve_scl_classes([3])
            == resolve_scl_classes(["cloud_shadows"])
        )

    def test_names_match_case_insensitively_and_ignore_whitespace(self):
        assert resolve_scl_classes([" Thin_Cirrus "]) == {SCLClass.THIN_CIRRUS}

    def test_duplicates_collapse(self):
        assert resolve_scl_classes([9, "cloud_high_probability", SCLClass(9)]) == {SCLClass(9)}

    def test_rejects_a_bare_string(self):
        # Iterating "water" yields characters, so accepting it would resolve
        # five nonexistent class names instead of one real one.
        with pytest.raises(ValidationError, match="sequence of scene classes"):
            resolve_scl_classes("water")

    def test_rejects_an_empty_sequence(self):
        with pytest.raises(ValidationError, match="at least one scene class"):
            resolve_scl_classes([])

    @pytest.mark.parametrize("number", [-1, 12, 255])
    def test_rejects_a_number_outside_the_table(self, number):
        with pytest.raises(ValidationError, match="not a Sentinel-2 scene class"):
            resolve_scl_classes([number])

    def test_rejects_an_unknown_name_and_lists_the_real_ones(self):
        with pytest.raises(ValidationError, match="no scene class named") as excinfo:
            resolve_scl_classes(["cloudy"])
        assert "cloud_high_probability" in str(excinfo.value)

    @pytest.mark.parametrize("value", [None, 3.0, True, ["cloud_shadows"]])
    def test_rejects_a_value_that_is_not_a_class_at_all(self, value):
        with pytest.raises(ValidationError, match="must be an SCLClass"):
            resolve_scl_classes([value])


class TestSCLMask:
    """The decoder, against arrays covering every class."""

    @staticmethod
    def _every_class():
        """A 3x4 array holding each of the twelve classes exactly once."""
        return np.arange(12, dtype="uint8").reshape(3, 4)

    def test_default_flags_exactly_the_default_classes(self):
        flagged = self._every_class()[scl_mask(self._every_class())]
        assert set(flagged.tolist()) == {0, 1, 3, 8, 9, 10}

    @pytest.mark.parametrize("number", sorted(ESA_CLASSES))
    def test_each_class_can_be_selected_on_its_own(self, number):
        scl = self._every_class()
        assert scl[scl_mask(scl, classes=[number])].tolist() == [number]

    def test_selecting_every_class_flags_every_pixel(self):
        scl = self._every_class()
        assert scl_mask(scl, classes=list(SCLClass)).all()

    def test_a_class_absent_from_the_scene_flags_nothing(self):
        scl = np.full((4, 4), SCLClass.VEGETATION, dtype="uint8")
        assert not scl_mask(scl, classes=["snow_ice"]).any()

    def test_returns_a_boolean_array_of_the_input_shape(self):
        mask = scl_mask(np.zeros((2, 5, 7), dtype="uint8"))
        assert mask.dtype == np.bool_
        assert mask.shape == (2, 5, 7)

    def test_does_not_modify_the_input(self):
        scl = self._every_class()
        before = scl.copy()
        scl_mask(scl)
        assert np.array_equal(scl, before)

    def test_decodes_a_float_band_by_exact_class_value(self):
        # Stacking SCL beside float reflectance widens its dtype; the class
        # numbers are unchanged by that and must still decode.
        scl = self._every_class().astype("float32")
        assert scl[scl_mask(scl)].tolist() == [0.0, 1.0, 3.0, 8.0, 9.0, 10.0]

    def test_invalid_classes_are_rejected_before_any_decoding(self):
        with pytest.raises(ValidationError):
            scl_mask(self._every_class(), classes=[99])


def test_class_numbers_read_out_of_an_array_are_accepted():
    """np.unique(scl) yields numpy integers, which are not ``int``."""
    scl = np.array([[3, 4], [9, 4]], dtype="uint8")
    present = np.unique(scl)
    assert resolve_scl_classes(present) == {
        SCLClass.CLOUD_SHADOWS,
        SCLClass.VEGETATION,
        SCLClass.CLOUD_HIGH_PROBABILITY,
    }


class TestRenamedClasses:
    """ESA renamed two classes without renumbering them.

    Class 2 was ``DARK_FEATURES`` before the 2022 processing-baseline change
    and is ``CAST_SHADOWS`` now; class 5 is ``NOT_VEGETATED``, which most
    tutorials still call "bare soil". A script written against the older
    vocabulary must keep resolving to the same class numbers.
    """

    @pytest.mark.parametrize(("name", "number"), sorted(RENAMED.items()))
    def test_older_name_resolves_to_the_same_class_number(self, name, number):
        assert resolve_scl_classes([name]) == {SCLClass(number)}

    def test_older_and_current_names_are_the_same_class(self):
        assert resolve_scl_classes(["bare_soil"]) == resolve_scl_classes(["not_vegetated"])
        assert resolve_scl_classes(["dark_features"]) == resolve_scl_classes(["cast_shadows"])

    def test_an_alias_is_not_a_thirteenth_class(self):
        # Aliases must map onto the enum, never widen it.
        assert len(SCLClass) == 12
        assert all(resolve_scl_classes([name]) <= set(SCLClass) for name in RENAMED)
