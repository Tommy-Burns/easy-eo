"""Tests for Landsat product identification (eeo/io/_landsat.py).

One canonical description of a product, from :mod:`product_fixtures`, is
rendered into all three metadata syntaxes, so "the three agree" is asserted
rather than assumed. The group layout, key names, and value spellings follow
real USGS output, including the parts most likely to catch a parser out: a
Level-1 section carrying the same key names as the Level-2 one with different
values, a scene centre time written with seven fractional digits, and a WRS row
that is zero-padded in the text syntax but not in the JSON.
"""

import datetime as dt
from pathlib import PurePosixPath

import pytest

from eeo.core.exceptions import ValidationError
from eeo.io._archive import open_product
from eeo.io._landsat import find_metadata, read_metadata_groups, read_product
from product_fixtures import (
    L1_ID as L1_PRODUCT_ID,
)
from product_fixtures import (
    L9_ID as PRODUCT_ID,
)
from product_fixtures import (
    as_xml,
)
from product_fixtures import (
    landsat_groups as spec,
)
from product_fixtures import (
    write_landsat_metadata as write_product,
)


class TestFindMetadata:
    """The metadata file is found by suffix, preferring XML."""

    @pytest.mark.parametrize("syntax", ["xml", "json", "txt"])
    def test_each_syntax_is_found(self, tmp_path, syntax):
        directory = write_product(tmp_path, syntax)
        assert find_metadata(directory).suffix == f".{syntax}"

    def test_xml_wins_when_several_are_present(self, tmp_path):
        directory = write_product(tmp_path, "txt")
        write_product(tmp_path, "json")
        write_product(tmp_path, "xml")
        assert find_metadata(directory).suffix == ".xml"

    def test_a_named_file_wins_over_the_preferred_suffix(self, tmp_path):
        # Naming the .txt must select it even though .xml is preferred when
        # searching, because the caller has already chosen.
        write_product(tmp_path, "xml")
        directory = write_product(tmp_path, "txt")
        named = directory / f"{PRODUCT_ID}_MTL.txt"
        assert find_metadata(named) == PurePosixPath(named.name)

    def test_two_products_in_one_folder_are_refused(self, tmp_path):
        write_product(tmp_path)
        write_product(tmp_path, mission_prefix="LE07")
        with pytest.raises(ValidationError, match="holds 2 Landsat products"):
            find_metadata(tmp_path)

    def test_missing_path_rejected(self, tmp_path):
        with pytest.raises(ValidationError, match="no such Landsat product"):
            find_metadata(tmp_path / "absent")

    def test_directory_without_metadata_rejected(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(ValidationError, match="holds no Landsat metadata file"):
            find_metadata(tmp_path / "empty")


class TestIdentityComesFromTheMetadata:
    """A renamed directory must not change what the product is."""

    def test_a_renamed_directory_is_ignored(self, tmp_path):
        # Renaming a folder is something users do; it must not be able to
        # change the mission, the level, or the scene the reader reports.
        directory = write_product(tmp_path)
        renamed = directory.rename(tmp_path / "landsat_scene")
        product = read_product(renamed)
        assert product.product_id == PRODUCT_ID
        assert product.mission == 9
        assert product.level == "L2SP"

    def test_a_directory_named_after_another_mission_is_ignored(self, tmp_path):
        directory = write_product(tmp_path)
        renamed = directory.rename(tmp_path / "LE07_L2SP_193028_20231024_20231119_02_T2")
        assert read_product(renamed).mission == 9


class TestSyntaxesAgree:
    """All three syntaxes carry identical content and must parse identically."""

    @pytest.mark.parametrize("syntax", ["xml", "json", "txt"])
    def test_identity_is_the_same_in_every_syntax(self, tmp_path, syntax):
        product = read_product(write_product(tmp_path, syntax))
        assert product.product_id == PRODUCT_ID
        assert product.level == "L2SP"
        assert (product.mission, product.instrument) == (9, "OLI-2/TIRS-2")
        assert product.spacecraft == "LANDSAT_9"
        assert (product.wrs_path, product.wrs_row) == ("193", "028")
        assert product.collection_number == "02"
        assert product.collection_category == "T1"

    def test_groups_are_unwrapped(self, tmp_path):
        directory = write_product(tmp_path, "json")
        source = open_product(directory, "Landsat")
        groups = read_metadata_groups(source, PurePosixPath(f"{PRODUCT_ID}_MTL.json"))
        assert "LANDSAT_METADATA_FILE" not in groups
        assert "PRODUCT_CONTENTS" in groups


class TestGroupScoping:
    """Level-2 values must win over the Level-1 record's identical key names."""

    @pytest.mark.parametrize("syntax", ["xml", "json", "txt"])
    def test_level_comes_from_product_contents(self, tmp_path, syntax):
        # LEVEL1_PROCESSING_RECORD also holds PROCESSING_LEVEL, as "L1TP".
        assert read_product(write_product(tmp_path, syntax)).level == "L2SP"

    @pytest.mark.parametrize("syntax", ["xml", "json", "txt"])
    def test_product_id_comes_from_product_contents(self, tmp_path, syntax):
        product = read_product(write_product(tmp_path, syntax))
        assert product.product_id == PRODUCT_ID
        assert product.product_id != L1_PRODUCT_ID

    @pytest.mark.parametrize("syntax", ["xml", "json", "txt"])
    def test_scaling_comes_from_the_level2_group(self, tmp_path, syntax):
        # LEVEL1_RADIOMETRIC_RESCALING carries top-of-atmosphere coefficients
        # under the same key names; picking those up would silently produce
        # the wrong physical values.
        product = read_product(write_product(tmp_path, syntax))
        assert product.reflectance_scaling["SR_B4"] == (2.75e-05, -0.2)
        assert product.reflectance_scaling["SR_B4"] != (2.0000e-05, -0.1)

    def test_temperature_scaling_is_separate_from_reflectance(self, tmp_path):
        product = read_product(write_product(tmp_path))
        assert product.temperature_scaling == {"ST_B10": (0.00341802, 149.0)}
        assert "ST_B10" not in product.reflectance_scaling


class TestAwkwardValues:
    """The spellings real products use, which naive parsing gets wrong."""

    def test_seven_fractional_digits_are_truncated_not_rejected(self, tmp_path):
        product = read_product(write_product(tmp_path))
        assert product.acquired == dt.datetime(
            2026, 8, 22, 10, 4, 22, 118358, tzinfo=dt.timezone.utc
        )
        assert product.acquired.tzinfo is not None

    def test_unpadded_wrs_row_is_normalised(self, tmp_path):
        # The JSON syntax writes "28" where the text syntax writes "028".
        assert read_product(write_product(tmp_path, row="28")).wrs_row == "028"

    def test_band_files_are_keyed_by_token(self, tmp_path):
        product = read_product(write_product(tmp_path))
        assert product.band_files["SR_B4"] == f"{PRODUCT_ID}_SR_B4.TIF"
        assert product.band_files["ST_B10"] == f"{PRODUCT_ID}_ST_B10.TIF"
        assert product.band_files["QA_PIXEL"] == f"{PRODUCT_ID}_QA_PIXEL.TIF"
        # The metadata file itself is listed too, but is not a band.
        assert not any(name.endswith("MTL.txt") for name in product.band_files.values())


class TestRefusals:
    """Unsupported products fail with a message naming the problem."""

    @pytest.mark.parametrize("level", ["L1TP", "L1GT", "L1GS"])
    def test_level1_is_refused(self, tmp_path, level):
        with pytest.raises(ValidationError, match="Level-1 is not supported") as excinfo:
            read_product(write_product(tmp_path, level=level))
        assert "L2SP or L2SR" in str(excinfo.value)

    def test_l2sr_is_accepted(self, tmp_path):
        # Reflectance-only products are valid Level-2, just without thermal.
        assert read_product(write_product(tmp_path, level="L2SR")).level == "L2SR"

    def test_unknown_level_is_refused(self, tmp_path):
        with pytest.raises(ValidationError, match="unrecognised Landsat processing level"):
            read_product(write_product(tmp_path, level="L9XX"))

    def test_unknown_mission_is_refused(self, tmp_path):
        with pytest.raises(ValidationError, match="unrecognised Landsat mission"):
            read_product(write_product(tmp_path, mission_prefix="LX99"))

    @pytest.mark.parametrize("prefix", ["LT04", "LT05", "LE07", "LC08", "LC09"])
    def test_supported_missions_are_recognised(self, tmp_path, prefix):
        product = read_product(write_product(tmp_path, mission_prefix=prefix))
        assert product.mission == int(prefix[2:])


class TestOverrideAndDamage:
    """`level=` asserts; damaged metadata fails clearly."""

    def test_agreeing_override_accepted(self, tmp_path):
        assert read_product(write_product(tmp_path), level="L2SP").level == "L2SP"

    def test_contradicting_override_rejected(self, tmp_path):
        with pytest.raises(ValidationError, match="contradicts the product"):
            read_product(write_product(tmp_path), level="L2SR")

    def test_missing_level_rejected(self, tmp_path):
        groups = spec()
        del groups["PRODUCT_CONTENTS"]["PROCESSING_LEVEL"]
        directory = tmp_path / PRODUCT_ID
        directory.mkdir()
        (directory / f"{PRODUCT_ID}_MTL.xml").write_text(as_xml(groups))
        with pytest.raises(ValidationError, match="states no PROCESSING_LEVEL"):
            read_product(directory)

    def test_missing_required_field_names_its_group(self, tmp_path):
        groups = spec()
        del groups["IMAGE_ATTRIBUTES"]["WRS_PATH"]
        directory = tmp_path / PRODUCT_ID
        directory.mkdir()
        (directory / f"{PRODUCT_ID}_MTL.xml").write_text(as_xml(groups))
        with pytest.raises(ValidationError, match="no WRS_PATH in IMAGE_ATTRIBUTES"):
            read_product(directory)

    def test_unparseable_xml_rejected(self, tmp_path):
        directory = write_product(tmp_path, "xml")
        (directory / f"{PRODUCT_ID}_MTL.xml").write_text("<not-closed>")
        with pytest.raises(ValidationError, match="is not readable"):
            read_product(directory)

    def test_unparseable_json_rejected(self, tmp_path):
        directory = write_product(tmp_path, "json")
        (directory / f"{PRODUCT_ID}_MTL.json").write_text("{oops")
        with pytest.raises(ValidationError, match="is not readable"):
            read_product(directory)

    def test_non_numeric_scaling_rejected(self, tmp_path):
        groups = spec()
        groups["LEVEL2_SURFACE_REFLECTANCE_PARAMETERS"]["REFLECTANCE_MULT_BAND_4"] = "lots"
        directory = tmp_path / PRODUCT_ID
        directory.mkdir()
        (directory / f"{PRODUCT_ID}_MTL.xml").write_text(as_xml(groups))
        with pytest.raises(ValidationError, match="are not numbers"):
            read_product(directory)

    def test_unreadable_time_rejected(self, tmp_path):
        groups = spec()
        groups["IMAGE_ATTRIBUTES"]["SCENE_CENTER_TIME"] = "half past ten"
        directory = tmp_path / PRODUCT_ID
        directory.mkdir()
        (directory / f"{PRODUCT_ID}_MTL.xml").write_text(as_xml(groups))
        with pytest.raises(ValidationError, match="could not read the acquisition time"):
            read_product(directory)

    def test_mult_without_a_matching_add_is_skipped(self, tmp_path):
        groups = spec()
        del groups["LEVEL2_SURFACE_REFLECTANCE_PARAMETERS"]["REFLECTANCE_ADD_BAND_4"]
        directory = tmp_path / PRODUCT_ID
        directory.mkdir()
        (directory / f"{PRODUCT_ID}_MTL.xml").write_text(as_xml(groups))
        assert read_product(directory).reflectance_scaling == {}
