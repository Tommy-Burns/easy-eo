"""Tests for Landsat product identification (eeo/io/_landsat.py).

One canonical description of a product is rendered into all three metadata
syntaxes, so "the three agree" is asserted rather than assumed. The group
layout, key names, and value spellings follow real USGS output, including the
parts most likely to catch a parser out: a Level-1 section carrying the same
key names as the Level-2 one with different values, a scene centre time written
with seven fractional digits, and a WRS row that is zero-padded in the text
syntax but not in the JSON.
"""

import datetime as dt
import json

import pytest

from eeo.core.exceptions import ValidationError
from eeo.io._landsat import find_metadata, read_metadata_groups, read_product

PRODUCT_ID = "LC09_L2SP_193028_20260822_20260823_02_T1"
L1_PRODUCT_ID = "LC09_L1TP_193028_20260822_20260823_02_T1"


def spec(*, level="L2SP", product_id=PRODUCT_ID, row="028", mission_prefix=None):
    """Describe a product as groups of key-value pairs."""
    if mission_prefix:
        product_id = mission_prefix + product_id[4:]
    return {
        "PRODUCT_CONTENTS": {
            "LANDSAT_PRODUCT_ID": product_id,
            "PROCESSING_LEVEL": level,
            "COLLECTION_NUMBER": "02",
            "COLLECTION_CATEGORY": "T1",
            "FILE_NAME_BAND_4": f"{product_id}_SR_B4.TIF",
            "FILE_NAME_BAND_ST_B10": f"{product_id}_ST_B10.TIF",
            "FILE_NAME_QUALITY_L1_PIXEL": f"{product_id}_QA_PIXEL.TIF",
            "FILE_NAME_METADATA_ODL": f"{product_id}_MTL.txt",
        },
        "IMAGE_ATTRIBUTES": {
            "SPACECRAFT_ID": "LANDSAT_9",
            "SENSOR_ID": "OLI_TIRS",
            "WRS_PATH": "193",
            "WRS_ROW": row,
            "DATE_ACQUIRED": "2026-08-22",
            # USGS writes seven fractional digits.
            "SCENE_CENTER_TIME": "10:04:22.1183580Z",
        },
        "LEVEL2_SURFACE_REFLECTANCE_PARAMETERS": {
            "REFLECTANCE_MULT_BAND_4": "2.75e-05",
            "REFLECTANCE_ADD_BAND_4": "-0.2",
        },
        "LEVEL2_SURFACE_TEMPERATURE_PARAMETERS": {
            "TEMPERATURE_MULT_BAND_ST_B10": "0.00341802",
            "TEMPERATURE_ADD_BAND_ST_B10": "149.0",
        },
        # The Level-1 record repeats the same key names with different values.
        # Everything above must win over everything here.
        "LEVEL1_PROCESSING_RECORD": {
            "LANDSAT_PRODUCT_ID": L1_PRODUCT_ID,
            "PROCESSING_LEVEL": "L1TP",
        },
        "LEVEL1_RADIOMETRIC_RESCALING": {
            "REFLECTANCE_MULT_BAND_4": "2.0000E-05",
            "REFLECTANCE_ADD_BAND_4": "-0.100000",
        },
    }


def as_odl(groups):
    lines = ["GROUP = LANDSAT_METADATA_FILE"]
    for group, members in groups.items():
        lines.append(f"  GROUP = {group}")
        lines += [f'    {key} = "{value}"' for key, value in members.items()]
        lines.append(f"  END_GROUP = {group}")
    lines += ["END_GROUP = LANDSAT_METADATA_FILE", "END"]
    return "\n".join(lines)


def as_json(groups):
    return json.dumps({"LANDSAT_METADATA_FILE": groups}, indent=2)


def as_xml(groups):
    body = "".join(
        f"<{group}>"
        + "".join(f"<{key}>{value}</{key}>" for key, value in members.items())
        + f"</{group}>"
        for group, members in groups.items()
    )
    return f"<LANDSAT_METADATA_FILE>{body}</LANDSAT_METADATA_FILE>"


RENDERERS = {"xml": as_xml, "json": as_json, "txt": as_odl}


def write_product(tmp_path, syntax="xml", **kwargs):
    """Write a product directory holding one metadata file."""
    groups = spec(**kwargs)
    name = groups["PRODUCT_CONTENTS"]["LANDSAT_PRODUCT_ID"]
    directory = tmp_path / name
    directory.mkdir(exist_ok=True)
    (directory / f"{name}_MTL.{syntax}").write_text(RENDERERS[syntax](groups))
    return directory


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

    def test_metadata_given_directly(self, tmp_path):
        directory = write_product(tmp_path, "json")
        path = directory / f"{PRODUCT_ID}_MTL.json"
        assert find_metadata(path) == path

    def test_missing_path_rejected(self, tmp_path):
        with pytest.raises(ValidationError, match="no such Landsat product"):
            find_metadata(tmp_path / "absent")

    def test_directory_without_metadata_rejected(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(ValidationError, match="holds no Landsat metadata file"):
            find_metadata(tmp_path / "empty")


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
        groups = read_metadata_groups(directory / f"{PRODUCT_ID}_MTL.json")
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
