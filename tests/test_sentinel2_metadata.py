"""Tests for Sentinel-2 product identification (eeo/io/_sentinel2.py).

The manifests here are trimmed but structurally faithful: the same namespaced
root with an unnamespaced body, the same element names, the same ``band_id``
indirection between the offset list and the spectral information list, and the
same ``HORIZONTAL_CS_CODE`` in a granule's tile manifest. Nothing is downloaded
and no image file is written — this layer never opens one.
"""

import datetime as dt

import pytest

from eeo.core.exceptions import ValidationError
from eeo.io._sentinel2 import Sentinel2Product, find_manifest, read_product

PRODUCT_URI = "S2B_MSIL2A_20240830T100559_N0511_R022_T32TPS_20240830T134009.SAFE"
GRANULE_ID = "L2A_T32TPS_A038765_20240830T100558"
IMAGE = f"GRANULE/{GRANULE_ID}/IMG_DATA/R10m/T32TPS_20240830T100559_B04_10m"

# Spectral_Information spells single-digit bands "B1"; the image files use "B01".
SPECTRAL = """
      <Spectral_Information_List>
        <Spectral_Information bandId="0" physicalBand="B1"/>
        <Spectral_Information bandId="3" physicalBand="B4"/>
        <Spectral_Information bandId="8" physicalBand="B8A"/>
        <Spectral_Information bandId="12" physicalBand="B12"/>
      </Spectral_Information_List>"""

OFFSETS = """
      <BOA_ADD_OFFSET_VALUES_LIST>
        <BOA_ADD_OFFSET band_id="0">-1000</BOA_ADD_OFFSET>
        <BOA_ADD_OFFSET band_id="3">-1000</BOA_ADD_OFFSET>
        <BOA_ADD_OFFSET band_id="8">-1000</BOA_ADD_OFFSET>
        <BOA_ADD_OFFSET band_id="12">-1000</BOA_ADD_OFFSET>
      </BOA_ADD_OFFSET_VALUES_LIST>"""


def manifest_xml(
    *,
    level="2A",
    product_type="S2MSI2A",
    baseline="05.11",
    quantification="10000",
    offsets=OFFSETS,
    start_time="2024-08-30T10:05:59.024Z",
    uri=PRODUCT_URI,
):
    """Build a product manifest with the parts under test made adjustable."""
    type_element = f"<PRODUCT_TYPE>{product_type}</PRODUCT_TYPE>" if product_type else ""
    quant = (
        f"""
      <QUANTIFICATION_VALUES_LIST>
        <BOA_QUANTIFICATION_VALUE unit="none">{quantification}</BOA_QUANTIFICATION_VALUE>
      </QUANTIFICATION_VALUES_LIST>"""
        if quantification
        else ""
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<n1:Level-{level}_User_Product
    xmlns:n1="https://psd-15.sentinel2.eo.esa.int/PSD/User_Product_Level-{level}.xsd">
  <n1:General_Info>
    <Product_Info>
      <PRODUCT_START_TIME>{start_time}</PRODUCT_START_TIME>
      <PRODUCT_URI>{uri}</PRODUCT_URI>
      <PROCESSING_LEVEL>Level-{level}</PROCESSING_LEVEL>
      {type_element}
      <PROCESSING_BASELINE>{baseline}</PROCESSING_BASELINE>
      <Product_Organisation>
        <Granule_List>
          <Granule granuleIdentifier="{GRANULE_ID}" imageFormat="JPEG2000">
            <IMAGE_FILE>{IMAGE}</IMAGE_FILE>
          </Granule>
        </Granule_List>
      </Product_Organisation>
    </Product_Info>
    <Product_Image_Characteristics>{quant}{offsets}{SPECTRAL}
    </Product_Image_Characteristics>
  </n1:General_Info>
</n1:Level-{level}_User_Product>
"""


TILE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<n1:Level-2A_Tile_ID
    xmlns:n1="https://psd-15.sentinel2.eo.esa.int/PSD/S2_PDI_Level-2A_Tile_Metadata.xsd">
  <n1:General_Info>
    <TILE_ID>S2B_OPER_MSI_L2A_TL_2BPS_20240830T134009_A038765_T32TPS_N05.11</TILE_ID>
    <SENSING_TIME>2024-08-30T10:06:21.919283Z</SENSING_TIME>
  </n1:General_Info>
  <n1:Geometric_Info>
    <Tile_Geocoding>
      <HORIZONTAL_CS_NAME>WGS84 / UTM zone 32N</HORIZONTAL_CS_NAME>
      <HORIZONTAL_CS_CODE>EPSG:32632</HORIZONTAL_CS_CODE>
      <Size resolution="10"><NROWS>10980</NROWS><NCOLS>10980</NCOLS></Size>
    </Tile_Geocoding>
  </n1:Geometric_Info>
</n1:Level-2A_Tile_ID>
"""


def write_safe(tmp_path, *, manifest_name="MTD_MSIL2A.xml", tile_xml=TILE_XML, **kwargs):
    """Write a minimal .SAFE tree and return its directory."""
    safe = tmp_path / PRODUCT_URI
    (safe / "GRANULE" / GRANULE_ID).mkdir(parents=True)
    (safe / manifest_name).write_text(manifest_xml(**kwargs))
    if tile_xml is not None:
        (safe / "GRANULE" / GRANULE_ID / "MTD_TL.xml").write_text(tile_xml)
    return safe


class TestFindManifest:
    """The manifest is found from the .SAFE directory, its parent, or directly."""

    def test_safe_directory(self, tmp_path):
        safe = write_safe(tmp_path)
        assert find_manifest(safe).name == "MTD_MSIL2A.xml"

    def test_directory_holding_a_safe(self, tmp_path):
        write_safe(tmp_path)
        assert find_manifest(tmp_path).name == "MTD_MSIL2A.xml"

    def test_manifest_given_directly(self, tmp_path):
        safe = write_safe(tmp_path)
        assert find_manifest(safe / "MTD_MSIL2A.xml").name == "MTD_MSIL2A.xml"

    def test_missing_path_rejected(self, tmp_path):
        with pytest.raises(ValidationError, match="no such Sentinel-2 product"):
            find_manifest(tmp_path / "absent.SAFE")

    def test_directory_without_a_manifest_rejected(self, tmp_path):
        with pytest.raises(ValidationError, match="holds no Sentinel-2 product manifest"):
            find_manifest(tmp_path)


class TestLevelDetection:
    """Level comes from the document, with the filename only as a fallback."""

    def test_l2a_is_read(self, tmp_path):
        product = read_product(write_safe(tmp_path))
        assert isinstance(product, Sentinel2Product)
        assert (product.level, product.product_type) == ("L2A", "S2MSI2A")

    def test_l1c_is_refused_with_an_explanation(self, tmp_path):
        safe = write_safe(
            tmp_path, manifest_name="MTD_MSIL1C.xml", level="1C", product_type="S2MSI1C"
        )
        with pytest.raises(ValidationError, match="Level-1C is not supported") as excinfo:
            read_product(safe)
        assert "download the L2A product" in str(excinfo.value)

    def test_renamed_manifest_does_not_mislead(self, tmp_path):
        # An L1C document saved under the L2A filename must still be refused:
        # the document is the authority, not the name it was given.
        safe = write_safe(tmp_path, manifest_name="MTD_MSIL2A.xml", product_type="S2MSI1C")
        with pytest.raises(ValidationError, match="Level-1C is not supported"):
            read_product(safe)

    def test_absent_product_type_falls_back_to_the_filename(self, tmp_path):
        safe = write_safe(tmp_path, product_type="")
        assert read_product(safe).level == "L2A"

    def test_unidentifiable_product_rejected(self, tmp_path):
        safe = write_safe(tmp_path, manifest_name="MTD_SOMETHING.xml", product_type="")
        with pytest.raises(ValidationError, match="identifies no\n?\\s*processing level"):
            read_product(safe / "MTD_SOMETHING.xml")


class TestLevelOverride:
    """`level=` asserts rather than overrides whenever the manifest can speak."""

    def test_agreeing_override_accepted(self, tmp_path):
        assert read_product(write_safe(tmp_path), level="L2A").level == "L2A"

    def test_contradicting_override_rejected(self, tmp_path):
        safe = write_safe(tmp_path, manifest_name="MTD_MSIL1C.xml", product_type="S2MSI1C")
        with pytest.raises(ValidationError, match="contradicts the product"):
            read_product(safe, level="L2A")

    def test_override_decides_when_the_manifest_is_silent(self, tmp_path):
        safe = write_safe(tmp_path, manifest_name="MTD_SOMETHING.xml", product_type="")
        assert read_product(safe / "MTD_SOMETHING.xml", level="l2a").level == "L2A"

    def test_unknown_level_value_rejected(self, tmp_path):
        with pytest.raises(ValidationError, match="level must be one of"):
            read_product(write_safe(tmp_path), level="L3B")


class TestMetadata:
    """The fields a loader needs are read from both manifests."""

    def test_identity_and_baseline(self, tmp_path):
        product = read_product(write_safe(tmp_path))
        assert product.tile_id == "T32TPS"
        assert product.processing_baseline == "05.11"
        assert product.crs == "EPSG:32632"

    def test_sensing_time_prefers_the_tile_manifest(self, tmp_path):
        # The product manifest carries the datatake's start (10:05:59); the
        # tile manifest carries this granule's own time (10:06:21). Real
        # granule manifests write microseconds, so the fixture does too.
        product = read_product(write_safe(tmp_path))
        assert product.sensing_time == dt.datetime(
            2024, 8, 30, 10, 6, 21, 919283, tzinfo=dt.timezone.utc
        )
        assert product.sensing_time.tzinfo is not None

    def test_millisecond_precision_also_parses(self, tmp_path):
        # The product manifest writes milliseconds where the granule writes
        # microseconds; Python 3.10's fromisoformat accepts 3 or 6 digits and
        # nothing between, so both spellings are exercised.
        tile = TILE_XML.replace("<SENSING_TIME>2024-08-30T10:06:21.919283Z</SENSING_TIME>", "")
        product = read_product(write_safe(tmp_path, tile_xml=tile))
        assert product.sensing_time == dt.datetime(
            2024, 8, 30, 10, 5, 59, 24000, tzinfo=dt.timezone.utc
        )

    def test_quantification_and_image_files(self, tmp_path):
        product = read_product(write_safe(tmp_path))
        assert product.quantification_value == 10000.0
        assert product.image_files == (IMAGE,)
        assert product.granule is not None and product.granule.name == GRANULE_ID

    def test_band_offsets_use_image_file_spelling(self, tmp_path):
        # band_id 0 is physicalBand "B1", which the image files call "B01".
        product = read_product(write_safe(tmp_path))
        assert product.band_offsets == {
            "B01": -1000.0,
            "B04": -1000.0,
            "B8A": -1000.0,
            "B12": -1000.0,
        }

    def test_older_baseline_has_no_offsets(self, tmp_path):
        # Products before baseline 04.00 carry no offset element at all, which
        # means they have none — not that theirs is unknown.
        product = read_product(write_safe(tmp_path, baseline="02.12", offsets=""))
        assert product.band_offsets == {}
        assert product.quantification_value == 10000.0


class TestMalformedProducts:
    """Damaged metadata fails with a message naming what is wrong."""

    def test_unparseable_xml_rejected(self, tmp_path):
        safe = write_safe(tmp_path)
        (safe / "MTD_MSIL2A.xml").write_text("<not-closed>")
        with pytest.raises(ValidationError, match="is not readable XML"):
            read_product(safe)

    def test_missing_tile_manifest_rejected(self, tmp_path):
        safe = write_safe(tmp_path, tile_xml=None)
        with pytest.raises(ValidationError, match="states no projection"):
            read_product(safe)

    def test_unparseable_tile_manifest_rejected(self, tmp_path):
        safe = write_safe(tmp_path, tile_xml="<nope")
        with pytest.raises(ValidationError, match="MTD_TL.xml is not readable XML"):
            read_product(safe)

    def test_non_numeric_quantification_rejected(self, tmp_path):
        safe = write_safe(tmp_path, quantification="not-a-number")
        with pytest.raises(ValidationError, match="BOA_QUANTIFICATION_VALUE is not a number"):
            read_product(safe)

    def test_non_numeric_offset_rejected(self, tmp_path):
        bad = OFFSETS.replace(">-1000<", ">minus one thousand<", 1)
        with pytest.raises(ValidationError, match="is not a number"):
            read_product(write_safe(tmp_path, offsets=bad))

    def test_unreadable_timestamp_rejected(self, tmp_path):
        tile = TILE_XML.replace("2024-08-30T10:06:21.919283Z", "last Tuesday")
        safe = write_safe(tmp_path, tile_xml=tile)
        with pytest.raises(ValidationError, match="could not read the acquisition time"):
            read_product(safe)

    def test_absent_quantification_is_recorded_as_unknown(self, tmp_path):
        product = read_product(write_safe(tmp_path, quantification=""))
        assert product.quantification_value is None

    def test_offsets_without_a_band_id_are_skipped(self, tmp_path):
        # A real product always attributes an offset to a band; one that does
        # not is unattributable, so it is dropped rather than guessed at.
        stray = OFFSETS.replace('<BOA_ADD_OFFSET band_id="0">', "<BOA_ADD_OFFSET>", 1)
        product = read_product(write_safe(tmp_path, offsets=stray))
        assert "B01" not in product.band_offsets
        assert product.band_offsets["B04"] == -1000.0

    def test_offsets_for_an_unlisted_band_are_skipped(self, tmp_path):
        # band_id 99 appears in no Spectral_Information entry, so there is no
        # band name to key it by.
        unlisted = OFFSETS.replace('band_id="0"', 'band_id="99"', 1)
        product = read_product(write_safe(tmp_path, offsets=unlisted))
        assert set(product.band_offsets) == {"B04", "B8A", "B12"}

    def test_product_without_any_acquisition_time_rejected(self, tmp_path):
        tile = TILE_XML.replace("<SENSING_TIME>2024-08-30T10:06:21.919283Z</SENSING_TIME>", "")
        safe = write_safe(tmp_path, tile_xml=tile, start_time="")
        with pytest.raises(ValidationError, match="states no acquisition time"):
            read_product(safe)
